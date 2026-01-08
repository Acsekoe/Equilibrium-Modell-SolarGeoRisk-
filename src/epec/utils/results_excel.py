from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

DECIMALS = 3

def _round_excel_value(v: Any, ndigits: int = DECIMALS) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, Real):
        return round(v, ndigits)
    return v


def _project_root_from_here(here: Path) -> Path:
    """
    Find project root by walking up until we find a folder that contains 'src'.
    Expected layout:
      <root>/src/epec/utils/results_excel.py
      <root>/experiments/run_small.py
    """
    here = here.resolve()
    for parent in [here] + list(here.parents):
        if (parent / "src").is_dir():
            return parent
    # fallback: 3 levels up (utils -> epec -> src -> root)
    return here.parents[3]


def ensure_results_dir(results_dir: Optional[Path] = None) -> Path:
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    root = _project_root_from_here(Path(__file__))
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


class _TeeStdout(io.TextIOBase):
    def __init__(self, real_stdout, buffer: io.StringIO):
        super().__init__()
        self._real = real_stdout
        self._buf = buffer

    def write(self, s: str) -> int:
        n1 = self._real.write(s)
        self._real.flush()
        n2 = self._buf.write(s)
        return max(n1, n2)

    def flush(self) -> None:
        self._real.flush()
        self._buf.flush()


@contextmanager
def capture_stdout(tee: bool = True) -> Iterable[io.StringIO]:
    """
    Capture everything printed to stdout during the block.
    If tee=True, still prints to console while capturing.
    """
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = _TeeStdout(old, buf) if tee else buf
        yield buf
    finally:
        sys.stdout = old


def _safe_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(x)


def _autosize(ws, max_width: int = 80) -> None:
    # crude but good enough: compute max string length per column
    col_widths: Dict[int, int] = {}
    for row in ws.iter_rows(values_only=True):
        for j, v in enumerate(row, start=1):
            s = "" if v is None else str(v)
            col_widths[j] = min(max(col_widths.get(j, 0), len(s)), max_width)

    for j, w in col_widths.items():
        ws.column_dimensions[get_column_letter(j)].width = max(10, min(w + 2, max_width))


def _make_header(ws, row_idx: int = 1) -> None:
    bold = Font(bold=True)
    for cell in ws[row_idx]:
        cell.font = bold
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = ws["A2"]


def _write_kv_sheet(wb: Workbook, name: str, items: Sequence[Tuple[str, Any]]) -> None:
    ws = wb.create_sheet(title=name)
    ws.append(["key", "value"])
    for k, v in items:
        val = v if isinstance(v, (int, float, str, bool)) or v is None else _safe_json(v)
        ws.append([k, _round_excel_value(val)])
    _make_header(ws)
    ws.auto_filter.ref = ws.dimensions
    _autosize(ws)


def _flatten_theta(sets: Any, theta: Any) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    # Expect: sets.R is iterable of regions, sets.RR iterable of (e,r) arcs
    R = list(getattr(sets, "R"))
    RR = list(getattr(sets, "RR"))

    q_rows = [["region", "q_man"]]
    for r in R:
        q_rows.append([r, float(theta.q_man[r])])

    d_rows = [["region", "d_offer"]]
    for r in R:
        d_rows.append([r, float(theta.d_offer[r])])

    tau_rows = [["exporter", "importer", "tau"]]
    for (e, r) in sorted(RR):
        tau_rows.append([e, r, float(theta.tau[(e, r)])])

    return q_rows, d_rows, tau_rows


def _write_table(ws, table: List[List[Any]]) -> None:
    for row in table:
        ws.append([_round_excel_value(v) for v in row])
    _make_header(ws)
    ws.auto_filter.ref = ws.dimensions
    _autosize(ws)


def _write_history_sheet(wb: Workbook, name: str, sets: Any, hist: List[dict]) -> None:
    ws = wb.create_sheet(title=name)

    R = list(getattr(sets, "R"))
    # fixed “nice” columns first
    base_cols = [
        "iter", "region", "accepted", "status", "term",
        "ulp_obj", "llp_obj",
        "br_q_man", "br_d_offer",
    ]
    # expand lambda into columns if present
    lambda_cols = [f"lambda_{r}" for r in R]

    # collect any extra keys (stringify dict/list values)
    extra_keys = set()
    for row in hist:
        extra_keys |= set(row.keys())
    # remove ones we already plan to show / expand
    for k in base_cols + ["lambda"]:
        extra_keys.discard(k)

    # keep br_tau_in if present but stringify
    # (and any other extras)
    extra_cols = sorted(extra_keys)

    header = base_cols + lambda_cols + extra_cols
    ws.append(header)

    for row in hist:
        out = []
        for k in base_cols:
            v = row.get(k, None)
            if isinstance(v, (dict, list, tuple)):
                v = _safe_json(v)
            out.append(v)

        lam = row.get("lambda", {}) or {}
        for r in R:
            v = lam.get(r, None)
            out.append(float(v) if isinstance(v, (int, float)) else v)

        for k in extra_cols:
            v = row.get(k, None)
            if isinstance(v, (dict, list, tuple)):
                v = _safe_json(v)
            out.append(v)

        ws.append([_round_excel_value(v) for v in out])

    _make_header(ws)
    ws.auto_filter.ref = ws.dimensions
    _autosize(ws)


def save_run_xlsx(
    *,
    run_name: str,
    sets: Any,
    run_cfg: Dict[str, Any],
    ipopt_options: Optional[Dict[str, Any]],
    theta_star: Any,
    hist: List[dict],
    raw_stdout: Optional[str] = None,
    results_dir: Optional[Path] = None,
    include_raw_log: bool = True,
) -> Path:
    out_dir = ensure_results_dir(results_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # short filename (don’t explode path lengths on Windows)
    eps = run_cfg.get("eps", None)
    eps_u = run_cfg.get("eps_u", None)
    max_iter = run_cfg.get("max_iter", None)
    tol = run_cfg.get("tol", None)
    fname = f"{run_name}_{ts}_eps={eps}_eps_u={eps_u}_tol={tol}_max_iter={max_iter}.xlsx"
    path = out_dir / fname

    wb = Workbook()
    # remove default "Sheet"
    wb.remove(wb.active)

    # Config sheet
    cfg_items = list(run_cfg.items())
    if ipopt_options:
        for k, v in ipopt_options.items():
            cfg_items.append((f"ipopt.{k}", v))
    _write_kv_sheet(wb, "Config", cfg_items)

    # Final theta sheets
    q_rows, d_rows, tau_rows = _flatten_theta(sets, theta_star)

    ws_q = wb.create_sheet("FinalTheta_q_man")
    _write_table(ws_q, q_rows)

    ws_d = wb.create_sheet("FinalTheta_d_offer")
    _write_table(ws_d, d_rows)

    ws_t = wb.create_sheet("FinalTheta_tau")
    _write_table(ws_t, tau_rows)

    # History
    _write_history_sheet(wb, "History", sets, hist)

    # Raw log (line-by-line)
    if include_raw_log and raw_stdout is not None:
        ws = wb.create_sheet("RawLog")
        ws.append(["line_no", "text"])
        for i, line in enumerate(raw_stdout.splitlines(), start=1):
            ws.append([i, line])
        _make_header(ws)
        ws.auto_filter.ref = ws.dimensions
        _autosize(ws, max_width=120)

    wb.save(path)
    return path
