import re
import html
import numpy as np
import polars as pl
from bisect import bisect_right

# CMAP = {
#     "string":   "rgb(160, 255, 160)", # light green
#     "int":      "rgb(150, 225, 255)", # light blue
#     "float":    "rgb(200, 175, 255)", # light purple
#     "datetime": "rgb(255, 150, 175)", # light red
#     "bool":     "rgb(255, 200, 150)", # light orange
#     "other":    "rgb(255, 255, 150)", # light yellow
# }
# CMAP = {
#     "string":   "rgba(230, 255, 230)", # light green
#     "int":      "rgb(225, 255, 255)", # light blue
#     "float":    "rgb(240, 225, 255)", # light purple
#     "datetime": "rgb(255, 215, 255)", # light red
#     "bool":     "rgb(255, 230, 215)", # light orange
#     "other":    "rgb(255, 255, 215)", # light yellow
# }
CMAP = {
    "string":   "rgb(225, 255, 215)", # light green
    "int":      "rgb(200, 255, 255)", # light blue
    "float":    "rgb(240, 220, 255)", # light purple
    "datetime": "rgb(255, 210, 255)", # light red
    "bool":     "rgb(255, 230, 215)", # light orange
    "other":    "rgb(255, 255, 215)", # light yellow
}

def dtype_to_key(dt):
    s = str(dt).lower()
    if "utf" in s or "str" in s or "object" in s:
        return "string"
    if "int" in s or "uint" in s:
        return "int"
    if "float" in s or "decimal" in s:
        return "float"
    if "date" in s or "time" in s or "datetime" in s or "duration" in s:
        return "datetime"
    if "bool" in s:
        return "bool"
    return "other"


def percentile_of(arr, v):
    """Return empirical percentile in [0,1] (fraction of values <= v)."""
    if (v is None) or not len(arr):
        return 0.5
    pos = bisect_right(arr, v)
    return pos / len(arr)


def get_rgba(color_str: str, alpha: float) -> str:
    r, g, b = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color_str).groups()
    return f"rgba({r},{g},{b},{alpha:.2f})"


def format_df(df: pl.DataFrame, header_color="#14215a", row_alpha=0.85, font_weight_range=(200, 750),
    normal_weight=400, max_rows=50, head_tail_size=5) -> str:
    """
    Modified format_df:
    - alternating row backgrounds use alpha 1.0 / 0.9 applied to the dtype-based hex colors
    - int/float values get font-weight mapped from 400..900 according to their empirical CDF (per column)
    - preserves all original behavior otherwise (header, truncation, booleans colored green/red, etc.)
    """
    cols = df.columns
    dtypes = df.dtypes
    # map dtype -> key used by cmap (expects dtype_to_key and cmap available in outer scope)
    dtype_keys = [dtype_to_key(dt) for dt in dtypes]


    # Truncation policy (conservative mimic of Polars): show all if small,
    # otherwise show head_n and tail_n separated by ellipsis row.
    total = df.height
    if total <= max_rows:
        head_n, tail_n = total, 0
    else:
        head_n, tail_n = head_tail_size, head_tail_size

    head_rows = list(df.head(head_n).rows()) if head_n else []
    tail_rows = list(df.tail(tail_n).rows()) if tail_n else []

    # Build row blocks with optional ellipsis row
    row_blocks = []
    row_blocks.extend(head_rows)
    if head_n + tail_n < total:
        # insert an ellipsis row marker (None indicates ellipsis)
        row_blocks.append(None)
        row_blocks.extend(tail_rows)

    min_weight, max_weight = font_weight_range

    css = f"""
    <style>
    table.pl-small {{ border-collapse: collapse; font-family: "Segoe UI"; font-size: 15px;}}
    table.pl-small th {{ background: {header_color}; color: white; padding: 6px 8px; text-align: center;}}
    table.pl-small td {{padding: 6px 6px; border: 1px solid #eee; color: black; text-align: center}}
    /* ellipsis row style */
    table.pl-small tbody tr.ellipsis td {{ background: {header_color}; color:white; font-size:18px; padding: 0px 0px; }}
    .dtype {{ color: white; font-size: 12px; display:block; margin-top:4px; }}
    .col-name {{ display:block; font-weight:{max_weight}; }}
    </style>
    """

    # Header: column name and dtype on new line (dtype in smaller text)
    dt_types_str = [str(dt).replace("time_unit=", "").replace("time_zone=", "") for dt in dtypes]
    header_html = "<tr>" + "".join(
        f"<th><span class='col-name'>{html.escape(c)}</span><span class='dtype'> \
                {html.escape(dt)}</span></th>"
        for c, dt in zip(cols, dt_types_str)
    ) + "</tr>"

    # Rows
    rows_html = []
    for view_idx, r in enumerate(row_blocks):
        # A None object is used to flag the ellipsis role
        if r is None:
            rows_html.append(f"<tr class='ellipsis'><td colspan='{len(cols)}'>…</td></tr>")

        else:
            # alpha alternation for intercalating shading of rows
            alpha = 1.0 if (view_idx % 2) == 0 else row_alpha

            # data row
            cells = []
            for (val, key, colname) in zip(r, dtype_keys, cols):
                txt = html.escape(str(val))
                text_color = "black"
                weight = normal_weight

                # Show None or NaN as empty entries
                if (val is None) or (key == "float" and np.isnan(val)):
                    txt = ""

                # Boolean: print True/False as colored text (green/red)
                if key == "bool":
                    text_color = "green" if val else "red"

                # Numeric/datetime: font-weight based on empirical percentile
                elif key in ("int", "float", "datetime"):
                    arr = df[colname].drop_nulls().drop_nans().sort()
                    weight = int(min_weight + percentile_of(arr, val)*(max_weight - min_weight))

                bg = CMAP.get(key, CMAP["other"])
                bg_rgba = get_rgba(bg, alpha)
                cells.append(f"<td style='background:{bg_rgba}; color:{text_color}; font-weight:{weight};'>{txt}</td>")

            rows_html.append(f"<tr class='data-row'>{''.join(cells)}</tr>")

    shape_html = f"<div style='font-family: monospace; color:white; margin-bottom: 4px;'>SHAPE: {df.shape[0]:_} x {df.shape[1]:_}</div>"
    table = shape_html + f"<table class='pl-small'>{header_html}{''.join(rows_html)}</table>"
    return css + table
