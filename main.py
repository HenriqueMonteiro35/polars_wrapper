import numpy as np
import pandas as pd
import polars as pl
import re

def parse_query(query: str) -> pl.Expr:
    """
    Parse a simple query string into a Polars expression.
    Supports operators: >, <, >=, <=, ==, !=, &, |.
    Example: "a > 2 | b == 'w'" -> (pl.col("a") > 2) | (pl.col("b") == 'w')
    """
    # Split on binary operators while keeping them
    tokens = re.split(r"(\s*\|\s*|\s*&\s*)", query)

    exprs = []
    ops = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        if tok in ("|", "&"):
            ops.append(tok)
        else:
            # Simple binary condition: col op value
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+)", tok)
            if not m:
                raise ValueError(f"Cannot parse condition: {tok}")

            col, op, val = m.groups()

            # Try to interpret val as number, else strip quotes
            if re.match(r"^-?\d+(\.\d+)?$", val):
                val = float(val) if "." in val else int(val)
            else:
                val = val.strip("\"'")

            # Build the polars expression
            if op == "==":
                exprs.append(pl.col(col) == val)
            elif op == "!=":
                exprs.append(pl.col(col) != val)
            elif op == ">":
                exprs.append(pl.col(col) > val)
            elif op == "<":
                exprs.append(pl.col(col) < val)
            elif op == ">=":
                exprs.append(pl.col(col) >= val)
            elif op == "<=":
                exprs.append(pl.col(col) <= val)

    # Combine expressions with | and &
    expr = exprs[0]
    for op, rhs in zip(ops, exprs[1:]):
        if op == "|":
            expr = expr | rhs
        elif op == "&":
            expr = expr & rhs

    return expr



class PandasLikePolars(pl.DataFrame):
    def __getitem__(self, item):
       # Case 1: Boolean mask as Polars series, Pandas series or NumPy array
        if isinstance(item, (pl.Series, pd.Series, np.ndarray)) and item.dtype in [bool, pl.Boolean]:
            return self.filter(item)

        # Case 2: Boolean mask as list/tuple
        if isinstance(item, (list, tuple)) and all(isinstance(x, bool) for x in item):
            return self.filter(pl.Series(item))

        # Case 3: string filter like "col_a > 2 | col_b == 'w'"
        if isinstance(item, str) and item not in df.columns:
            expr = parse_query(item)
            return self.filter(expr)

        # Case 4: tuple/list of string filters (assumes AND operation)
        if isinstance(item, (tuple, list)) and all(isinstance(x, str) for x in item):
            if not all(x in self.columns for x in item):
                return self[" & ".join(item)]

        # Default: fall back to polars native __getitem__
        return super().__getitem__(item)

# Example usage
df = pl.DataFrame({
    "a": [0, 1, 2, 3, 4, 5, 3, 1],
    "b": ["x", "y", "z", "b", "e", "b", "w", "w"]
})

df = PandasLikePolars(df)

print(df[df["a"] > 2])
print(df["a > 2"])
print(df["a <= 2 & b != 'y' & b != 'x'"])
print(df["a < 3", "b != 'w'", "a != 0"])
df["a", "b"]
