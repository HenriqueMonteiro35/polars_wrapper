import polars as pl
import numpy as np
import pandas as pd
from IPython.display import display

from queries import format_query

class ResearchFrame(pl.DataFrame):
    def __init__(self, *args, **kwargs):
        self.logs = kwargs.pop("logs", list())
        super().__init__(*args, **kwargs)


    # ----------- Core Indexing -----------
    def __getitem__(self, item):
        print("__getitem__", item, "|")
        # Case 1: Boolean mask as Polars series, Pandas series, or NumPy array
        if isinstance(item, (pl.Series, pd.Series, np.ndarray)) and item.dtype in [bool, pl.Boolean]:
            return self.filter(item)

        # Case 2: Boolean mask as list/tuple
        if isinstance(item, (list, tuple)) and all(isinstance(x, bool) for x in item):
            return self.filter(pl.Series(item))

        # Case 3: String query expression
        if isinstance(item, str) and item not in self.columns:# and self._looks_like_expression(item):
            expr = self._eval_query(item)
            return self.filter(expr)

        # Default: fall back to Polars
        return super().__getitem__(item)


    def filter(self, *args, **kwargs):
        logs = self.logs + [[*args, kwargs]]
        return ResearchFrame(super().filter(*args, **kwargs), logs=logs)


    # ----------- Pandas-like Assignment -----------
    def __setitem__(self, key, value):
        print("__setitem__", key, value, "|")
        if not isinstance(key, str):
            raise TypeError("Column name must be a string")

        # If RHS is an expression: evaluate it and make it a column
        if isinstance(value, pl.Expr):
            new_col = value.alias(key)
            new_df = self.with_columns(new_col).alias(key)

        # If RHS is a polars Series, just make it a column
        elif isinstance(value, pl.Series):
            new_df = self.with_columns(value.rename(key))

        # If RHS is a list, np.ndarray, pd.Series or scalar, make it Polars series first
        elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            new_df = self.with_columns(pl.Series(key, value))

        else:
            # Assume scalar broadcast
            new_df = self.with_columns(pl.lit(value).alias(key))

        # Mutate in place (safe, zero-copy under Arrow semantics)
        self._df = new_df._df


    def _eval_query(self, query: str) -> pl.Expr:
        expr = format_query(query)
        print("_eval_query", query, expr)

        # 4) safe eval environment: expose only pl, pl.col, pl.lit, and column-name aliases (already substituted,
        # but exposing mapping is harmless and convenient)
        safe_locals = {name: pl.col(name) for name in self.columns}
        safe_locals.update({"pl": pl, "col": pl.col, "lit": pl.lit})

        try:
            expr = eval(expr, {"__builtins__": {}}, safe_locals)
        except Exception as e:
            raise ValueError(f"Failed to evaluate query '{query}': {e}")

        if not isinstance(expr, pl.Expr):
            raise ValueError(f"Query did not produce a Polars expression: {query!r}")

        return expr


    def __call__(self):
        if self.logs:
            last_filter = [expr.__str__() for expr in self.logs[-1]]
            filter_str = " & ".join(last_filter)
            filter_str = filter_str.replace("[", "").replace("]", "").replace(" & {}", "")
            print(f"===== {filter_str} =====")
        display(self)
        return self
