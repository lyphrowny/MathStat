from collections import namedtuple
from typing import Iterable

caplab = namedtuple("caplab", "cap lab".split())


def _wrap(s, wrapper="$"):
    if not s:
        wrapper = ""
    return f"{wrapper}{s}{wrapper}"


def _trunc(num, tol):
    if isinstance(num, float):
        return f"{num:.{tol}f}"
    elif isinstance(num, int):
        return f"{num}"
    elif isinstance(num, str):
        return _wrap(num)
    elif isinstance(num, tuple):
        return "; ".join(map(lambda x: _trunc(x, tol), num))
    else:
        raise RuntimeError(f"no truncate for this type {type(num)}")


class SubTable:
    def __init__(self, headers, lines, footer=None):
        self._hdr = self._to_mat(headers)
        self._lines = self._to_mat(lines)
        self._footer = self._to_mat(footer) if footer else footer

    @staticmethod
    def _to_mat(obj):
        if not isinstance(obj[0], Iterable) or isinstance(obj[0], str):
            return [obj]
        return obj

    @staticmethod
    def _gen_line(line_vals, func, spacer=" " * 8, new_line="\\\\"):
        return f"{spacer}{' & '.join(map(func, line_vals))} {new_line}"

    def get_subtbl(self, spacer=" " * 8, tol=6, break_line="\\hline"):
        hdrs = []
        for hs in self._hdr:
            hdrs.append(self._gen_line(hs, _wrap))
            hdrs.append(f"{spacer}{break_line}")

        lines = [self._gen_line(line, lambda x: _trunc(x, tol), spacer=spacer) for line in self._lines]
        hdrs.extend(lines)
        if self._footer:
            footer = [self._gen_line(line, lambda x: _trunc(x, tol), spacer=spacer) for line in self._footer]
            hdrs.append(f"{spacer}{break_line}")
            hdrs.extend(footer)
        return "\n".join(hdrs)


class Table:
    def __init__(self, n_cols, subtbls, caplab):
        self._n_cols = n_cols
        self._subtbls = subtbls if isinstance(subtbls, list) else [subtbls]
        self._caplab = caplab

    def gen_table(self, tol=6, spacer=" " * 8, break_line="\\hline"):
        end_of_subtbl = f"\n{spacer}\\multicolumn{{{self._n_cols}}}{{c}}{{}} \\\\\n"
        tab = f"""\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{{'|'.join('c' * self._n_cols)}}}
{end_of_subtbl.join(subtbl.get_subtbl(tol=tol, spacer=spacer, break_line=break_line) for subtbl in self._subtbls)}
    \\end{{tabular}}
    \\caption{{{self._caplab.cap}}}
    \\label{{{self._caplab.lab}}}
\\end{{table}}"""
        return tab
