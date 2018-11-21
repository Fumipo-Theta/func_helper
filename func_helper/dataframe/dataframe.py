import pandas as pd
from typing import List, Tuple, Iterator, TypeVar
import func_helper.func_helper.iterator as it

T = TypeVar('T')


def time_range(f):
    def wrap(*arg, **kwargs):
        return f(*pd.to_datetime(arg), **kwargs)
    return wrap


def right_open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower and upper:
            return df[(lower <= dt) & (dt < upper)]
        elif lower:
            return df[lower <= dt]
        elif upper:
            return df[dt < upper]
        else:
            return df
    return apply


def left_open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower and upper:
            return df[(lower < dt) & (dt <= upper)]
        elif lower:
            return df[lower < dt]
        elif upper:
            return df[dt <= upper]
        else:
            return df
    return apply


def open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower and upper:
            return df[(lower < dt) & (dt < upper)]
        elif lower:
            return df[lower < dt]
        elif upper:
            return df[dt < upper]
        else:
            return df
    return apply


def close_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower and upper:
            return df[(lower < dt) & (dt < upper)]
        elif lower:
            return df[lower < dt]
        elif upper:
            return df[dt < upper]
        else:
            return df
    return apply


def filter_between(lower, upper, open_left=False, open_right=True):
    if open_left and open_right:
        return open_interval(lower, upper)
    elif open_left:
        return left_open_interval(lower, upper)
    elif open_right:
        return right_open_interval(lower, upper)
    else:
        return close_interval(lower, upper)


def setTimeSeriesIndex(*columnName):
    """
    Set time series index to pandas.DataFrame
    datatime object is created from a column or two columns of
        date and time.
    The column "datetime" is temporally created,
        then it is set as index.

    Parameters
    ----------
    columnName: Union[str,List[str]]
        They can be multiple strings of column name or
            list of strings.

    Returns
    -------
    Callable[[pandas.DataFrame], pandas.DataFrame]

    """

    columns = columnName[0] if type(
        columnName[0]) == list else list(columnName)

    def f(df: pd.DataFrame)->pd.DataFrame:
        datetime_str = it.reducing(
            lambda a, e: a+" "+e)("")([df[c] for c in columns])

        df["datetime"] = pd.to_datetime(datetime_str)

        df.set_index("datetime", inplace=True)
        return df
    return f
