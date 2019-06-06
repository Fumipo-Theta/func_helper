import pandas as pd
import math
import numpy as np
from typing import List, Tuple, Iterator, TypeVar, Union, Callable, Optional
from functools import reduce

T = TypeVar('T')
Number = Union[int, float, complex]

identity = lambda a: a

def time_range(f):
    def wrap(*arg, **kwargs):
        return f(*pd.to_datetime(arg), **kwargs)
    return wrap


_invalid_range = [None, pd.NaT, np.nan]


def right_open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower not in _invalid_range and upper not in _invalid_range:
            return df[(lower <= dt) & (dt < upper)]
        elif lower not in _invalid_range:
            return df[lower <= dt]
        elif upper not in _invalid_range:
            return df[dt < upper]
        else:
            return df
    return apply


def left_open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower not in _invalid_range and upper not in _invalid_range:
            return df[(lower < dt) & (dt <= upper)]
        elif lower not in _invalid_range:
            return df[lower < dt]
        elif upper not in _invalid_range:
            return df[dt <= upper]
        else:
            return df
    return apply


def open_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        print(df)
        print(dt)
        print()
        if lower not in _invalid_range and upper not in _invalid_range:
            return df[(lower < dt) & (dt < upper)]
        elif lower not in _invalid_range:
            return df[(lower < dt)]
        elif upper not in _invalid_range:
            return df[dt < upper]
        else:
            return df
    return apply


def close_interval(lower, upper):
    def apply(df: pd.DataFrame, column=None) -> pd.DataFrame:
        dt = df.index if column is None else df[column]
        if lower not in _invalid_range and upper not in _invalid_range:
            return df[(lower <= dt) & (dt <= upper)]
        elif lower not in _invalid_range:
            return df[lower <= dt]
        elif upper not in _invalid_range:
            return df[dt <= upper]
        else:
            return df
    return apply


def filter_between(*range, open_left=False, open_right=True):

    if len(range) >= 2:
        lower = range[0]
        upper = range[1]
    elif type(range[0]) is list:
        lower = range[0][0] if type(range[0]) is list else range[0]
        upper = range[0][1] if len(range[0]) >= 2 else None
    else:
        raise SystemError("range must be number or list of numbers.")

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
        datetime_str = reduce(
            lambda a, e: a+" "+e,
            [df[c] for c in columns],
            ""
        )

        df["datetime"] = pd.to_datetime(datetime_str)

        df.set_index("datetime", inplace=True)
        return df
    return f


def flatten_table(primary_key: str, concat_keys_func=lambda p, c: f"{p}-{c}"):
    """
    flatten_table(
        'primary',
        lambda p_name,c_name: f'{p_name}-{c_name}'
    )(
        "new_col",
        "value"
    )(df)

    convert df: pandas.DataFrame such that

    primary col1 col2 col3
    p1      d11  d12  d13
    p2      d21  d22  d23

    into

    new_col value
    p1-col1 d11
    p1-col2 d12
    p1-col3 d13
    p2-col1 d21
    p2-col2 d22
    p2-col3 d23


    """
    def apply(df: pd.DataFrame, primary_name, data_name)->pd.DataFrame:
        new_data = [[], []]
        for i, row in df.iterrows():
            for col in df.columns:
                if col is primary_key:
                    pass
                else:
                    new_data[0].append(concat_keys_func(row[primary_key], col))
                    new_data[1].append(row[col])
        new_df = pd.DataFrame(
            {k: v for k, v in zip([primary_name, data_name], new_data)})
        return new_df

    def set_column_names(index_name=primary_key, data_name="data"):
        return lambda df: apply(df, index_name, data_name)
    return set_column_names


def transform(*args, **kwargs):
    """
    Return the new DataFrame transformed from the older one.

    transform(
        {"col1[unit]":lambda df: df["col1"]},
        col1=lambda df: df["col1"]**2
    )(df)
    """
    mutate_dict = args[0] if (len(args) > 0 and type(args[0]) is dict) else {}
    methods = {**mutate_dict, **kwargs}

    def apply(df):
        return reduce(lambda _df, kv: _df.assign(**{kv[0]: kv[1](df)}), methods.items(), df)

    return apply


def subset(*pred, **kwargs):
    """
    sub_df = subset(
        lambda df: df["col1"] > 0,
        col2 = "factor1",
        col3 = lambda df: df["col3"].isin(["A","B"])
    )
    """
    def apply_func(kv):
        key, selector = kv
        if (callable(selector)):
            return lambda df: df[selector(df)]
        else:
            return lambda df: df[df[key] == selector]

    def apply(df):
        d = reduce(lambda df, e: df[e(df)], pred, df)
        return reduce(lambda df, e: apply_func(e)(df), kwargs.items(), d)
    return apply


def create_complete_block_designed_df(mat_selector, group_selector, block_selector):
    """
    pandas.DataFrameから,完全ブロックデザインデータのpandas.DataFrameを作る.

    parameters
    ----------
    df: pandas.DataFrame
    group_selector: str
        ブロックデータの列方向のファクターに用いる列名.
    block_selector: List[str]
        ブロックデータの行方向のファクターに用いる列名のリスト.
    """

    _block = block_selector
    _group = group_selector
    _mat = mat_selector

    def apply(df):
        block_column = df[_block].apply(lambda r: reduce(
            lambda acc, e: acc+"_"+str(e) if len(acc) > 0 else str(e),
            r[_block],
            ""
        ), axis=1)

        temp_df = df.assign(created_block=block_column)

        block_factor = temp_df["created_block"].astype(
            "category").cat.categories
        group_factor = temp_df[_group].astype("category").cat.categories

        matrix = []
        for b in block_factor:
            entry = []
            for g in group_factor:
                item = temp_df[
                    (temp_df["created_block"] == b) & (temp_df[_group] == g)
                ][_mat].values
                entry.append(item[0] if len(item) > 0 else None)
            matrix.append(entry)

        return pd.DataFrame(
            matrix,
            index=block_factor,
            columns=group_factor
        ).dropna()

    return apply


def create_bands(ini: float, fin: float, step: float, with_outer: bool=True)\
        -> List[Tuple[Number, Number]]:
    """
    Create list of tuples defining lower and upper limits.

    Parameters
    ----------
    ini: float
        The first boundary of bands.
    fin: float
        The second boundary of bands.
    step: float
        Width of each band.
    with_outer: bool
        If true, bands of outside of [ini, fin] (or [fin, ini])
            are added.
        Therefore, returns [(-math.inf, ini), (ini, ini+step),...
            (,ini+n*step), (ini+n*step,math.inf)]
        Default is True.

    Return
    ------
    level: list[tuple[float]]
    """
    n = int((fin-ini)/step)
    outer = [(-math.inf, ini), (ini+n*step, math.inf)]\
        if ini <= fin else\
        [(math.inf, ini), (ini+n*step, -math.inf)]
    if n < 0:
        raise SystemError("Invalid sign of step.")

    level = []
    if with_outer:
        level.append(outer[0])
    for i in range(n):
        level.append((ini + i*step, ini+(i+1)*step))
    if with_outer:
        level.append(outer[1])
    return level


def _get_factor_of_bands(bands: List[Tuple[Number, Number]])->List[str]:
    """
    """
    def stringifier(band: Tuple[Number, Number]):
        fst, snd = band
        _fst = "" if fst == -math.inf else\
            "" if fst == math.inf else\
            str(fst)
        _snd = "" if snd == -math.inf else\
            ""if snd == math.inf else\
            str(snd)
        return _fst+"~"+_snd if fst <= snd else _snd+"~"+_fst
    return list(map(stringifier, bands))

def _get_mean_of_bands(bands: List[Tuple[Number,Number]])->List[Number]:
    def apply(band: Tuple[Number,Number])->Number:
        fst,snd = band
        return 0.5*(fst+snd)
    return list(map(apply,bands))

def _get_lower_of_bands(bands):
    return list(map(lambda band: band[0],bands))

def _get_upper_of_bands(bands):
    return list(map(lambda band: band[1],bands))

def _get_index_of_bands(bands):
    return list(map(lambda item: item[0],enumerate(bands)))

def _i_position_selector(position_type):
    if position_type is "index":
        return _get_index_of_bands
    if position_type is "center":
        return _get_mean_of_bands
    if position_type is "lower":
        return _get_lower_of_bands
    if position_type is "upper":
        return _get_upper_of_bands
    raise Exception(f"position_type{position_type} can not be recognized ! Choose 'index|lower|center|upper'.")

def band_factor(
    bands: List[Tuple[Number, Number]],
    position: str="index"
    )->Callable[[pd.DataFrame, Optional[str]], Tuple[pd.Series, List[str]]]:
    """
    Limit of bands must increase monotonic.

    Domains are (lower, upper], or lower < x <= upper.

    Usage
    -----
    cat_depth = categorize_by_bands(depth_bands)(df, "depth")
    cat_depth = categorize_by_bands(depth_bands)(df, lambda df: df["depth"])
    cat_depth = categorize_by_bands(depth_bands)(df, lambda df: df["depth"] * df["depth_factor"])

    """
    factor = _get_factor_of_bands(bands)
    position = _i_position_selector(position)(bands)
    cut_array = list(map(lambda band: band[0], bands))
    cut_array.append(bands[-1][1])

    def apply_to_df(df: pd.DataFrame, selector: Optional[str]=None)\
            ->Tuple[pd.Series, List[str]]:

        target = df if selector is None else df[selector]

        return (pd.cut(target, cut_array, labels=factor), factor, position)
    return apply_to_df




assert(create_bands(-1, 10, 2, False) ==
       [(-1, 1), (1, 3), (3, 5), (5, 7), (7, 9)])
assert(create_bands(0, 10, 2, False) == [
       (0, 2), (2, 4), (4, 6), (6, 8), (8, 10)])
assert(create_bands(10, 0, -2, False) ==
       [(10, 8), (8, 6), (6, 4), (4, 2), (2, 0)])
assert(create_bands(10, 0, -2, True) ==
       [(math.inf, 10), (10, 8), (8, 6), (6, 4), (4, 2), (2, 0), (0, -math.inf)])

assert(list(_get_factor_of_bands(create_bands(-1, 10, 2, False)))
       == ['-1~1', '1~3', '3~5', '5~7', '7~9'])
assert(list(_get_factor_of_bands(create_bands(-1, 10, 2, True)))
       == ['~-1', '-1~1', '1~3', '3~5', '5~7', '7~9', '9~'])
assert(list(_get_factor_of_bands(create_bands(10, 0, -2, True)))
       == ['10~', '8~10', '6~8', '4~6', '2~4', '0~2', '~0'])


