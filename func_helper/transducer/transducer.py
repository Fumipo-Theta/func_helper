from ..func_helper import compose
from functools import reduce
"""
const transduce = (compose => {
    const mapping = f => reducer => (acc, e) => reducer(acc, f(e));
    const filtering = f => reducer => (acc, e) =>
      !f(e)
        ? acc
        : reducer(acc, e);
    const folding = f => x => reducer => (acc, e) =>
      acc.length === 0
        ? reducer(acc, f(x, e))
        : reducer(acc, f(acc[acc.length - 1], e));
    const taking = n => reducer => (acc, e) =>
      acc.length < n
        ? reducer(acc, e)
        : reducer(acc, undefined);
    const concatReducer = (acc, e) =>
      e || e === 0
        ? [...acc, e]
        : [...acc];

    const _intoArray = ts => xs => xs.reduce(ts(concatReducer), []);
    const intoArray = (...fs) => xs => xs.reduce(compose(...fs)(concatReducer), []);
"""


def concatReducer(acc, e):
    return [*acc, e] if (e is not None or e is 0) else [*acc]


def mapT(func):
    return lambda reducer: lambda acc, e: reducer(acc, func(e))


def filterT(pred):
    return lambda reducer: lambda acc, e: reducer(acc, e) if pred(e) else acc


def foldT(f):
    return lambda x: lambda reducer: lambda acc, e: reducer(acc, f(x, e)) if len(acc) is 0 else reducer(acc, f(acc[-1], e))


def takeT(n):
    return lambda reducer: lambda acc, e: reducer(acc, e) if len(acc) < n else reducer(acc, None)


def intoArray(*funcs):
    return lambda xs: reduce(compose(*funcs)(concatReducer), xs, [])
