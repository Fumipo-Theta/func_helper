from typing import List, Tuple, Iterable, Iterator, TypeVar, Callable, Union, MappingView
from functools import reduce

T = TypeVar('T')
S = TypeVar('S')


def mapping(mapFunc: Callable[[T], S])->Callable[[Iterable[T]], Iterable[S]]:
    return lambda arr: map(
        mapFunc,
        arr
    )


def filtering(pred: Callable[[T], bool]) -> Callable[[Iterable[T]], Iterable[T]]:
    return lambda arr: filter(
        pred,
        arr
    )


def reducing(reduceFunc: Callable[[T, S], T])->Callable[[T], Callable[[Iterable[S]], T]]:
    """
    reducing: (T, S -> T) -> T -> ([S] -> T)
    reducing: (acc, e -> acc) -> a -> ([a] -> acc)

    Usage
    -----
    reducing(reduce_func)(initial)(arr)

    """
    return lambda initial: lambda arr: reduce(
        reduceFunc,
        arr,
        initial
    )


def is_all(pred: Callable[[T], bool])->Callable[[Iterable[T]], bool]:
    """
    pred: a -> bool
    arr: list, tuple

    assert(is_all(lambda x: x > 0)([1,2,3,4, 5]) is True)
    assert(is_all(lambda x: x > 0)((1,2,3,4, 0)) is not True)
    """
    return lambda arr: all(map(pred, arr))


assert(is_all(lambda x: x > 0)([1, 2, 3, 4, 5]) is True)
assert(is_all(lambda x: x > 0)((1, 2, 3, 4, 0)) is not True)


def is_any(pred: Callable[[T], bool]):
    """
    pred:a -> bool
    arr: list, tuple

    assert(is_any(lambda x: x%2 is 0)([1,3,5,7,8]) is True)
    assert(is_any(lambda x: x%2 is 0)((1,3,5,7,9)) is not True)
    """
    return lambda arr: any(map(pred, arr))


assert(is_any(lambda x: x % 2 is 0)([1, 3, 5, 7, 8]) is True)
assert(is_any(lambda x: x % 2 is 0)((1, 3, 5, 7, 9)) is not True)


def all_equal(arr):
    """
    assert(all_equal([1,1,1]) is True)
    assert(all_equal((1,1,2)) is not True)
    """

    first, *rest = arr
    return is_all(lambda e: e == first)(rest)


assert(all_equal([1, 1, 1]) is True)
assert(all_equal((1, 1, 2)) is not True)


def interval_generator(array: List[T]) -> Iterator[Tuple[T, T]]:
    """
    Iterate tuple of pair of values from start of the list.

    Usage
    -----
    interval_generator([0,1,2,3]) -> [(0,1),(1,2),(2,3)]
    """
    lower = array[0:-1]
    upper = array[1:]
    return zip(lower, upper)


assert(list(interval_generator([0, 1, 2, 3, 4]))
       == [(0, 1), (1, 2), (2, 3), (3, 4)])


def rich_mapping(map_func):
    """
    (T, int, Array[T] -> S) -> Array[T] -> Array[S]
    (e,i,arr -> v) -> arr -> arr
    """
    def apply(arr):
        ret = []
        for i, v in enumerate(arr):
            ret.append(map_func(v, i, arr))
        return ret
    return apply


def tuple_mapping(map_func):
    """
    (*T -> S) -> Tuple[T] -> List[S]

    Usage
    -----
    tuple_mapping(
        lambda x,y : x+y
    )([(0,1),(1,1),(2,1)])

    # -> [1,2,3]
    """
    def apply(tpls):
        ret = []
        for t in tpls:
            ret.append(map_func(*t))
        return ret
    return apply


def over_iterator(*fns):
    return lambda it: map(lambda f: [f(i) for i in it], fns)
