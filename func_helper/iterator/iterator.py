from functools import reduce


def mapping(mapFunc):
    return lambda arr: map(
        mapFunc,
        arr
    )


def filtering(pred):
    return lambda arr: filter(
        pred,
        arr
    )


def reducing(reduceFunc):
    return lambda initial: lambda arr: reduce(
        reduceFunc,
        arr,
        initial
    )


def is_all(pred):
    """
    pred: a -> bool
    arr: list, tuple

    assert(is_all(lambda x: x > 0)([1,2,3,4, 5]) is True)
    assert(is_all(lambda x: x > 0)((1,2,3,4, 0)) is not True)
    """
    return lambda arr: all(map(pred, arr))


assert(is_all(lambda x: x > 0)([1, 2, 3, 4, 5]) is True)
assert(is_all(lambda x: x > 0)((1, 2, 3, 4, 0)) is not True)


def is_any(pred):
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
