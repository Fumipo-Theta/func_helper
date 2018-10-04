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
