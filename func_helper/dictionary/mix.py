from functools import reduce


def _tee(f):
    def apply(a):
        f(a)
        return a
    return apply


def mix(*dicts):
    return reduce(
        lambda acc, e: _tee(lambda d: d.update(e))(acc),
        dicts,
        {}
    )
