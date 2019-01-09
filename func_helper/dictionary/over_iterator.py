def over_iterator(d: dict={}, **kwargs):
    _d = {**d, **kwargs}
    return lambda it: dict(zip(_d.keys(), map(lambda f: [f(i) for i in it], _d.values())))
