def over_iterator(d):
    return lambda it: dict(zip(d.keys(), map(lambda f: [f(i) for i in it], d.values())))
