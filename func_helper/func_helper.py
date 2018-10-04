# -*- coding: utf-8 -*-
from functools import reduce


def identity(df):
    return df


def over_args(fn, transforms):
    """
    double = lambda x: x*2
    square = lambda x: x**2
    over_args(lambda *arg: [*arg], [square, double])()
    """
    def f(*args):
        return fn(
            list(
                map(lambda arg, transform: transform(arg),
                    args, transforms
                    )
            )
        )
    return f


def compose(*funcs):
    """
    [f,g,h] -> x -> f(g(h(x)))
    """
    def f(arg):
        return reduce(
            lambda acc, f: f(acc),
            funcs[::-1],
            arg
        )
    return f


def pip(*funcs):
    """
    [f,g,h] -> x -> h(g(f(x)))
    """
    def g(a):
        return reduce(
            lambda acc, f: f(acc),
            funcs,
            a
        )
    return g


def trace(f):
    def traced(*arg):
        print("called with argument " + str(arg))
        return f(*arg)
    return traced


def memoize(cache={}):

    def memoized(f):
        def with_cache(*arg):
            if arg not in cache:
                cache[arg] = f(*arg)
            return cache.get(arg)
        return with_cache
    return memoized


def Y(f):
    """
    F -> (a -> F.a) ->
    (x -> F.(a -> x.x.y)).(x -> F.(a -> x.x.y))
    """
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))


def recursiveExtender(decorator):
    """
    decorator -> templateFunc ->

    f: a -> b
    decorator: f -> (a -> f.a)
    templateFunc: f -> (a -> f.a)

    Usage
    -----

    def fib_template(f):

        # F -> (a -> F.a)

        def func(arg):
            n = int(arg)
            if n < 2:
                return 1
            else:
                return f(n-1) + f(n-2)
        return func

    traced_fib = Y(recursiveExtender(trace)(fib_template))
    traced_fib(5)
    """
    return lambda templateFunc: lambda f: templateFunc(decorator(f))


def tee(f):
    def g(a):
        f(a)
        return a
    return g


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


class dotdict(object):
    """オブジェクトグラフ内の辞書要素をプロパティ風にアクセスすることを可能にするラッパー。
        DotAccessible( { 'foo' : 42 } ).foo==42

    メンバーを帰納的にワップすることによりこの挙動を下層オブジェクトにも与える。
        DotAccessible( { 'lst' : [ { 'foo' : 42 } ] } ).lst[0].foo==42
    """

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return "DotAccessible(%s)" % repr(self.obj)

    def __getitem__(self, i):
        """リストメンバーをラップ"""
        return self.wrap(self.obj[i])

    def __getslice__(self, i, j):
        """リストメンバーをラップ"""

        return map(self.wrap, self.obj.__getslice__(i, j))

    def __getattr__(self, key):
        """辞書メンバーをプロパティとしてアクセス可能にする。
        辞書キーと同じ名のプロパティはアクセス不可になる。
        """

        if isinstance(self.obj, dict):
            try:
                v = self.obj[key]
            except KeyError:
                v = self.obj.__getattribute__(key)
        else:
            v = self.obj.__getattribute__(key)

        return self.wrap(v)

    def wrap(self, v):
        """要素をラップするためのヘルパー"""

        if isinstance(v, (dict, list, tuple)):  # xx add set
            return self.__class__(v)
        return v
