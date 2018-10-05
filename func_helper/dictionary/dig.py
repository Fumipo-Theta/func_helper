import re
from functools import reduce


def dig(obj, key, look_array=False):
    """
    Return value of a given key in dictionary.
    The first found is returned if there are multiple matched keys.

    Parameters:
        obj: dict
        key: any
        look_array: bool, optional
            If true, this function searchs into list or tuple.
            Default is False.

    Return:
        subobject: dict
    """
    #print(obj, key)
    if type(obj) is dict:
        if key in obj:
            return obj.get(key)

        else:
            return reduce(
                lambda acc, subobj: dig(
                    subobj, key, look_array) if acc is None else acc,
                obj.values(),
                None
            )

    elif type(obj) in [list, tuple]:

        return reduce(
            lambda acc, hit: hit if acc is None else acc,
            map(
                lambda subobj: dig(subobj, key, look_array),
                obj
            ),
            None
        ) if look_array else None

    else:

        return None


def dig_all(obj, key, look_array=False, store=[], path=""):
    """
    Return list of values having a given key name with path to the value.
    If there are lists and tuples, this method checks all elements in them.

    Parameters
    ----------
    obj: dict
    key: str
    look_array: bool, optional
        If true, this function searchs into list or tuple.
        Default is False


    Return
    ------
    objects_and_paths: list[tuple[dict, str]]
        List of tuple, which has 2 elements.
        The first is value in the object matching the given key.
        The second is path to the value in the object
            expressed as period separated keys.

    Usage
    -----
    obj = {
        "a": {
            "level1" : {
                "level2" : {         # target
                    "level3" : [0],
                    "additional" : 0,
                    0 : 0
                },
            }
        },
        "b" : {
            "level1" : {
                "level2" : {         # target
                    "level3" : [1]
                }
            }
        }
    }

    dig_all(obj, "level2") == [
        ({"level3" : [0], "additional" : 0}, ".a.level1.level2"),
        ({"level3" : [1]}, ".b.level1.level2")
    ]
    """

    if type(obj) is dict:
        if key in obj:
            #print(obj, key, store, path)
            return reduce(
                lambda acc, kv: dig_all(
                    kv[1],
                    key,
                    look_array,
                    [*acc, (obj.get(key), path+"."+str(kv[0]))
                     ] if kv[0] is key else acc,
                    path+"."+str(kv[0])
                ),
                zip(obj.keys(), obj.values()),
                store
            )
        else:
            return reduce(
                lambda acc, kv: dig_all(
                    kv[1], key, look_array, acc, path+"."+str(kv[0])),
                zip(obj.keys(), obj.values()),
                store
            )
    elif type(obj) in [list, tuple]:
        return reduce(
            lambda acc, iv: dig_all(
                iv[1], key, look_array, acc, path+"["+str(iv[0])+"]"),
            enumerate(obj),
            store
        ) if look_array else store

    else:
        return store


def get(obj, *expression, separator="."):
    """
    access attribute of dictionary by expression of string
        separated by separator such as ".".

    Parameters
    ----------
    obj: dict
    *expression: any
        Period separated string or symbol for key of dictionary.
        Index of list or tuple are also aveilable.
        If key of dictionary is not string, it must be separated parameter.
    separator: str, optional
        Default value is "."

    Usage
    -----
    obj = {
        "a" : {
            "level1" : {
                "level2" : {
                    "level3" : [0]
                },
                0 : 0,
                "array" : [{"x" : 1}]
            }
        }
    }

    # Accessor separated by "."
    dot_access(obj, ".a.level1.level2") == {'level3' : [0]}
    dot_access(obj, "a.level1.level2") == {'level3' : [0]}
    dot_access(obj, "a..level1.level2") == {'level3' : [0]}

    # skip some levels.
    dot_access(obj, "a.level2") == {'level3' : [0]}
    dot_access(obj, "level2") == {'level3' : [0]}

    # multiple parameters
    dot_access(obj, "a", "level1", "level2") == {'level3' : [0]}
    dot_access(obj, "a.level1", 0) == 0
    dot_access(obj, "a.level1.0") == {}

    # Access in array
    dot_access(obj, "a.level1.array", 0) == {"x" : 1}
    dot_access(obj, "a.level1.array.x") == 1
    dot_access(obj, "a.level1.array", 0, "x") == 1
    dot_access(obj, "a.level1.array[0].x") == 1
    """
    def convert_array_element(s):
        """
        s = "a1b[1][21][3]"
        re.findall(r"\[(\d+)\]",s) == ['1','21','3']
        re.findall(r"^([\w\d]+)",s) == ['a1b']

        s = "a1b"
        re.findall(r"\[(\d+)\]",s) == []
        re.findall(r"^([\w\d]+)",s) == ['a1b']

        s = "a1b[][]"
        re.findall(r"\[(\d+)\]",s) == []
        re.findall(r"^([\w\d]+)",s) == ['a1b']

        s = ""
        re.findall(r"\[(\d+)\]",s) == []
        re.findall(r"^([\w\d]+)",s) == []
        """
        invarid_syntax = re.findall(r"(\][\w\d])", s)
        if len(invarid_syntax) > 0:
            raise SyntaxError(*invarid_syntax)

        index = re.findall(r"\[(\d+)\]", s)
        attr = re.findall(r"^([\w\d]+)", s)
        return [attr[0] if len(attr) > 0 else "", *[int(i) for i in index]]

    def spliter(e, separator):
        return reduce(lambda acc, e: [*acc, *e],
                      map(
            convert_array_element,
            e.split(separator)
        ),
            []
        )

    accessor = reduce(
        lambda acc, e: [*acc, *e],
        map(
            lambda e: spliter(e, separator) if type(e) is str else [e],
            expression
        ),
        []
    )

    def dig_func(acc, key):
        """
        if type(acc) is dict:
            return dig(acc,key) if key is not "" else acc
        """
        if type(acc) is dict:
            return dig(acc, key, True) if key is not "" else acc

        elif type(acc) in [list, tuple]:
            objectized = {}
            for i, e in enumerate(acc):
                objectized[i] = e

            return dig(objectized, key, True) if key is not "" else acc

    return reduce(
        dig_func,
        accessor,
        obj
    )
