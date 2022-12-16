def f1(arg=2.66):
    return arg


def f2(arg=5.0):
    r1 = f1()
    print('r1', arg, r1)
    r2 = f1(arg)
    print('r2', arg, r2)
    arg=3
    r3 = f1(arg)
    print('r3', arg, r3)
    r4 = f1(arg)
    print('r4', arg, r4)
    r5 = f1()
    print('r5', arg, r5)
