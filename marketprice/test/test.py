# test dynamic function name in class

class A():
    def fcn1(x):
        print(f'x = {x}')

    def fcn2(x):
        print(f'x square = {x**2}')

def make_method(name):
    def _method(self):
        print("method {0} in {1}".format(name, self))
    return _method

for name in ('fcn1', 'fcn2'):
    _method = make_method(name)
    setattr(A, name, _method)	