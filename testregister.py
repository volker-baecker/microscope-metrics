class Registry:
    def __init__(self):
        self.r = {}

    def register(self):
        def wrapper(fn, *args, **kwargs):
            self.r[fn.__name__] = fn
            return fn
        return wrapper


class MyClass(object):
    reg = Registry()

    def __init__(self):
        pass

    @reg.register()
    def my_method(self, nr1):
        print('my_method')
        print(nr1)

    @reg.register()
    def my_other_method(self, nr2):
        print('my other method')
        print(nr2)


c = MyClass()
print(c.reg.r)
for k in c.reg.r:
    c.reg.r[k](c, 'Hi')


# from abc import ABC
# from decorator import print_function
#
# class Base():
#     # class Register:
#     #
#     #     @classmethod
#     #     def register(cls, registered):
#     #         pass
#     #
#     r = {}
#
#     def __init__(self):
#         pass
#         # self.r = {}
#
#     @classmethod
#     def register(cls, f):
#         cls.r[f.__name__] = f
#         return f
#
#
# class Derived(Base):
#     def __init__(self):
#         super().__init__()
#
#     # def register(self, f):
#     #     self.r[f.__name__] = f
#     #     return f
#
#     @Derived.register
#     def to_reg(self):
#         print("I'm there")
#
#
# d = Derived()
# d.r['to_reg']()
