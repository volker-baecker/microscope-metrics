PLUGINS = {}

class Registry:
    def __init__(self):
        self.r = {}


def register(fn):
    PLUGINS[fn.__name__] = fn
    return fn


class MyClass(object):
    reg = Registry()

    def __init__(self):
        pass

    @register
    def my_method(self, nr1):
        print('my_method')
        print(nr1)

    @register
    def my_other_method(self, nr2):
        print('my other method')
        print(nr2)


c = MyClass()
for k in PLUGINS:
    PLUGINS[k](c, 'Hi')


# # from abc import ABC
# # from decorator import print_function
# #
# # class Base():
# #     # class Register:
# #     #
# #     #     @classmethod
# #     #     def register(cls, registered):
# #     #         pass
# #     #
# #     r = {}
# #
# #     def __init__(self):
# #         pass
# #         # self.r = {}
# #
# #     @classmethod
# #     def register(cls, f):
# #         cls.r[f.__name__] = f
# #         return f
# #
# #
# # class Derived(Base):
# #     def __init__(self):
# #         super().__init__()
# #
# #     # def register(self, f):
# #     #     self.r[f.__name__] = f
# #     #     return f
# #
# #     @Derived.register
# #     def to_reg(self):
# #         print("I'm there")
# #
# #
# # d = Derived()
# # d.r['to_reg']()
