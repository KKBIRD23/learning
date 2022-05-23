"""
多继承
语法： class 子类名(父类名1, 父类名2)
"""


class A:
    def test_a(self):
        print("class A方法")


class B:
    def test_b(self):
        print("class B方法")


class C(A, B):
    pass


c = C()

c.test_a()
c.test_b()
