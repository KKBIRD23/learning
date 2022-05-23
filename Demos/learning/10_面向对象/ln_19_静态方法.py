class Dog(object):

    @staticmethod
    def run():
        print("小狗要跑...")


# 通过 类名. 调用静态方法
print(Dog.mro())
Dog.run()
