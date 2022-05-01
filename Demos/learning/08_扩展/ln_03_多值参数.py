def demo(num, *nums, **person):
    print(num)
    print(nums)
    print(person)


demo(1)
print("-" * 50)

demo(1, 2, 3, 4, 5)
print("-" * 50)

demo(1, 2, 3, 4, 5, name="小米", age="18")
