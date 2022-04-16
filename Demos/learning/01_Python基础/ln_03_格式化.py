name = '小明'
print('我的名字叫 %s ,请多多关照' % name)

student_no = 1
print('我的学号是 %06d' % student_no)

price = 8.5
weight = 7.5
money = price * weight
print('苹果单价 %.2f 元/斤，购买了 %.3f 斤，需要支付 %.4f 元' % (price,weight,money))

scale = 0.25
print('数据比例是 %.2f%%,我的名字是%s' % (scale * 100,name))
print('数据比例是 %.2f' % scale ,'%')
print('123''321')

# f-string的用法
r = 2.5
s = 3.1415926 * r **2
d = 5
print(f'the area of acircle with radius {r} is {s:.8f}')
print(f'{d:06d}')