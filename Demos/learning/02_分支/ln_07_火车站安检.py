"""
定义布尔变量has_ticket表示是否有票
定义整型变量knife_length表示刀的长度，单位：厘米
有票才允许案件
刀的长度不能超过20厘米
"""

has_ticket = True
knife_length = 21

if has_ticket:
    print("车票检查通过，准备开始安检")
    if knife_length > 20:
        print(f"您携带的刀太长了，有 {knife_length:.2f} 公分长！")
#       print("您携带的刀太长了，有 %05d 公分长！" % knife_length)
        print("不允许上车！")
    else:
        print("安检通过，旅途愉快")
else:
    print("大哥，请先买票")