"""
需求：
1. 房子(House)有 户型 总面积 和 家具名称列表
   - 新房子没有任何家具
2. 家具(HouseItem)有 名称 和 占地面积，其中
   - 席梦思(bed)占地 4 平米
   - 衣柜(chest)占地 2 平米
   - 餐桌(table)占地 1.5 平米
3. 将以上三件 家具 添加到 房子中
   - 需要判断 家具的面积 是否 超过剩余面积，如果超过，提示不能添加这件家具
   - 将家具的名称 追加到 家具名称列表 中
   - 用 房子的剩余面积 - 家具的面积
4. 打印房子时，要求输出： 户型、 总面积、 剩余面积、 家具名称列表
"""


class HouseItem:
    def __init__(self, name, area):
        self.name = name
        self.area = area

    def __str__(self):
        return f'{self.name} 占地 {self.area:.2f}'


class House:
    def __init__(self, house_type, area):
        self.house_type = house_type
        self.area = area
        # 剩余面积
        self.free_area = area
        # 家具名称列表
        self.item_list = []

    def __str__(self):
        return f'户型：{self.house_type}\n' \
               f'总面积:{self.area:.2f}[剩余面积：{self.free_area:.2f}]\n' \
               f'家具列表：{self.item_list}'

    def add_item(self, item):
        print(f'要添加 {item}')
        # - 需要判断 家具的面积 是否 超过剩余面积，如果超过，提示不能添加这件家具
        if item.area > self.free_area:
            print(f"{item.name}家具太大！放不下了！！！")
            return
        # - 将家具的名称 追加到 家具名称列表 中
        self.item_list.append(item.name)
        # - 用 房子的剩余面积 - 家具的面积
        self.free_area -= item.area


# 1. 创建家具
bed = HouseItem("席梦思", 4)
chest = HouseItem("衣柜", 2)
tabel = HouseItem("餐桌", 1.5)
print(bed)
print(chest)
print(tabel)

# 2. 创建房子
my_house = House("三室两厅", 120)

my_house.add_item(bed)
my_house.add_item(chest)
my_house.add_item(tabel)

print(my_house)
