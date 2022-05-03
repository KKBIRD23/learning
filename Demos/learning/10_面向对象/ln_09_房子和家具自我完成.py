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


# 房屋家具类
class HouseItem:
    def __init__(self, item_name, item_area):
        # 定义变量接受传参——家具名称
        self.item_name = item_name
        # 定义变量接受传参——家具面积
        self.item_area = item_area

    def __str__(self):
        return f'{self.item_name} 的面积是 {self.item_area:.2f}'


# 房屋类
class House:
    def __init__(self, house_type, house_area):
        # 定义变量接受传参——房屋类型
        self.house_type = house_type
        # 定义变量接受传参——房屋面积
        self.house_area = house_area
        # 定义变量房屋剩余面积，初始值为房屋面积
        self.free_area = house_area
        # 定义 列表 变量存放家具列表
        self.item_list = []

    def __str__(self):
        return f'房屋户型为：{self.house_type}\n' \
               f'房屋总面积：{self.house_area:.2f} 剩余面积为：{self.free_area:.2f}\n' \
               f'房屋中的家具有：{self.item_list}'

    def add_house_item(self, house_item):
        print(f'要添加[{house_item}]这件家具')
    # - 需要判断 家具的面积 是否 超过剩余面积，如果超过，提示不能添加这件家具
        if house_item.item_area > self.house_area:
            print(f'{house_item}的面积太大了！不能摆进来哟！')
            return
    # - 将家具的名称 追加到 家具名称列表 中
        self.item_list.append(house_item.item_name)
    # - 用 房子的剩余面积 - 家具的面积
        self.free_area -= house_item.item_area


# 1. 创建家具
bed = HouseItem("席梦思", 4)
chest = HouseItem("衣柜", 2)
tabel = HouseItem("餐桌", 1.5)
print(bed)
print(chest)
print(tabel)

# 2. 创建房屋
my_house = House("三室两厅", 120)

# 3. 添加家具
my_house.add_house_item(bed)
my_house.add_house_item(chest)
my_house.add_house_item(tabel)

print(my_house)
