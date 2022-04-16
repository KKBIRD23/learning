# _*_ coding:utf-8 _*_
__author__ = 'Hui'

import codecs
import configparser
import os,sys
import binascii

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(BASE_DIR)

# 配置文件
# TEST_CONFIG = os.path.join(BASE_DIR,"OBUFormatJudgment","OBUDate.txt")
TEST_CONFIG = os.path.join(BASE_DIR,"OBUDate.txt")
TEST_CONFIG = TEST_CONFIG.replace('\\', '/')
CARDMODEL = {"01":"一类车","02":"二类车","03":"三类车","04":"四类车","05":"五类车","06":"六类车","07":"自定义1","08":"自定义2",\
             "09":"自定义3","0A":"自定义4","0B":"一类货车","0C":"二类货车","0D":"三类货车","0E":"四类货车","0F":"五类货车","10":"六类货车",\
             "15":"一类专项作业车","16":"二类专项作业车","17":"三类专项作业车","18":"四类专项作业车","19":"五类专项作业车","1A":"六类专项作业车"}
CARNOCOLOCK = {"0000":"蓝色","0001":"黄色","0002":"黑色","0003":"白色","0004":"渐变绿色","0005":"黄绿双拼色","0006":"蓝白渐变"}

fd = open(TEST_CONFIG)
data = fd.read()

        #  remove BOM
if data[:3] == codecs.BOM_UTF8:
    data = data[3:]
    file = codecs.open(TEST_CONFIG, "w")
    file.write(data)
    file.close()
fd.close()

con = configparser.ConfigParser()
con.read(TEST_CONFIG, encoding='utf-8')

CARFILE = con.get("OBU", "carfile")
SYSTEMFILE = con.get("OBU", "systemfile")
# VERSION = con.get("VS", "version")
MODEL = CARFILE[28:30]

# print(CARDFILE)
# print(SYSTEMFILE)
# print(VERSION)
print("==" * 64)

def OBU2015(CARFILE,SYSTEMFILE):
    print("==" * 64)
    print("进行2015OBU格式检查")
    print("系统文件信息拆解如下：")
    Issuer = SYSTEMFILE[:16]
    print("发行方标识：" + Issuer)
    Treaty = SYSTEMFILE[16:18]
    print("协约类型：" + Treaty)
    edition = SYSTEMFILE[18:20]
    print("版本：" + edition)
    id = SYSTEMFILE[20:36]
    print("合同序列号：" + id)
    dates = SYSTEMFILE[36:44]
    print("签署日期：" + dates)
    daten = SYSTEMFILE[44:52]
    print("过期日期：" + daten)
    stat = SYSTEMFILE[52:54]
    print("拆卸状态：" + stat)
    FF = SYSTEMFILE[54:198]
    print("预留字段：" + FF)
    print("==" * 32)
    print("车辆信息拆解如下：")
    carNo = CARFILE[:24]
    province = carNo[:4]
    verStr = binascii.unhexlify(province)
    verArea = "%s" % (verStr[0:2].decode('gbk'))
    LetterAndNum = carNo[4:24]
    LA =binascii.a2b_hex(LetterAndNum).decode()
    print("车牌：" + carNo + "("+ verArea + LA +")")
    colock = CARFILE[24:28]
    print("车牌颜色：" + colock + "("+CARNOCOLOCK[colock]+")")
    Model = CARFILE[28:30]
    print("车辆类型：" + Model + "("+CARDMODEL[Model]+")")
    user = CARFILE[30:32]
    print("车辆用户类型：" + user)
    ch = str(int(CARFILE[32:36],16))
    k = str(int(CARFILE[36:38],16))
    g = str(int(CARFILE[38:40],16))
    print("车辆长：" +ch + "dm")
    print("宽：" + k + "dm")
    print("高：" + g + "dm")
    wheel = CARFILE[40:42]
    print("车轮数：" + wheel)
    axle = CARFILE[42:44]
    print("车轴数：" + axle)
    between = CARFILE[44:48]
    print("轴距：" + between)
    vehicle = CARFILE[48:54]
    print("车辆载重/座位数：" + vehicle)
    features = CARFILE[54:86]
    print("车辆特征描述：" + features)
    engineNo = CARFILE[86:118]
    print("车辆发动机号：" + binascii.a2b_hex(engineNo).decode())
    FE = CARFILE[118:158]
    print("预留字段：" + FE)


def OBU2019(CARFILE,SYSTEMFILE):
    print("==" * 64)
    print("进行2019OBU格式检查")
    print("系统文件信息拆解如下：")
    Issuer = SYSTEMFILE[:16]
    print("发行方标识：" + Issuer)
    Treaty = SYSTEMFILE[16:18]
    print("协约类型：" + Treaty)
    edition = SYSTEMFILE[18:20]
    print("版本：" + edition)
    id = SYSTEMFILE[20:36]
    print("合同序列号：" + id)
    dates = SYSTEMFILE[36:44]
    print("签署日期：" + dates)
    daten = SYSTEMFILE[44:52]
    print("过期日期：" + daten)
    stat = SYSTEMFILE[52:54]
    print("拆卸状态：" + stat)
    FF = SYSTEMFILE[54:198]
    print("预留字段：" + FF)
    print("==" * 32)
    print("车辆信息拆解如下：")
    carNo = CARFILE[:24]
    province = carNo[:4]
    verStr = binascii.unhexlify(province)
    verArea = "%s" % (verStr[0:2].decode('gbk'))
    LetterAndNum = carNo[4:24]
    LA =binascii.a2b_hex(LetterAndNum).decode()
    print("车牌：" + carNo + "("+ verArea + LA +")")
    colock = CARFILE[24:28]
    print("车牌颜色：" + colock + "("+CARNOCOLOCK[colock]+")")
    Model = CARFILE[28:30]
    print("车辆类型：" + Model + "("+CARDMODEL[Model]+")")
    user = CARFILE[30:32]
    print("车辆用户类型：" + user)
    ch = str(int(CARFILE[32:36],16))
    k = str(int(CARFILE[36:40],16))
    g = str(int(CARFILE[40:44],16))
    print(ch)
    print("车辆长(转换十进制)：" + ch +"mm")
    print("宽(转换十进制)：" + k +"mm")
    print("高(转换十进制)：" + g +"mm")
    wheel = str(int(CARFILE[44:50],16))
    print("车辆核定载重(转换十进制)：" + wheel+"kg")
    axle = str(int(CARFILE[50:56],16))
    print("整备质量(转换十进制)：" + axle +"kg" )
    between = CARFILE[56:62]
    if between == "FFFFFF":
        print("车辆总质量为空：" + between)
    else:
        between2 =str(int(between,16))
        print("车辆总质量(转换十进制)：" + between2 + "kg")
    # print("车辆总质量：" + between)
    vehicle = CARFILE[62:64]
    print("核定载客人数：" + vehicle)
    features = CARFILE[64:98]
    print("车辆识别代码：" + binascii.a2b_hex(features).decode())
    engineNo = CARFILE[98:130]
    print("车辆特征描述（原数据）：" + engineNo)
    FE = CARFILE[130:158]
    print("预留字段：" + FE)

if MODEL < "0A":
    print("车辆为客车，按2014规范检查")
    OBU2015(CARFILE, SYSTEMFILE)
else:
    print("车辆为非客车，按2019规范检查")
    OBU2019(CARFILE, SYSTEMFILE)

sec = input("记得拷贝数据，回车后将关闭")
print("检查完毕，结束了")
