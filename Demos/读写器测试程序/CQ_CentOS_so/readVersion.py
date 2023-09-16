# -*- coding: utf-8 -*-
import ctypes
import sys
import time
from ctypes import *

# 引入动态库libDemo.so
library = cdll.LoadLibrary("./libICC_VFJCQ.so")
JT_OpenReader = library.JT_OpenReader
re = JT_OpenReader(0, "COM0")
if re < 0:
    print("连接读写器失败")
    print(re)
    sys.exit()
Handle = re
LEDDisplay = library.JT_LEDDisplay
re = LEDDisplay(Handle, 1, 1, 1)
# print(re)
time.sleep(0.3)
# 获取读写器版本
rdversion = create_string_buffer(10 * 3)
apiversion = create_string_buffer(10 * 3)
re = library.JT_ReaderVersion(
    Handle, rdversion, len(rdversion), apiversion, len(apiversion)
)
print("读写器版本信息：")
print(rdversion.value, apiversion.value)


def psamType(psamlenrep, handle, nSockID):
    psamreply = create_string_buffer(256)
    psamlenrep = ctypes.c_int()
    command = b"00A40000023F00"
    re = library.JT_SamCommand(
        Handle,
        nSockID,
        ctypes.c_char_p(command),
        len(command),
        psamreply,
        ctypes.byref(psamlenrep),
    )
    # iprint(re)
    # print(psamreply.value)
    command = b"00A40000020015"
    re = library.JT_SamCommand(
        Handle,
        nSockID,
        ctypes.c_char_p(command),
        len(command),
        psamreply,
        ctypes.byref(psamlenrep),
    )
    # print(re)
    # print(psamreply.value)
    command = b"00B000000E"
    re = library.JT_SamCommand(
        Handle,
        nSockID,
        ctypes.c_char_p(command),
        len(command),
        psamreply,
        ctypes.byref(psamlenrep),
    )
    print(re)
    print(psamreply.value)
    filedate = psamreply.value
    psamType = filedate[20:22]
    print("PSAM卡号：" + filedate[0:20])
    if psamType == "05":
        print("版本号：" + psamType + "，国密PSAM")
    else:
        print("版本号：" + psamType + "，非国密PSAM")
    return re


for i in range(1, 5):
    time.sleep(0.5)
    re = library.JT_SamReset(Handle, i, 0)
    print("=" * 60)
    if re == 0:
        print("复位卡槽：" + str(i) + "成功!")
        psamlenrep = ctypes.c_int()
        psamType(psamlenrep, Handle, i)
    else:
        print("复位卡槽:" + str(i) + "败或没有卡!")
