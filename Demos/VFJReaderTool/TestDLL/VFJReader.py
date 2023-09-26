# -*- coding: utf-8 -*-

"""VFJ读写器类，判断不同系统返回不同的读写器"""
import os.path
import ctypes
from pathlib import Path
from ctypes import create_string_buffer
import platform
import re
# import numpy as np



def loadlibrary(systemtype, cn, dllpath="./DLL"):
    if systemtype not in ("Windows", "Linux"):
        raise ValueError("TYPE只能是win或centOS", systemtype)
    if cn not in ("GZ", "CQ"):
        raise ValueError("目前只支持贵州和重庆读写器，请设置cn为GZ或CQ", cn)

    my_file = Path(dllpath)
    
    if not my_file.is_dir():
        raise Exception("加载DLL失败，找不到文件夹:", dllpath)
    if systemtype == "Windows":
        if cn == "GZ":
            path = os.path.join(my_file, "ICC_VFJGZ.dll")
        elif cn == "CQ":
            path = os.path.join(my_file, "ICC_VFJCQ.dll")
        else:
            pass
    elif systemtype == "Linux":
        if cn == "GZ":
            path = os.path.join(my_file, "libICC_VFJGZ.so")
        elif cn == "CQ":
            path = os.path.join(my_file, "libICC_VFJCQ.so")
        else:
            pass
    #print("动态库地址:s%"%path)
    readerlibrary = ctypes.cdll.LoadLibrary(path)
    return readerlibrary

def openreader_test(readerlibrarytest, com):
    com = bytes(com, 'utf-8')
    openstat = readerlibrarytest.JT_OpenReader(ctypes.c_int(0), ctypes.c_char_p(com))
    nHandle = int(openstat)
    return nHandle

def readerversion_test(readerlibrarytest, com, rdversion, rdvermxlen, apiversion, apivermaxlen):
    stat = readerlibrarytest.JT_ReaderVersion(ctypes.c_int(com), rdversion, ctypes.c_int(rdvermxlen), apiversion, ctypes.c_int(apivermaxlen))
    return stat

def closereader_test(readerlibrarytest, com):
    closereaderstat = readerlibrarytest.JT_CloseReader(ctypes.c_int(com))
    return closereaderstat
    
    



class VFJReader():
    """ """
    # systemtype = None
    # pointer = b"com1"
    # cn = None
    # readerlibrary = None
    # statuscode = None
    # nHandle = None
    def __init__(self, systemtype, cn, pointer, dllpath="./DLL"):
        self.exist = 0
        self.pointer = pointer
        self.systemtype = systemtype
        self.cn = cn
        if systemtype not in ("Windows", "Linux"):
            raise ValueError("TYPE只能是win或centOS", systemtype)
        if cn not in ("GZ", "CQ"):
            raise ValueError("目前只支持贵州和重庆读写器，请设置cn为GZ或CQ", cn)

        my_file = Path(dllpath)
        if not my_file.is_dir():
            raise Exception("加载DLL失败，找不到文件夹:", dllpath)
        if systemtype == "Windows":
            if cn == "GZ":
                path = os.path.join(my_file, "ICC_VFJGZ.dll")
            elif cn == "CQ":
                path = os.path.join(my_file, "ICC_VFJCQ.dll")
            else:
                pass
        elif systemtype == "Linux":
            if cn == "GZ":
                path = os.path.join(my_file, "libICC_VFJGZ.so")
            elif cn == "CQ":
                path = os.path.join(my_file, "libICC_VFJCQ.so")
            else:
                pass
        self.readerlibrary = ctypes.cdll.LoadLibrary(path)
        #print(path)
        #print("加载的动态库是："+str(ctypes.LibraryLoader.__repr__))
        # LIBRARY.reverse.argtypes = [c_int,c_char_p]

        # return self.LIBRARY


    def __del__(self):
        self.closereader(self.nHandle if self.nHandle else 1)
        if re.search('Linux', platform.platform()):
            # 如果是Linux系统，使用ctypes.CDLL()._handle释放动态库
            ctypes.CDLL('libc.so.6').free(self.readerlibrary._handle)
            print("linux释放动态库")
        elif re.search('Win', platform.platform()):
            # 如果是Windows系统，使用ctypes.windll.kernel32.FreeLibrary(handle)释放动态库
            ctypes.windll.kernel32.FreeLibrary(self.readerlibrary._handle)
            print("window释放动态库")
        else:
            print("Unsupported system")

        


    # 打开读写器
    def openreader(self):
        openstat = self.readerlibrary.JT_OpenReader(ctypes.c_int(0), ctypes.c_char_p(self.pointer))
        self.nHandle = int(openstat)
        return openstat

    def closereader(self, nHandle = 1):
        # if self.nHandle is None:
        #     return 0
        # else:
        closereaderstat = self.readerlibrary.JT_CloseReader(ctypes.c_int(nHandle))
        return closereaderstat

    def leddisplay(self, nHandle, red, green, blue):
        stat = self.readerlibrary.JT_LEDDisplay(ctypes.c_int(nHandle), ctypes.c_int(red), ctypes.c_int(green), ctypes.c_int(blue))
        return stat

    def audiocontrol(self, nHandle, time, voice):
        stat = self.readerlibrary.JT_AudioControl(ctypes.c_int(nHandle), ctypes.c_int(time),ctypes.c_int(voice))
        return stat

    def getstatus(self, nHandle, statuscode):
        stat = self.readerlibrary.JT_GetStatus(ctypes.c_int(nHandle), ctypes.byref(statuscode))
        self.statuscode = statuscode.value
        return stat

    def readerversion(self, nHandle, rdversion, rdvermxlen, apiversion, apivermaxlen):
        # stat = self.readerlibrary.JT_ReaderVersion(ctypes.c_int(nHandle), ctypes.c_char_p(rdversion), ctypes.c_int(rdvermxlen), ctypes.c_char_p(apiversion), ctypes.c_int(apivermaxlen))
        stat = self.readerlibrary.JT_ReaderVersion(ctypes.c_int(nHandle), rdversion, ctypes.c_int(rdvermxlen), apiversion, ctypes.c_int(apivermaxlen))
        return stat

    def getversion(self, version, verlen):
        stat = self.readerlibrary.JT_GetVersion(version, ctypes.c_int(verlen))
        return stat

    def getstatusmsg(self, ststuscode, ststusmag, maglen):
        stat = self.readerlibrary.JT_GetStatusMsg(ctypes.c_int(ststuscode), ststusmag, ctypes.c_int(maglen))
        return stat

    def opencard(self, nHandle, cardplace, cardsn):
        stat = self.readerlibrary.JT_OpenCard(ctypes.c_int(nHandle), ctypes.byref(cardplace), cardsn)
        return stat

    def closecard(self, nHandle):
        stat = self.readerlibrary.JT_CloseCard(ctypes.c_int(nHandle))
        return stat

    def cpucommandself(self, nHandle, command, lencom, reply, lenrep):
        # stat = self.readerlibrary.JT_CPUCommand(ctypes.c_int(nHandle), ctypes.c_char_p(command), ctypes.c_int(lencom), reply, ctypes.byref(lenrep))
        stat = self.readerlibrary.JT_CPUCommand(ctypes.c_int(nHandle), ctypes.c_char_p(command), ctypes.c_int(lencom), reply, ctypes.byref(lenrep))
        return stat

    def samreset(self, nHandle, nSockID, protocoltype):
        stat = self.readerlibrary.JT_SamReset(ctypes.c_int(nHandle), ctypes.c_int(nSockID), ctypes.c_int(protocoltype))
        return stat

    def samcommand(self, nHandle, nSockID, command, lencom, reply, lenrep):
        stat = self.readerlibrary.JT_SamCommand(ctypes.c_int(nHandle), ctypes.c_int(nSockID), ctypes.c_char_p(command), ctypes.c_int(lencom), reply, ctypes.byref(lenrep))
        return stat
    
    def resetRF(self, nHandle):
        stat = self.readerlibrary.JT_ResetRF(ctypes.c_int(nHandle))
        return stat
    
    def selectRWUnit(self, nHandle, Unit):
        stat = self.readerlibrary.JT_SelectRWUnit(ctypes.c_int(nHandle), ctypes.c_ubyte(Unit))
        return stat
    
    def setScanRWUnits(self, nHandle, Unit):
        stat = self.readerlibrary.JT_SetScanRWUnits(ctypes.c_int(nHandle), ctypes.c_ubyte(Unit))
        return stat





if __name__ == '__main__':
    try:
        #实例读写器
        reader = VFJReader("Windows", "CQ", b"com3")
        arr = ctypes.c_int()
        #打开读写器
        openstat = reader.openreader()
        print(openstat)
        #获取读写器状态
        reader.getstatus(openstat, arr)
        print("读写器状态：" + str(arr.value))
        #设置LED
        stat = reader.leddisplay(openstat, 1, 2, 1)
        print("设置LED状态：" + str(stat))
        #设置字符串缓存区
        rdversion = create_string_buffer(10 * 3)
        apiversion = create_string_buffer(10 * 3)
        #读写器版本
        readerversion = reader.readerversion(openstat, rdversion, len(rdversion), apiversion, len(apiversion))
        print(rdversion.value, apiversion.value)
        version = create_string_buffer(10 * 3)
        #动态库版本
        dllversion = reader.getversion(version, len(version))
        print(version.value)
# =============================================================================
#         #打开卡片
#         cardplace = ctypes.c_int()
#         cardsn = ctypes.create_string_buffer(8)
#         opencardstatus = reader.opencard(openstat, cardplace, cardsn)
#         print("卡片SNR：" + str(cardsn.value))
#         print(cardplace.value)
#         print(opencardstatus)
#         #执行命令
#         command = b"00A4000002DF01"
#         reply = create_string_buffer(256)
#         lenrep = ctypes.c_int()
#         SS = reader.cpucommandself(openstat, command, len(command), reply, lenrep)
#         print(SS)
#         print(reply.value)
#         #关闭卡片
#         closecardststus = reader.closecard(openstat)
#         print("关闭卡片" + str(closecardststus))
#         #获取状态说明
#         ststusmag = create_string_buffer(10 * 3)
#         getstatusmsg = reader.getstatusmsg(opencardstatus,ststusmag,len(ststusmag))
#         print(ststusmag.value.decode(encoding='GB18030',errors="replace"))
# =============================================================================
        #复位psam
        samresetsata = reader.samreset(openstat, 1, 0)
        print(samresetsata)
        #PSAM卡测试
        psamreply = create_string_buffer(256)
        psamlenrep = ctypes.c_int()
        command2 = b"00A40000023F00"
        aa = reader.samcommand(openstat, 1, command2, len(command2), psamreply, psamlenrep)
        print(aa)
        print(psamreply.value)
        
        reader.closereader(1)
    finally:
        del reader


    
