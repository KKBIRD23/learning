#-*- coding:utf-8 -*-

import sys
import platform
import re
from VFJReader import VFJReader,loadlibrary,openreader_test,readerversion_test,closereader_test
from ctypes import create_string_buffer
import ctypes
import time
import threading

if sys.version_info[0] == 2:
    from Tkinter import *
    import Tkinter as tk
    from tkFont import Font
    from ttk import *
    #Usage:showinfo/warning/error,askquestion/okcancel/yesno/retrycancel
    from tkMessageBox import *
    #Usage:f=tkFileDialog.askopenfilename(initialdir='E:/Python')
    #import tkFileDialog
    #import tkSimpleDialog
else:  #Python 3.x
    from tkinter import *
    from tkinter.font import Font
    from tkinter.ttk import *
    from tkinter.messagebox import *
    from tkinter import  messagebox
    import tkinter as tk
    #import tkinter.filedialog as tkFileDialog
    #import tkinter.simpledialog as tkSimpleDialog    #askstring()

class Application_ui(Frame):
    #这个类仅实现界面生成功能，具体事件处理代码在子类Application中。
    Handle = 0
    lock = False
    i = 0
    success = 0
    failure = 0
    
    
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('VFJ读写器综合工具')
        self.master.geometry('825x614')
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        self.style = Style()

        self.style.configure('TFrame4.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame4.TLabelframe.Label', font=('宋体',9))
        self.Frame4 = LabelFrame(self.top, text='天线设置', style='TFrame4.TLabelframe')
        self.Frame4.place(relx=0.611, rely=0.003, relwidth=0.37, relheight=0.181)

        self.style.configure('TFrame1.TLabelframe', font=('微软雅黑',9))
        self.style.configure('TFrame1.TLabelframe.Label', font=('微软雅黑',9))
        self.Frame1 = LabelFrame(self.top, text='基本设置', style='TFrame1.TLabelframe')
        self.Frame1.place(relx=0.019, rely=0., relwidth=0.573, relheight=0.184)

        self.style.configure('TFrame3.TLabelframe', font=('微软雅黑',9))
        self.style.configure('TFrame3.TLabelframe.Label', font=('微软雅黑',9))
        self.Frame3 = LabelFrame(self.top, text='返回：', style='TFrame3.TLabelframe')
        self.Frame3.place(relx=0.019, rely=0.56, relwidth=0.961, relheight=0.432)

        self.TabStrip1 = Notebook(self.top)
        self.TabStrip1.place(relx=0.019, rely=0.195, relwidth=0.961, relheight=0.353)
        self.TabStrip1.bind('<<NotebookTabChanged>>', self.TabStrip1_NotebookTabChanged)

        self.TabStrip1__Tab1 = Frame(self.TabStrip1)
        self.style.configure('TFrame6.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame6.TLabelframe.Label', font=('宋体',9))
        self.Frame6 = LabelFrame(self.TabStrip1__Tab1, text='', style='TFrame6.TLabelframe')
        self.Frame6.place(relx=0.347, rely=0.086, relwidth=0.622, relheight=0.827)
        self.P1_Text_Buzzer2Var = StringVar(value='2')
        self.P1_Text_Buzzer2 = Entry(self.Frame6, textvariable=self.P1_Text_Buzzer2Var, font=('微软雅黑',9))
        self.P1_Text_Buzzer2.setText = lambda x: self.P1_Text_Buzzer2Var.set(x)
        self.P1_Text_Buzzer2.text = lambda : self.P1_Text_Buzzer2Var.get()
        self.P1_Text_Buzzer2.place(relx=0.795, rely=0.654, relwidth=0.053, relheight=0.163)
        self.P1_Text_Buzzer1Var = StringVar(value='2')
        self.P1_Text_Buzzer1 = Entry(self.Frame6, textvariable=self.P1_Text_Buzzer1Var, font=('微软雅黑',9))
        self.P1_Text_Buzzer1.setText = lambda x: self.P1_Text_Buzzer1Var.set(x)
        self.P1_Text_Buzzer1.text = lambda : self.P1_Text_Buzzer1Var.get()
        self.P1_Text_Buzzer1.place(relx=0.575, rely=0.654, relwidth=0.053, relheight=0.163)
        self.P1_Command_DiodeVar = StringVar(value='发光二极管')
        self.style.configure('TP1_Command_Diode.TButton', font=('宋体',9))
        self.P1_Command_Diode = Button(self.Frame6, text='发光二极管', textvariable=self.P1_Command_DiodeVar, command=self.P1_Command_Diode_Cmd, style='TP1_Command_Diode.TButton')
        self.P1_Command_Diode.setText = lambda x: self.P1_Command_DiodeVar.set(x)
        self.P1_Command_Diode.text = lambda : self.P1_Command_DiodeVar.get()
        self.P1_Command_Diode.place(relx=0.152, rely=0.261, relwidth=0.188, relheight=0.216)
        self.P1_Command_BuzzerVar = StringVar(value='蜂鸣器')
        self.style.configure('TP1_Command_Buzzer.TButton', font=('宋体',9))
        self.P1_Command_Buzzer = Button(self.Frame6, text='蜂鸣器', textvariable=self.P1_Command_BuzzerVar, command=self.P1_Command_Buzzer_Cmd, style='TP1_Command_Buzzer.TButton')
        self.P1_Command_Buzzer.setText = lambda x: self.P1_Command_BuzzerVar.set(x)
        self.P1_Command_Buzzer.text = lambda : self.P1_Command_BuzzerVar.get()
        self.P1_Command_Buzzer.place(relx=0.152, rely=0.627, relwidth=0.188, relheight=0.216)
        self.P1_Text_Diode1Var = StringVar(value='2')
        self.P1_Text_Diode1 = Entry(self.Frame6, textvariable=self.P1_Text_Diode1Var, font=('微软雅黑',9))
        self.P1_Text_Diode1.setText = lambda x: self.P1_Text_Diode1Var.set(x)
        self.P1_Text_Diode1.text = lambda : self.P1_Text_Diode1Var.get()
        self.P1_Text_Diode1.place(relx=0.474, rely=0.261, relwidth=0.053, relheight=0.163)
        self.P1_Text_Diode2Var = StringVar(value='2')
        self.P1_Text_Diode2 = Entry(self.Frame6, textvariable=self.P1_Text_Diode2Var, font=('微软雅黑',9))
        self.P1_Text_Diode2.setText = lambda x: self.P1_Text_Diode2Var.set(x)
        self.P1_Text_Diode2.text = lambda : self.P1_Text_Diode2Var.get()
        self.P1_Text_Diode2.place(relx=0.626, rely=0.261, relwidth=0.053, relheight=0.163)
        self.P1_Text_Diode3Var = StringVar(value='2')
        self.P1_Text_Diode3 = Entry(self.Frame6, textvariable=self.P1_Text_Diode3Var, font=('微软雅黑',9))
        self.P1_Text_Diode3.setText = lambda x: self.P1_Text_Diode3Var.set(x)
        self.P1_Text_Diode3.text = lambda : self.P1_Text_Diode3Var.get()
        self.P1_Text_Diode3.place(relx=0.778, rely=0.261, relwidth=0.053, relheight=0.163)
        self.Label10Var = StringVar(value='声调')
        self.style.configure('TLabel10.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label10 = Label(self.Frame6, text='声调', textvariable=self.Label10Var, style='TLabel10.TLabel')
        self.Label10.setText = lambda x: self.Label10Var.set(x)
        self.Label10.text = lambda : self.Label10Var.get()
        self.Label10.place(relx=0.71, rely=0.654, relwidth=0.07, relheight=0.163)
        self.Label9Var = StringVar(value='发音次数')
        self.style.configure('TLabel9.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label9 = Label(self.Frame6, text='发音次数', textvariable=self.Label9Var, style='TLabel9.TLabel')
        self.Label9.setText = lambda x: self.Label9Var.set(x)
        self.Label9.text = lambda : self.Label9Var.get()
        self.Label9.place(relx=0.44, rely=0.654, relwidth=0.121, relheight=0.163)
        self.Label6Var = StringVar(value='R')
        self.style.configure('TLabel6.TLabel', anchor='w', font=('宋体',9))
        self.Label6 = Label(self.Frame6, text='R', textvariable=self.Label6Var, style='TLabel6.TLabel')
        self.Label6.setText = lambda x: self.Label6Var.set(x)
        self.Label6.text = lambda : self.Label6Var.get()
        self.Label6.place(relx=0.44, rely=0.314, relwidth=0.019, relheight=0.111)
        self.Label7Var = StringVar(value='G')
        self.style.configure('TLabel7.TLabel', anchor='w', font=('宋体',9))
        self.Label7 = Label(self.Frame6, text='G', textvariable=self.Label7Var, style='TLabel7.TLabel')
        self.Label7.setText = lambda x: self.Label7Var.set(x)
        self.Label7.text = lambda : self.Label7Var.get()
        self.Label7.place(relx=0.592, rely=0.314, relwidth=0.019, relheight=0.111)
        self.Label8Var = StringVar(value='Y')
        self.style.configure('TLabel8.TLabel', anchor='w', font=('宋体',9))
        self.Label8 = Label(self.Frame6, text='Y', textvariable=self.Label8Var, style='TLabel8.TLabel')
        self.Label8.setText = lambda x: self.Label8Var.set(x)
        self.Label8.text = lambda : self.Label8Var.get()
        self.Label8.place(relx=0.744, rely=0.314, relwidth=0.019, relheight=0.111)
        self.style.configure('TFrame5.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame5.TLabelframe.Label', font=('宋体',9))
        self.Frame5 = LabelFrame(self.TabStrip1__Tab1, text='', style='TFrame5.TLabelframe')
        self.Frame5.place(relx=0.042, rely=0.086, relwidth=0.275, relheight=0.827)
        self.P1_Command_ReaderVerVar = StringVar(value='读写器版本查询')
        self.style.configure('TP1_Command_ReaderVer.TButton', font=('宋体',9))
        self.P1_Command_ReaderVer = Button(self.Frame5, text='读写器版本查询', textvariable=self.P1_Command_ReaderVerVar, command=self.P1_Command_ReaderVer_Cmd, style='TP1_Command_ReaderVer.TButton')
        self.P1_Command_ReaderVer.setText = lambda x: self.P1_Command_ReaderVerVar.set(x)
        self.P1_Command_ReaderVer.text = lambda : self.P1_Command_ReaderVerVar.get()
        self.P1_Command_ReaderVer.place(relx=0.191, rely=0.157, relwidth=0.579, relheight=0.268)
        self.P1_Command_DllVersionVar = StringVar(value='动态库版本查询')
        self.style.configure('TP1_Command_DllVersion.TButton', font=('宋体',9))
        self.P1_Command_DllVersion = Button(self.Frame5, text='动态库版本查询', textvariable=self.P1_Command_DllVersionVar, command=self.P1_Command_DllVersion_Cmd, style='TP1_Command_DllVersion.TButton')
        self.P1_Command_DllVersion.setText = lambda x: self.P1_Command_DllVersionVar.set(x)
        self.P1_Command_DllVersion.text = lambda : self.P1_Command_DllVersionVar.get()
        self.P1_Command_DllVersion.place(relx=0.191, rely=0.575, relwidth=0.579, relheight=0.268)
        self.TabStrip1.add(self.TabStrip1__Tab1, text='基本功能')

        self.TabStrip1__Tab2 = Frame(self.TabStrip1)
        self.style.configure('TFrame7.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame7.TLabelframe.Label', font=('宋体',9))
        self.Frame7 = LabelFrame(self.TabStrip1__Tab2, text='', style='TFrame7.TLabelframe')
        self.Frame7.place(relx=0.357, rely=0.043, relwidth=0.622, relheight=0.914)
        self.P2_Command_3F00Var = StringVar(value='选择 3F00')
        self.style.configure('TP2_Command_3F00.TButton', font=('微软雅黑',9))
        self.P2_Command_3F00 = Button(self.Frame7, text='选择 3F00', textvariable=self.P2_Command_3F00Var, command=self.P2_Command_3F00_Cmd, style='TP2_Command_3F00.TButton')
        self.P2_Command_3F00.setText = lambda x: self.P2_Command_3F00Var.set(x)
        self.P2_Command_3F00.text = lambda : self.P2_Command_3F00Var.get()
        self.P2_Command_3F00.place(relx=0.034, rely=0.142, relwidth=0.188, relheight=0.195)
        self.P2_Command_1001Var = StringVar(value='选择 1001')
        self.style.configure('TP2_Command_1001.TButton', font=('微软雅黑',9))
        self.P2_Command_1001 = Button(self.Frame7, text='选择 1001', textvariable=self.P2_Command_1001Var, command=self.P2_Command_1001_Cmd, style='TP2_Command_1001.TButton')
        self.P2_Command_1001.setText = lambda x: self.P2_Command_1001Var.set(x)
        self.P2_Command_1001.text = lambda : self.P2_Command_1001Var.get()
        self.P2_Command_1001.place(relx=0.271, rely=0.142, relwidth=0.188, relheight=0.195)
        self.P2_Command_1005Var = StringVar(value='读 0015')
        self.style.configure('TP2_Command_1005.TButton', font=('微软雅黑',9))
        self.P2_Command_1005 = Button(self.Frame7, text='读 0015', textvariable=self.P2_Command_1005Var, command=self.P2_Command_1005_Cmd, style='TP2_Command_1005.TButton')
        self.P2_Command_1005.setText = lambda x: self.P2_Command_1005Var.set(x)
        self.P2_Command_1005.text = lambda : self.P2_Command_1005Var.get()
        self.P2_Command_1005.place(relx=0.271, rely=0.426, relwidth=0.188, relheight=0.195)
        self.P2_Command_1006Var = StringVar(value='读 0016')
        self.style.configure('TP2_Command_1006.TButton', font=('微软雅黑',9))
        self.P2_Command_1006 = Button(self.Frame7, text='读 0016', textvariable=self.P2_Command_1006Var, command=self.P2_Command_1006_Cmd, style='TP2_Command_1006.TButton')
        self.P2_Command_1006.setText = lambda x: self.P2_Command_1006Var.set(x)
        self.P2_Command_1006.text = lambda : self.P2_Command_1006Var.get()
        self.P2_Command_1006.place(relx=0.271, rely=0.71, relwidth=0.188, relheight=0.195)
        self.P2_Command_DF01Var = StringVar(value='选择 DF01')
        self.style.configure('TP2_Command_DF01.TButton', font=('微软雅黑',9))
        self.P2_Command_DF01 = Button(self.Frame7, text='选择 DF01', textvariable=self.P2_Command_DF01Var, command=self.P2_Command_DF01_Cmd, style='TP2_Command_DF01.TButton')
        self.P2_Command_DF01.setText = lambda x: self.P2_Command_DF01Var.set(x)
        self.P2_Command_DF01.text = lambda : self.P2_Command_DF01Var.get()
        self.P2_Command_DF01.place(relx=0.524, rely=0.142, relwidth=0.188, relheight=0.195)
        self.P2_Command_EF01Var = StringVar(value='读 EF01')
        self.style.configure('TP2_Command_EF01.TButton', font=('微软雅黑',9))
        self.P2_Command_EF01 = Button(self.Frame7, text='读 EF01', textvariable=self.P2_Command_EF01Var, command=self.P2_Command_EF01_Cmd, style='TP2_Command_EF01.TButton')
        self.P2_Command_EF01.setText = lambda x: self.P2_Command_EF01Var.set(x)
        self.P2_Command_EF01.text = lambda : self.P2_Command_EF01Var.get()
        self.P2_Command_EF01.place(relx=0.761, rely=0.142, relwidth=0.188, relheight=0.195)
        self.P2_Command_EF02Var = StringVar(value='读 EF02')
        self.style.configure('TP2_Command_EF02.TButton', font=('微软雅黑',9))
        self.P2_Command_EF02 = Button(self.Frame7, text='读 EF02', textvariable=self.P2_Command_EF02Var, command=self.P2_Command_EF02_Cmd, style='TP2_Command_EF02.TButton')
        self.P2_Command_EF02.setText = lambda x: self.P2_Command_EF02Var.set(x)
        self.P2_Command_EF02.text = lambda : self.P2_Command_EF02Var.get()
        self.P2_Command_EF02.place(relx=0.761, rely=0.426, relwidth=0.188, relheight=0.195)
        self.P2_Command_EF04Var = StringVar(value='读 EF04')
        self.style.configure('TP2_Command_EF04.TButton', font=('微软雅黑',9))
        self.P2_Command_EF04 = Button(self.Frame7, text='读 EF04', textvariable=self.P2_Command_EF04Var, command=self.P2_Command_EF04_Cmd, style='TP2_Command_EF04.TButton')
        self.P2_Command_EF04.setText = lambda x: self.P2_Command_EF04Var.set(x)
        self.P2_Command_EF04.text = lambda : self.P2_Command_EF04Var.get()
        self.P2_Command_EF04.place(relx=0.761, rely=0.71, relwidth=0.188, relheight=0.195)
        self.style.configure('TFrame2.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame2.TLabelframe.Label', font=('宋体',9))
        self.Frame2 = LabelFrame(self.TabStrip1__Tab2, text='', style='TFrame2.TLabelframe')
        self.Frame2.place(relx=0.021, rely=0.043, relwidth=0.285, relheight=0.914)
        self.P2_Command_OpenCardVar = StringVar(value='打开卡片')
        self.style.configure('TP2_Command_OpenCard.TButton', font=('微软雅黑',9))
        self.P2_Command_OpenCard = Button(self.Frame2, text='打开卡片', textvariable=self.P2_Command_OpenCardVar, command=self.P2_Command_OpenCard_Cmd, style='TP2_Command_OpenCard.TButton')
        self.P2_Command_OpenCard.setText = lambda x: self.P2_Command_OpenCardVar.set(x)
        self.P2_Command_OpenCard.text = lambda : self.P2_Command_OpenCardVar.get()
        self.P2_Command_OpenCard.place(relx=0.074, rely=0.189, relwidth=0.41, relheight=0.243)
        self.P2_Command_CloseCardVar = StringVar(value='关闭卡片')
        self.style.configure('TP2_Command_CloseCard.TButton', font=('微软雅黑',9))
        self.P2_Command_CloseCard = Button(self.Frame2, text='关闭卡片', textvariable=self.P2_Command_CloseCardVar, command=self.P2_Command_CloseCard_Cmd, style='TP2_Command_CloseCard.TButton')
        self.P2_Command_CloseCard.setText = lambda x: self.P2_Command_CloseCardVar.set(x)
        self.P2_Command_CloseCard.text = lambda : self.P2_Command_CloseCardVar.get()
        self.P2_Command_CloseCard.place(relx=0.074, rely=0.568, relwidth=0.41, relheight=0.243)
        self.P2_Command_RFresetVar = StringVar(value='射频复位')
        self.style.configure('TP2_Command_RFreset.TButton', font=('微软雅黑',9))
        self.P2_Command_RFreset = Button(self.Frame2, text='射频复位', textvariable=self.P2_Command_RFresetVar, command=self.P2_Command_RFreset_Cmd, style='TP2_Command_RFreset.TButton')
        self.P2_Command_RFreset.setText = lambda x: self.P2_Command_RFresetVar.set(x)
        self.P2_Command_RFreset.text = lambda : self.P2_Command_RFresetVar.get()
        self.P2_Command_RFreset.place(relx=0.59, rely=0.189, relwidth=0.3, relheight=0.621)
        self.TabStrip1.add(self.TabStrip1__Tab2, text='卡片测试')

        self.TabStrip1__Tab3 = Frame(self.TabStrip1)
        self.style.configure('TFrame9.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame9.TLabelframe.Label', font=('宋体',9))
        self.Frame9 = LabelFrame(self.TabStrip1__Tab3, text='PSAM Test', style='TFrame9.TLabelframe')
        self.Frame9.place(relx=0., rely=0.043, relwidth=0.327, relheight=0.957)
        # self.P3_Text_CommandVar = StringVar(value='11')
        # self.P3_Text_Command = Entry(self.Frame9, textvariable=self.P3_Text_CommandVar, font=('宋体',9))
        # self.P3_Text_Command.setText = lambda x: self.P3_Text_CommandVar.set(x)
        # self.P3_Text_Command.text = lambda : self.P3_Text_CommandVar.get()
        # self.P3_Text_Command.place(relx=0.193, rely=0.588, relwidth=0.112, relheight=0.102)
        self.Text7Var = StringVar(value='00A400023F00')
        self.Text7 = Entry(self.Frame9, textvariable=self.Text7Var, font=('微软雅黑',9))
        self.Text7.setText = lambda x: self.Text7Var.set(x)
        self.Text7.text = lambda : self.Text7Var.get()
        self.Text7.place(relx=0., rely=0.723, relwidth=1., relheight=0.322)
        self.P3_Command_RFpsamVar = StringVar(value='PSAM复位')
        self.style.configure('TP3_Command_RFpsam.TButton', font=('宋体',9))
        self.P3_Command_RFpsam = Button(self.Frame9, text='PSAM复位', textvariable=self.P3_Command_RFpsamVar, command=self.P3_Command_RFpsam_Cmd, style='TP3_Command_RFpsam.TButton')
        self.P3_Command_RFpsam.setText = lambda x: self.P3_Command_RFpsamVar.set(x)
        self.P3_Command_RFpsam.text = lambda : self.P3_Command_RFpsamVar.get()
        self.P3_Command_RFpsam.place(relx=0.064, rely=0.158, relwidth=0.325, relheight=0.141)
        self.P3_Command_PSAMcommandVar = StringVar(value='PSAM指令')
        self.style.configure('TP3_Command_PSAMcommand.TButton', font=('宋体',9))
        self.P3_Command_PSAMcommand = Button(self.Frame9, text='PSAM指令', textvariable=self.P3_Command_PSAMcommandVar, command=self.P3_Command_PSAMcommand_Cmd, style='TP3_Command_PSAMcommand.TButton')
        self.P3_Command_PSAMcommand.setText = lambda x: self.P3_Command_PSAMcommandVar.set(x)
        self.P3_Command_PSAMcommand.text = lambda : self.P3_Command_PSAMcommandVar.get()
        self.P3_Command_PSAMcommand.place(relx=0.514, rely=0.362, relwidth=0.325, relheight=0.141)
        self.P3_Combo_PSAMpositionList = ['1 ','2 ','3 ','4 ',]
        self.P3_Combo_PSAMpositionVar = StringVar(value='1 ')
        self.P3_Combo_PSAMposition = Combobox(self.Frame9, text='1 ', textvariable=self.P3_Combo_PSAMpositionVar, values=self.P3_Combo_PSAMpositionList, font=('宋体',9))
        self.P3_Combo_PSAMposition.setText = lambda x: self.P3_Combo_PSAMpositionVar.set(x)
        self.P3_Combo_PSAMposition.text = lambda : self.P3_Combo_PSAMpositionVar.get()
        self.P3_Combo_PSAMposition.place(relx=0.675, rely=0.169, relwidth=0.133)
        self.P3_Command_CPUcommandVar = StringVar(value='CPU指令')
        self.style.configure('TP3_Command_CPUcommand.TButton', font=('微软雅黑',9))
        self.P3_Command_CPUcommand = Button(self.Frame9, text='CPU指令', textvariable=self.P3_Command_CPUcommandVar, command=self.P3_Command_CPUcommand_Cmd, style='TP3_Command_CPUcommand.TButton')
        self.P3_Command_CPUcommand.setText = lambda x: self.P3_Command_CPUcommandVar.set(x)
        self.P3_Command_CPUcommand.text = lambda : self.P3_Command_CPUcommandVar.get()
        self.P3_Command_CPUcommand.place(relx=0.064, rely=0.362, relwidth=0.325, relheight=0.141)
        self.Label12Var = StringVar(value='指令：')
        self.style.configure('TLabel12.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label12 = Label(self.Frame9, text='指令：', textvariable=self.Label12Var, style='TLabel12.TLabel')
        self.Label12.setText = lambda x: self.Label12Var.set(x)
        self.Label12.text = lambda : self.Label12Var.get()
        self.Label12.place(relx=0.032, rely=0.588, relwidth=0.165, relheight=0.096)
        self.Label11Var = StringVar(value='卡槽：')
        self.style.configure('TLabel11.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label11 = Label(self.Frame9, text='卡槽：', textvariable=self.Label11Var, style='TLabel11.TLabel')
        self.Label11.setText = lambda x: self.Label11Var.set(x)
        self.Label11.text = lambda : self.Label11Var.get()
        self.Label11.place(relx=0.514, rely=0.181, relwidth=0.165, relheight=0.096)
        self.style.configure('TFrame10.TLabelframe', font=('宋体',9))
        self.style.configure('TFrame10.TLabelframe.Label', font=('宋体',9))
        self.Frame10 = LabelFrame(self.TabStrip1__Tab3, text='卡片连续测试', style='TFrame10.TLabelframe')
        self.Frame10.place(relx=0.336, rely=0.043, relwidth=0.664, relheight=0.957)
        self.P3_Command_StopTestVar = StringVar(value='停止测试')
        self.style.configure('TP3_Command_StopTest.TButton', font=('微软雅黑',9))
        self.P3_Command_StopTest = Button(self.Frame10, text='停止测试', textvariable=self.P3_Command_StopTestVar, command=self.P3_Command_StopTest_Cmd, style='TP3_Command_StopTest.TButton')
        self.P3_Command_StopTest.setText = lambda x: self.P3_Command_StopTestVar.set(x)
        self.P3_Command_StopTest.text = lambda : self.P3_Command_StopTestVar.get()
        self.P3_Command_StopTest.place(relx=0.792, rely=0.588, relwidth=0.176, relheight=0.277)
        self.P3_Command_StartTestVar = StringVar(value='开始测试')
        self.style.configure('TP3_Command_StartTest.TButton', font=('微软雅黑',9))
        self.P3_Command_StartTest = Button(self.Frame10, text='开始测试', textvariable=self.P3_Command_StartTestVar, command=lambda :self.asynchronous(self.P3_Command_StartTest_Cmd), style='TP3_Command_StartTest.TButton')
        self.P3_Command_StartTest.setText = lambda x: self.P3_Command_StartTestVar.set(x)
        self.P3_Command_StartTest.text = lambda : self.P3_Command_StartTestVar.get()
        self.P3_Command_StartTest.place(relx=0.792, rely=0.181, relwidth=0.176, relheight=0.277)
        self.P3_Text_BlocksVar = StringVar(value='3')
        self.P3_Text_Blocks = Entry(self.Frame10, textvariable=self.P3_Text_BlocksVar, font=('微软雅黑',9))
        self.P3_Text_Blocks.setText = lambda x: self.P3_Text_BlocksVar.set(x)
        self.P3_Text_Blocks.text = lambda : self.P3_Text_BlocksVar.get()
        self.P3_Text_Blocks.place(relx=0.174, rely=0.791, relwidth=0.129, relheight=0.141)
        self.Text_StarBlockVar = StringVar(value='0')
        self.Text_StarBlock = Entry(self.Frame10, textvariable=self.Text_StarBlockVar, font=('微软雅黑',9))
        self.Text_StarBlock.setText = lambda x: self.Text_StarBlockVar.set(x)
        self.Text_StarBlock.text = lambda : self.Text_StarBlockVar.get()
        self.Text_StarBlock.place(relx=0.174, rely=0.559, relwidth=0.129, relheight=0.141)
        self.P3_Text_MAIDVar = StringVar(value='3001')
        self.P3_Text_MAID = Entry(self.Frame10, textvariable=self.P3_Text_MAIDVar, font=('微软雅黑',9))
        self.P3_Text_MAID.setText = lambda x: self.P3_Text_MAIDVar.set(x)
        self.P3_Text_MAID.text = lambda : self.P3_Text_MAIDVar.get()
        self.P3_Text_MAID.place(relx=0.174, rely=0.328, relwidth=0.129, relheight=0.147)
        self.P3_Text_TestedVar = StringVar(value='0')
        self.P3_Text_Tested = Entry(self.Frame10, textvariable=self.P3_Text_TestedVar, font=('微软雅黑',9))
        self.P3_Text_Tested.setText = lambda x: self.P3_Text_TestedVar.set(x)
        self.P3_Text_Tested.text = lambda : self.P3_Text_TestedVar.get()
        self.P3_Text_Tested.place(relx=0.586, rely=0.768, relwidth=0.081, relheight=0.147)
        self.P3_Text_TestTotalVar = StringVar(value='10')
        self.P3_Text_TestTotal = Entry(self.Frame10, textvariable=self.P3_Text_TestTotalVar, font=('微软雅黑',9))
        self.P3_Text_TestTotal.setText = lambda x: self.P3_Text_TestTotalVar.set(x)
        self.P3_Text_TestTotal.text = lambda : self.P3_Text_TestTotalVar.get()
        self.P3_Text_TestTotal.place(relx=0.586, rely=0.452, relwidth=0.081, relheight=0.147)
        self.P3_Text_TestTimesVar = StringVar(value='5')
        self.P3_Text_TestTimes = Entry(self.Frame10, textvariable=self.P3_Text_TestTimesVar, font=('微软雅黑',9))
        self.P3_Text_TestTimes.setText = lambda x: self.P3_Text_TestTimesVar.set(x)
        self.P3_Text_TestTimes.text = lambda : self.P3_Text_TestTimesVar.get()
        self.P3_Text_TestTimes.place(relx=0.586, rely=0.136, relwidth=0.081, relheight=0.147)
        self.P3_Text_ReadTimerVar = StringVar(value='0.5')
        self.P3_Text_ReadTimer = Entry(self.Frame10, textvariable=self.P3_Text_ReadTimerVar, font=('微软雅黑',9))
        self.P3_Text_ReadTimer.setText = lambda x: self.P3_Text_ReadTimerVar.set(x)
        self.P3_Text_ReadTimer.text = lambda : self.P3_Text_ReadTimerVar.get()
        self.P3_Text_ReadTimer.place(relx=0.238, rely=0.136, relwidth=0.065, relheight=0.147)
        self.Label16Var = StringVar(value='Blocks')
        self.style.configure('TLabel16.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label16 = Label(self.Frame10, text='Blocks', textvariable=self.Label16Var, style='TLabel16.TLabel')
        self.Label16.setText = lambda x: self.Label16Var.set(x)
        self.Label16.text = lambda : self.Label16Var.get()
        self.Label16.place(relx=0.032, rely=0.814, relwidth=0.113, relheight=0.141)
        self.Label14Var = StringVar(value='StarBlock')
        self.style.configure('TLabel14.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label14 = Label(self.Frame10, text='StarBlock', textvariable=self.Label14Var, style='TLabel14.TLabel')
        self.Label14.setText = lambda x: self.Label14Var.set(x)
        self.Label14.text = lambda : self.Label14Var.get()
        self.Label14.place(relx=0.032, rely=0.582, relwidth=0.113, relheight=0.141)
        self.Label15Var = StringVar(value='MAID')
        self.style.configure('TLabel15.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label15 = Label(self.Frame10, text='MAID', textvariable=self.Label15Var, style='TLabel15.TLabel')
        self.Label15.setText = lambda x: self.Label15Var.set(x)
        self.Label15.text = lambda : self.Label15Var.get()
        self.Label15.place(relx=0.032, rely=0.35, relwidth=0.081, relheight=0.141)
        self.Label19Var = StringVar(value='已测试卡数')
        self.style.configure('TLabel19.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label19 = Label(self.Frame10, text='已测试卡数', textvariable=self.Label19Var, style='TLabel19.TLabel')
        self.Label19.setText = lambda x: self.Label19Var.set(x)
        self.Label19.text = lambda : self.Label19Var.get()
        self.Label19.place(relx=0.444, rely=0.794, relwidth=0.145, relheight=0.096)
        self.Label18Var = StringVar(value='总测试卡数')
        self.style.configure('TLabel18.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label18 = Label(self.Frame10, text='总测试卡数', textvariable=self.Label18Var, style='TLabel18.TLabel')
        self.Label18.setText = lambda x: self.Label18Var.set(x)
        self.Label18.text = lambda : self.Label18Var.get()
        self.Label18.place(relx=0.444, rely=0.48, relwidth=0.145, relheight=0.096)
        self.Label17Var = StringVar(value='尝试次数')
        self.style.configure('TLabel17.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label17 = Label(self.Frame10, text='尝试次数', textvariable=self.Label17Var, style='TLabel17.TLabel')
        self.Label17.setText = lambda x: self.Label17Var.set(x)
        self.Label17.text = lambda : self.Label17Var.get()
        self.Label17.place(relx=0.444, rely=0.164, relwidth=0.145, relheight=0.141)
        self.Label13Var = StringVar(value='ReadTimer(s)')
        self.style.configure('TLabel13.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label13 = Label(self.Frame10, text='ReadTimer(ms)', textvariable=self.Label13Var, style='TLabel13.TLabel')
        self.Label13.setText = lambda x: self.Label13Var.set(x)
        self.Label13.text = lambda : self.Label13Var.get()
        self.Label13.place(relx=0.032, rely=0.119, relwidth=0.176, relheight=0.141)
        self.TabStrip1.add(self.TabStrip1__Tab3, text='高级功能')

        self.P0_Check_antenna3Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna3.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna3 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna3Var, style='TP0_Check_antenna3.TCheckbutton')
        self.P0_Check_antenna3.setValue = lambda x: self.P0_Check_antenna3Var.set(x)
        self.P0_Check_antenna3.value = lambda : self.P0_Check_antenna3Var.get()
        self.P0_Check_antenna3.place(relx=0.609, rely=0.198, relwidth=0.082, relheight=0.252)

        self.P0_Check_antenna2Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna2.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna2 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna2Var, style='TP0_Check_antenna2.TCheckbutton')
        self.P0_Check_antenna2.setValue = lambda x: self.P0_Check_antenna2Var.set(x)
        self.P0_Check_antenna2.value = lambda : self.P0_Check_antenna2Var.get()
        self.P0_Check_antenna2.place(relx=0.514, rely=0.198, relwidth=0.082, relheight=0.252)

        self.P0_Check_antenna1Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna1.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna1 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna1Var, style='TP0_Check_antenna1.TCheckbutton')
        self.P0_Check_antenna1.setValue = lambda x: self.P0_Check_antenna1Var.set(x)
        self.P0_Check_antenna1.value = lambda : self.P0_Check_antenna1Var.get()
        self.P0_Check_antenna1.place(relx=0.42, rely=0.198, relwidth=0.082, relheight=0.252)

        self.P0_Combo_antennaList = ['1 ','2 ','3 ','4 ','5 ','6 ',]
        self.P0_Combo_antennaVar = StringVar(value='1 ')
        self.P0_Combo_antenna = Combobox(self.Frame4, text='1 ', textvariable=self.P0_Combo_antennaVar, values=self.P0_Combo_antennaList, font=('微软雅黑',9))
        self.P0_Combo_antenna.setText = lambda x: self.P0_Combo_antennaVar.set(x)
        self.P0_Combo_antenna.text = lambda : self.P0_Combo_antennaVar.get()
        self.P0_Combo_antenna.place(relx=0.42, rely=0.577, relwidth=0.134)

        self.Label5Var = StringVar(value='选择读写头：')
        self.style.configure('TLabel5.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label5 = Label(self.Frame4, text='选择读写头：', textvariable=self.Label5Var, style='TLabel5.TLabel')
        self.Label5.setText = lambda x: self.Label5Var.set(x)
        self.Label5.text = lambda : self.Label5Var.get()
        self.Label5.place(relx=0.105, rely=0.649, relwidth=0.266, relheight=0.225)

        self.Label4Var = StringVar(value='天线轮询设置')
        self.style.configure('TLabel4.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label4 = Label(self.Frame4, text='天线轮询设置', textvariable=self.Label4Var, style='TLabel4.TLabel')
        self.Label4.setText = lambda x: self.Label4Var.set(x)
        self.Label4.text = lambda : self.Label4Var.get()
        self.Label4.place(relx=0.105, rely=0.252, relwidth=0.266, relheight=0.225)

        self.P0_Check_antenna6Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna6.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna6 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna6Var, style='TP0_Check_antenna6.TCheckbutton')
        self.P0_Check_antenna6.setValue = lambda x: self.P0_Check_antenna6Var.set(x)
        self.P0_Check_antenna6.value = lambda : self.P0_Check_antenna6Var.get()
        self.P0_Check_antenna6.place(relx=0.892, rely=0.198, relwidth=0.082, relheight=0.252)

        self.P0_Command_FindVar = StringVar(value='搜索读写器')
        self.style.configure('TP0_Command_Find.TButton', font=('微软雅黑',9))
        self.P0_Command_Find = Button(self.Frame1, text='搜索读写器', textvariable=self.P0_Command_FindVar, command=lambda:self.asynchronous(self.P0_Command_Find_Cmd), style='TP0_Command_Find.TButton')
        # self.P0_Command_Find = Button(self.Frame1, text='搜索读写器', textvariable=self.P0_Command_FindVar, command=self.P0_Command_Find_Cmd, style='TP0_Command_Find.TButton')
        self.P0_Command_Find.setText = lambda x: self.P0_Command_FindVar.set(x)
        self.P0_Command_Find.text = lambda : self.P0_Command_FindVar.get()
        self.P0_Command_Find.place(relx=0.04, rely=0.566, relwidth=0.51, relheight=0.363)

        self.P0_Command1Var = StringVar(value='打开读写器')
        self.style.configure('TP0_Command1.TButton', font=('微软雅黑',9))
        self.P0_Command1 = Button(self.Frame1, text='打开读写器', textvariable=self.P0_Command1Var, command=self.P0_Command1_Cmd, style='TP0_Command1.TButton')
        self.P0_Command1.setText = lambda x: self.P0_Command1Var.set(x)
        self.P0_Command1.text = lambda : self.P0_Command1Var.get()
        self.P0_Command1.place(relx=0.677, rely=0.142, relwidth=0.205, relheight=0.363)

        self.P0_Command2Var = StringVar(value='关闭读写器')
        self.style.configure('TP0_Command2.TButton', font=('微软雅黑',9))
        self.P0_Command2 = Button(self.Frame1, text='关闭读写器', textvariable=self.P0_Command2Var, command=self.P0_Command2_Cmd, style='TP0_Command2.TButton')
        self.P0_Command2.setText = lambda x: self.P0_Command2Var.set(x)
        self.P0_Command2.text = lambda : self.P0_Command2Var.get()
        self.P0_Command2.place(relx=0.677, rely=0.566, relwidth=0.205, relheight=0.363)

        self.P0_Combo_portList = ['COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',]
        self.P0_Combo_portVar = StringVar(value='COM1')
        self.P0_Combo_port = Combobox(self.Frame1, text='COM1', textvariable=self.P0_Combo_portVar, values=self.P0_Combo_portList, font=('微软雅黑',9))
        self.P0_Combo_port.setText = lambda x: self.P0_Combo_portVar.set(x)
        self.P0_Combo_port.text = lambda : self.P0_Combo_portVar.get()
        self.P0_Combo_port.place(relx=0.406, rely=0.212, relwidth=0.144)

        self.P0_Combo_regionList = ['CQ','GZ',]
        self.P0_Combo_regionVar = StringVar(value='CQ')
        self.P0_Combo_region = Combobox(self.Frame1, text='CQ', textvariable=self.P0_Combo_regionVar, values=self.P0_Combo_regionList, font=('微软雅黑',9))
        self.P0_Combo_region.setText = lambda x: self.P0_Combo_regionVar.set(x)
        self.P0_Combo_region.text = lambda : self.P0_Combo_regionVar.get()
        self.P0_Combo_region.place(relx=0.152, rely=0.212, relwidth=0.11)

        self.Label2Var = StringVar(value='串口：')
        self.style.configure('TLabel2.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label2 = Label(self.Frame1, text='串口：', textvariable=self.Label2Var, style='TLabel2.TLabel')
        self.Label2.setText = lambda x: self.Label2Var.set(x)
        self.Label2.text = lambda : self.Label2Var.get()
        self.Label2.place(relx=0.321, rely=0.212, relwidth=0.076, relheight=0.248)

        self.Label3Var = StringVar(value='地域：')
        self.style.configure('TLabel3.TLabel', anchor='w', font=('微软雅黑',9))
        self.Label3 = Label(self.Frame1, text='地域：', textvariable=self.Label3Var, style='TLabel3.TLabel')
        self.Label3.setText = lambda x: self.Label3Var.set(x)
        self.Label3.text = lambda : self.Label3Var.get()
        self.Label3.place(relx=0.051, rely=0.212, relwidth=0.076, relheight=0.248)

        self.P0_Check_antenna5Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna5.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna5 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna5Var, style='TP0_Check_antenna5.TCheckbutton')
        self.P0_Check_antenna5.setValue = lambda x: self.P0_Check_antenna5Var.set(x)
        self.P0_Check_antenna5.value = lambda : self.P0_Check_antenna5Var.get()
        self.P0_Check_antenna5.place(relx=0.797, rely=0.198, relwidth=0.082, relheight=0.252)

        self.P0_Text_MsgVar = StringVar(value='')
        # self.P0_Text_Msg = Text(self.Frame3, textvariable=self.P0_Text_MsgVar, font=('微软雅黑',9))
        self.P0_Text_Msg = Text(self.Frame3, font=('微软雅黑',9))
        self.P0_Text_Msg.setText = lambda x: self.P0_Text_MsgVar.set(x)
        self.P0_Text_Msg.text = lambda : self.P0_Text_MsgVar.get()
        self.P0_Text_Msg.place(relx=0., rely=0.03, relwidth=1., relheight=0.97)

        self.P0_Check_antenna4Var = IntVar(value=0)
        self.style.configure('TP0_Check_antenna4.TCheckbutton', font=('宋体',9))
        self.P0_Check_antenna4 = Checkbutton(self.Frame4, variable=self.P0_Check_antenna4Var, style='TP0_Check_antenna4.TCheckbutton')
        self.P0_Check_antenna4.setValue = lambda x: self.P0_Check_antenna4Var.set(x)
        self.P0_Check_antenna4.value = lambda : self.P0_Check_antenna4Var.get()
        self.P0_Check_antenna4.place(relx=0.703, rely=0.198, relwidth=0.082, relheight=0.252)


class Application(Application_ui):
    #这个类实现具体的事件处理回调函数。界面生成代码在Application_ui中。
    def __init__(self, master=None):
        Application_ui.__init__(self, master)

    def TabStrip1_NotebookTabChanged(self, event):
        #TODO, Please finish the function here!
        pass

    def P1_Command_Diode_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            if self.P1_Text_Diode1.get().isdigit() and self.P1_Text_Diode2.get().isdigit() and self.P1_Text_Diode3.get().isdigit():
                R = int(self.P1_Text_Diode1.get())
                G = int(self.P1_Text_Diode2.get())
                Y = int(self.P1_Text_Diode3.get())
                leddisplaystat = str(self.vfjreaderexample.leddisplay(self.Handle, R, G, Y))
                self.P0_Text_Msg.insert("end", "leddisplay:" + leddisplaystat + "\n")
            else:
                  self.showmessage("设置leddisplay失败!R:%s,G:%s,Y:%s," %(self.P1_Text_Diode1.get(), self.P1_Text_Diode2.get(), self.P1_Text_Diode3.get()))
        else:
             self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P1_Command_Buzzer_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            if self.P1_Text_Buzzer1.get().isdigit() and self.P1_Text_Buzzer2.get().isdigit():
                num = int(self.P1_Text_Buzzer1.get())
                yd = int(self.P1_Text_Buzzer2.get())
                audiocontrolstat = str(self.vfjreaderexample.audiocontrol(self.Handle, num, yd))
                self.P0_Text_Msg.insert("end","audiocontrol:" + audiocontrolstat + "\n")
            else:
                self.showmessage("设置audiocontrol失败!Num:%s,YD:%s" %(self.P1_Text_Buzzer1.get(),self.P1_Text_Buzzer2.get()))
        else:
             self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P1_Command_ReaderVer_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            if self.Handle > 0:
                hd = self.Handle
            else:
                hd = 1 
            rdv = create_string_buffer(10 * 3)
            apiv = create_string_buffer(10 * 3)
            readerversionstat = self.vfjreaderexample.readerversion(hd, rdv, len(rdv), apiv, len(apiv))
            self.P0_Text_Msg.insert("end","reader_readerversion:" + str(readerversionstat) + 
                                 "\nversion:" + rdv.value.decode('utf-8') + 
                                 " \napp:"+apiv.value.decode('utf-8')+"\n")
        else:
            self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P1_Command_DllVersion_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            version = create_string_buffer(10 * 3)
            dllversion = self.vfjreaderexample.getversion(version, len(version))
            self.P0_Text_Msg.insert("end","stat:" + str(dllversion) +
                                 "\ndllversion:" + version.value.decode('utf-8')+"\n")
        else:
            self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def cpu_cmd(self, command):
        reply = create_string_buffer(256)
        lenrep = ctypes.c_int()
        stat = self.vfjreaderexample.cpucommandself(self.Handle, command, len(command), reply, lenrep)
        return {'stat':stat, 'reply':reply}


    def P2_Command_3F00_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            command = b"00A40000023F00"
            cmd_stat = self.cpu_cmd(command)
            if cmd_stat["stat"] == 0:
                self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(command, cmd_stat["stat"]) + cmd_stat["reply"].value.decode('utf-8')+"\n")
            else:
                self.P0_Text_Msg.insert("end","执行失败：%s"% (cmd_stat["stat"])+"\n")
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_1001_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            command = b"00A40000021001"
            cmd_stat = self.cpu_cmd(command)
            if cmd_stat["stat"] == 0:
                self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(command, cmd_stat["stat"]) + cmd_stat["reply"].value.decode('utf-8')+"\n")
            else:
                self.P0_Text_Msg.insert("end","执行失败：%s"% (cmd_stat["stat"])+"\n")
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_1005_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A40000023F00", b"00A40000021001", b"00B0950000"]
            for command in commands:
                cmd_stat = self.cpu_cmd(command)
                if cmd_stat["stat"] == 0:
                    self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(command, cmd_stat["stat"]) + cmd_stat["reply"].value.decode('utf-8')+"\n")
                else:
                    self.P0_Text_Msg.insert("end","执行失败：%s"% (cmd_stat["stat"])+"\n")
                    break
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_1006_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A40000023F00",  b"00B0960000"]
            for command in commands:
                cmd_stat = self.cpu_cmd(command)
                if cmd_stat["stat"] == 0:
                    self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(command, cmd_stat["stat"]) + cmd_stat["reply"].value.decode('utf-8')+"\n")
                else:
                    self.P0_Text_Msg.insert("end","执行失败：%s"% (cmd_stat["stat"])+"\n")
                    break
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")
           
           
    def cpu_apdu_list(self, commands):
        for command in commands:
            cmd_stat = self.cpu_cmd(command)
            if cmd_stat["stat"] == 0:
                self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(command, cmd_stat["stat"]) + cmd_stat["reply"].value.decode('utf-8')+"\n")
            else:
                self.P0_Text_Msg.insert("end","执行失败：%s"% (cmd_stat["stat"])+"\n")
                break


    def P2_Command_DF01_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A4000002DF01"]
            self.cpu_apdu_list(commands)
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_EF01_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A40000023F00",b"00A4000002EF01", b"00B0000000" ]
            self.cpu_apdu_list(commands)
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_EF02_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A40000023F00",b"00A4000002EF02", b"00B0000000" ]
            self.cpu_apdu_list(commands)
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_EF04_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = [b"00A4000002DF01",b"00A4000002EF04", b"00B0000080" ]
            self.cpu_apdu_list(commands)
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_OpenCard_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            cardplace = ctypes.c_int()
            cardsn = ctypes.create_string_buffer(8)
            opencardstatus = self.vfjreaderexample.opencard(self.Handle, cardplace, cardsn)
            if opencardstatus == -1:
                self.P0_Text_Msg.insert("end","没有卡，如果是同一卡片再打开，请反转一下卡片或执行复位天线再打开\n")
            elif opencardstatus == -2 or opencardstatus == -2000:
                self.P0_Text_Msg.insert("end","打开卡失败：%d\n" %(opencardstatus))
            else:  
                self.P0_Text_Msg.insert("end","打开卡片：%d ,SNR:%s\n" %(opencardstatus, cardsn.value.decode('utf-8')))
        else:
            self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_CloseCard_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            closecardststus = self.vfjreaderexample.closecard(self.Handle)
            self.P0_Text_Msg.insert("end","关闭卡片：%d \n" %(closecardststus))
        else:
            self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P2_Command_RFreset_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            resetrfststus = self.vfjreaderexample.resetRF(self.Handle)
            self.P0_Text_Msg.insert("end","复位天线：%d \n" %(resetrfststus))
        else:
            self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P3_Command_RFpsam_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            resamststus = self.vfjreaderexample.samreset(self.Handle, int(self.P3_Combo_PSAMposition.get()), 0)
            self.P0_Text_Msg.insert("end","复位卡槽%dpsam：%d \n" %(int(self.P3_Combo_PSAMposition.get()), resamststus))
        else:
             self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P3_Command_PSAMcommand_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            commands = self.Text7.get().encode('utf-8')
            reply = create_string_buffer(256)
            lenrep = ctypes.c_int()
            stat = self.vfjreaderexample.samcommand(self.Handle, int(self.P3_Combo_PSAMposition.get()), commands, len(commands), reply, lenrep)
            if stat == 0:
                self.P0_Text_Msg.insert("end","执行指令:%s\n执行结果：%d,返回数据："%(commands, stat) + reply.value.decode('utf-8')+"\n")
            else:
                self.P0_Text_Msg.insert("end","执行失败：%s"% (stat)+"\n")
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P3_Command_CPUcommand_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            command = self.Text7.get().encode('utf-8')
            commands = [command]
            self.cpu_apdu_list(commands)
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P3_Command_StopTest_Cmd(self, event=None):
        #TODO, Please finish the function here!
        if 'vfjreaderexample' in dir(self):
            if self.i != 0:
                self.lock = True
                self.P0_Text_Msg.insert("end","总测试数：%d,成功：%d,失败：%d\n" %(self.i, self.success, self.failure))
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")


        


    def P3_Command_StartTest_Cmd(self, event=None):
        #TODO, Please finish the function here!
        if 'vfjreaderexample' in dir(self):
            if self.P3_Text_TestTimes.get().isdigit() and self.P3_Text_TestTotal.get().isdigit() and self.P3_Text_Tested.get().isdigit() and self.P3_Text_ReadTimer.get().isdigit():
                n = int(self.P3_Text_TestTimes.get())
                total =  int(self.P3_Text_TestTotal.get())
                rt = int(self.P3_Text_ReadTimer.get())
                self.P3_Text_Tested.setText("0")
                self.lock = False
                cardplace = ctypes.c_int()
                cardsn = ctypes.create_string_buffer(8)
                command = b'00A4000002DF01'
                self.i = 0
                self.success = 0
                self.failure = 0
                self.P0_Text_Msg.insert("end","总测试次数:%d,尝试次数：%d,间隔时间：%d\n" %(total,n,rt))
                # print("总测试次数:%d,尝试次数：%d,间隔时间：%d" %(total,n,rt))
                while True:
                    a = 0
                    while True:
                        opencardstatus = self.vfjreaderexample.opencard(self.Handle, cardplace, cardsn)
                        a += 1
                        if opencardstatus > 0:
                            self.P0_Text_Msg.insert("end","开卡成功：%s\n" %(cardsn.value.decode('utf-8')))
                            stat = self.cpu_cmd(command)
                            if stat['stat'] == 0:
                                self.success += 1
                                self.i += 1
                                self.vfjreaderexample.closecard(self.Handle)
                                # print(self.success)
                                break  #执行指令成功后退出
                                print("跳出去了吗")                                
                        else:
                        # self.P0_Text_Msg.insert("end","开卡失败：%d\n" %(opencardstatus))

                        #     print("i:%d"%(self.i))                 
                        # print("a:%d,n:%d" %(a,n))
                            if a == n or self.lock:    
                                self.failure += 1
                                self.i += 1
                                self.P0_Text_Msg.insert("end","开卡失败%d次\n" %(a))
                                break   #达到尝试次数后退出
                        time.sleep(rt)
                    if self.i == total or self.lock:
                         break 
                self.P0_Text_Msg.insert("end","总测试数：%d,成功：%d,失败：%d\n" %(self.i, self.success, self.failure))
                
            # P3_Text_Tested
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开"+"\n")

    def P0_Command_Find_Cmd(self, event=None):
        try:
            os_type = platform.system()
            rdv = create_string_buffer(10 * 3)
            apiv = create_string_buffer(10 * 3)
            readerexample = loadlibrary(os_type, self.P0_Combo_region.get())
            self.P0_Text_Msg.insert("end","开始扫描读写器,请不要关闭程序！！\n")
            for i in self.P0_Combo_portList:
                stat = openreader_test(readerexample, i)
                if stat > 0 :              
                    readerversion_test(readerexample, stat, rdv, len(rdv), apiv, len(apiv))
                    self.P0_Text_Msg.insert("end","打开%s读写器成功,version:" %(i) + rdv.value.decode('utf-8') + "app:"+apiv.value.decode('utf-8')+"\n")
                    closereader_test(readerexample,stat)
                else:
                    self.P0_Text_Msg.insert("end","打开%s读写器失败:\n"%(i))
            self.P0_Text_Msg.insert("end","完成扫描\n")
        finally:
            if re.search('linux', platform.platform()):
                # 如果是Linux系统，使用ctypes.CDLL()._handle释放动态库
                ctypes.CDLL('libc.so.6').free(readerexample._handle)
                print("linux释放动态库")
            elif re.search('Win', platform.platform()):
                # 如果是Windows系统，使用ctypes.windll.kernel32.FreeLibrary(handle)释放动态库
                ctypes.windll.kernel32.FreeLibrary(readerexample._handle)
                print("window释放动态库")
            else:
                print("Unsupported system")
                
    # def P0_Command_Find_Cmd(self, event=None):
    #     thread = threading.Thread(target=self.P0_Command_Find_Cmd_1)
    #     thread.start()
    
    def asynchronous(self, fc):
        thread = threading.Thread(target=fc)
        thread.start()
    

    def P0_Command1_Cmd(self, event=None):
        os_type = platform.system()
        try:
            self.vfjreaderexample = VFJReader(os_type, self.P0_Combo_region.get(), bytes(self.P0_Combo_port.get(), 'utf-8'))
            openstat_int = self.vfjreaderexample.openreader()
            if openstat_int > 0:
                self.P0_Combo_region["state"]="disabled"
                self.P0_Combo_port["state"]="disabled"
                self.Handle = openstat_int
                self.P0_Text_Msg.insert("end", "打开读写器成功！句柄：%d\n" %(openstat_int))
            else:
                self.P0_Text_Msg.insert("end", "打开读写器失败！返回码：%d\n" %(openstat_int))
        except Exception as e:
            print('Error:', e)

    def P0_Command2_Cmd(self, event=None):
        if 'vfjreaderexample' in dir(self):
            if self.Handle > 0:
                closestat = self.vfjreaderexample.closereader(self.Handle)
                self.P0_Combo_region["state"]="normal"
                self.P0_Combo_port["state"]="normal"
                self.P0_Text_Msg.insert("end", "关闭读写器，返回码：%d\n" %(closestat))
            else:
                closestat = self.vfjreaderexample.closereader()
                if closestat == 0:
                    self.P0_Text_Msg.insert("end", "关闭COM1读写器成功：%d\n" %(closestat))
                else:
                    self.P0_Text_Msg.insert("end", "关闭读写器失败，返回码：%d\n" %(closestat))
        else:
           self.P0_Text_Msg.insert("end", "读写器未打开\n")

                
                
    def showmessage(self, messge):
        if sys.version_info[0] == 2:
            tkMessageBox.showinfo("Message",messge)
        else:
            messagebox.showinfo("Message",messge)
            
class MyApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if ap.Handle > 0 :
                ap.P0_Command2_Cmd()
                print("关闭读写器")
            else:
                ap.P0_Command2_Cmd()
            top.destroy()
             

if __name__ == "__main__":
    top = MyApp()
    ap = Application(top)
    ap.mainloop()



