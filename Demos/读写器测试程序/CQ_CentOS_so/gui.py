import tkinter as tk
import tkinter.font as tkFont


class App:
    def __init__(self, root):
        # setting title
        root.title("undefined")
        # setting window size
        width = 640
        height = 480
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GButton_285 = tk.Button(root)
        GButton_285["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_285["font"] = ft
        GButton_285["fg"] = "#000000"
        GButton_285["justify"] = "center"
        GButton_285["text"] = "检测"
        GButton_285.place(x=170, y=30, width=294, height=40)
        GButton_285["command"] = self.GButton_285_command

        GMessage_388 = tk.Message(root)
        ft = tkFont.Font(family='Times', size=10)
        GMessage_388["font"] = ft
        GMessage_388["fg"] = "#333333"
        GMessage_388["justify"] = "center"
        GMessage_388["text"] = "Message"
        GMessage_388.place(x=40, y=190, width=553, height=267)

        GLabel_749 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        GLabel_749["font"] = ft
        GLabel_749["fg"] = "#333333"
        GLabel_749["justify"] = "center"
        GLabel_749["text"] = "串口选择"
        GLabel_749.place(x=190, y=100, width=165, height=42)

        GMessage_564 = tk.Message(root)
        ft = tkFont.Font(family='Times', size=10)
        GMessage_564["font"] = ft
        GMessage_564["fg"] = "#333333"
        GMessage_564["justify"] = "center"
        GMessage_564["text"] = "Message"
        GMessage_564.place(x=370, y=100, width=47, height=41)

    def GButton_285_command(self):
        print("command")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
