# -*- coding: utf-8 -*-
"""
需求:
1、 创建检查类
    - 测试连接、取配置
2、 地址及密码管理类
    - 从文件取IP和密码
    - 把因为超连接次数的IP放到临时列表
    - 把放到临时列表的IP剩下的未尝试密码放到单独的列表
"""

import os
import paramiko

# import retrying


# 定义几个常量放文件名字和命令,方便以后修改
PW_FILE_NAME = "PW"
IP_FILE_NAME = "IP"
CMD = "cat /root/VFJ/AuthModelFrnt2/config/system.properties"


class Check(object):
    """配置检查"""

    # 鲁迅先生说过,能够with open就不要open,毕竟谁也说不准就忘记close——比如阿辉的代码,所以这里不选open
    def __init__(self):
        # 初始化方法把要操作的文件搞好
        self.ip_file_path = os.path.join(os.getcwd(), IP_FILE_NAME)
        self.pw_file_path = os.path.join(os.getcwd(), PW_FILE_NAME)
        self.er_file_path = os.path.join(os.getcwd(), "error_log.txt")
        self.ck_file_path = os.path.join(os.getcwd(), "check_log.txt")
        self.pw_dict_path = os.path.join(os.getcwd(), "password_dict.txt")
        self.ip = ""
        self.pw = ""

    # 定义实例方法,管理用户名密码
    def start_check(self):

        with open(self.ip_file_path, "r") as ip_file:
            ip_list = ip_file.readlines()
            for ip in ip_list:
                ip = ip.strip("\n")
                if ip == "\n":
                    print("谁他妈加的空行？坑爹呢！！！")
                else:
                    # 这里算下密码个数,以后用得到
                    with open(self.pw_file_path, "r") as pw_file:
                        count_pw = len(pw_file.readlines())
                    # 这里读密码开始搞了
                    with open(self.pw_file_path, "r") as pw_file:
                        pw_file = pw_file.readlines()
                        for pw in pw_file:
                            pw = pw.strip("\n")
                            if pw == "\n":
                                print("谁他妈加的空行？坑爹呢！！！")
                            else:
                                self.ip = ip
                                self.pw = pw

                                if self.check_server() == 1:
                                    break

                                if (count_pw := (count_pw - 1)) == 0:
                                    with open(self.er_file_path, "a") as error_file:
                                        print(f'{self.ip} 密码错完了,内心开始骂娘了……')
                                        error_file.writelines(f' {self.ip} 密码错完了,内心开始骂娘了……\n')
                                        error_file.writelines("-" * 50 + "\n")

    def check_server(self):
        try:
            # 实例化一个transport对象并测试通道
            print(f'开始连{self.ip}')
            transport = paramiko.Transport((self.ip, 22))

        except Exception as ex:
            print(ex)
            if str(ex) in 'Error reading SSH protocol banner[WinError 10054] 远程主机强迫关闭了一个现有的连接':
                print(f'{self.ip}::: 这个沙比拒绝我,打电话日决他！！！')
                with open(self.er_file_path, "a") as error_file:
                    error_file.writelines(f'{self.ip} 这个沙比拒绝我,打电话日决他！(人在屋檐下，顺便问密码...)\n')
                    error_file.writelines("-" * 50 + "\n")
                    return 1
            else:
                # 报错就是连不通,写到错误记录
                print(f'{self.ip} 的连接不通,请核对省门架表变更或通知机电')
                with open(self.er_file_path, "a") as error_file:
                    error_file.writelines(f'{self.ip} 的连接不通,请核对省门架表变更或通知机电\n')
                    error_file.writelines("-" * 50 + "\n")
                    return 1

        else:
            try:
                # 创建连接
                transport.connect(username="root", password=self.pw)
                print(f'开始检查: {self.ip} 的配置:')
                # 创建SSH对象
                ssh = paramiko.SSHClient()
                # 将ssh对象的私有_transport指定为以上的transport
                ssh._transport = transport
                # 打开一个Channel并执行命令
                stdin, stdout, stderr = ssh.exec_command(CMD)
                # 获取命令结果
                res, err = stdout.read().decode(encoding="utf-8-sig"), stderr.read().decode(encoding="utf-8-sig")
                result = res if res else err
                print(result)

                # 看看服务器是不是傻逼偷偷被重装了
                if "No such file or directory" in result:
                    print("妈的又偷偷重装老子的机！\n")
                    with open(self.er_file_path, "a") as error_file:
                        error_file.writelines(f'妈的又偷偷重装老子的机！ {self.ip} 要重新部署！\n')
                        error_file.writelines("-" * 50 + "\n")
                        return 1

                # 啥事都没有的话,就默默把配置拉到日志中
                with open(self.ck_file_path, "a") as check_file:
                    check_file.writelines(f'开始检查 {self.ip} 的配置\n')
                    check_file.writelines(f'{result}\n')
                    check_file.writelines("=" * 80 + "\n\n")
                with open(self.pw_dict_path, "a") as pw_dict:
                    pw_dict.writelines(f'{self.ip}::::{self.pw}\n')
                    return 1

            except Exception as ex:
                if str(ex) == 'Authentication failed.':
                    print(f'{self.ip}:::{self.pw} 密码一个不对再换一个,怼死为止~~')

                else:
                    with open(self.er_file_path, "a") as error_file:
                        print(ex)
                        print("我也不知道这是什么错误,写一笔再说……")
                        error_file.writelines(f'我也不知道这是什么错误,写一笔再说……\n{ex}\n')
                        error_file.writelines("-" * 50 + "\n")
            finally:
                transport.close()


if __name__ == '__main__':
    check = Check()
    check.start_check()
