# -*- coding: utf-8 -*-
"""
需求:
1、 创建检查类
    - 测试连接、取配置
2、 地址及密码管理类
    - 从文件取IP和密码
    - 把因为超连接次数的IP放到临时列表
    - 把放到临时列表的IP剩下的未尝试密码放到单独的列表
——————————————————————————————————————————————————————
Check

"""

import os
import paramiko
import retrying


class Check(object):
    """配置检查"""

    # 定义类属性pw_ip_dict,接收用户名密码字典
    ip_file_path = os.path.join(os.getcwd(), "IP")
    pw_file_path = os.path.join(os.getcwd(), "PW")
    er_file_path = os.path.join(os.getcwd(), "error_log.txt")
    ck_file_path = os.path.join(os.getcwd(), "check_log.txt")
    count_pw = 0
    pw_ip_dict = {}

    # 定义类方法,生产用户名密码字典
    @classmethod
    def pw_ip_manage(cls):
        with open(cls.ip_file_path, "r") as ip_file:
            ip_list = ip_file.readlines()
            for ip in ip_list:
                ip = ip.strip("\n")
                if ip == "\n":
                    print("谁他妈加的空行？坑爹呢！！！")
                else:
                    with open(cls.pw_file_path, "r") as pw_file:
                        pw_file = pw_file.readlines()
                        global cls.count_pw
                        cls.count_pw = len(open(cls.pw_file_path).readlines())
                        for pw in pw_file:
                            pw = pw.strip("\n")
                            if pw == "\n":
                                print("谁他妈加的空行？坑爹呢！！！")
                            else:
                                cls.pw_ip_dict[ip] = pw

        # with open(cls.pw_file_path) as len_pw:
        #     count_pw = len(len_pw.readlines())

    def check_server(self):
        for ip in self.pw_ip_dict:
            try:
                # 实例化一个transport对象并测试通道
                transport = paramiko.Transport((ip, 22))
            except Exception as ex:
                print(f'{ip} 的连接不通,请核对省门架表变更或通知机电')
                # 报错就是连不通,写到错误记录
                with open(self.er_file_path) as error_file:
                    error_file.writelines(f'{ip} 的连接不通,请核对省门架表变更或通知机电\n')
                    error_file.writelines("-" * 50 + "\n")
                break

            try:
                # 创建连接
                transport.connect(username="root", password=self.pw_ip_dict[ip])
                print(f'开始检查:" + {ip} + "的配置:')
                # 创建SSH对象
                ssh = paramiko.SSHClient
                # 将sshclient的对象的私有_transport指定为以上的transport
                ssh._transport = transport
                # 打开一个Channel并执行命令
                stdin, stdout, stderr = ssh.exec_command("cat /root/VFJ/AuthModelFrnt2/config/system.properties")
                # 获取命令结果
                res, err = stdout.read(), stderr.read()
                result = res if res else err

                # 看看服务器是不是傻逼偷偷被重装了
                if "No such file or directory" in result:
                    print("妈的又偷偷重装老子的机！")
                    with open(self.er_file_path) as error_file:
                        error_file.writelines(f'妈的又偷偷重装老子的机！ {ip} 要重新部署！\n')
                        check_file.writelines("\n--------------------------------------------------------------\n")

                # 啥事都没有的话,就默默把配置拉到日志中
                with open(self.ck_file_path) as check_file:
                    check_file.writelines(f'开始检查 {ip} 的配置,密码为: {self.pw_ip_dict[ip]}\n')
                    check_file.writelines("-------------------------")
                    check_file.writelines(result)
                    check_file.writelines("\n--------------------------------------------------------------\n")

                transport.close()

            except Exception as ex:
                if str(ex) == 'Authentication failed.':
                    print("一个不对再换一个,怼死为止~~")
                    if (Check.count_pw := (self.cou - 1)) == 0:
                else:
                    print("BBB")