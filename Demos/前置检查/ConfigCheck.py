# -*- coding: utf-8 -*-
import os
import sys
import paramiko

DIRNAME = os.path.dirname(os.path.realpath(sys.argv[0]))
ip_file = os.path.join(DIRNAME, "IP")
pw_file = os.path.join(DIRNAME, "PW")
log_file = os.path.join(DIRNAME, "CHECKLOG.txt")
error_file = os.path.join(DIRNAME, "ERRORLOG.txt")

IP_file = open(ip_file, "r")
CHECKLOG = open(log_file, "w")
ERRORLOG = open(error_file, "w")

count_pw = len(open(pw_file).readlines())

for IP in IP_file.readlines():
    ip = IP.replace("\n", "").replace("\r", "")
    PW_file = open(pw_file, "r")
    pw_list = []

    for PW in PW_file.readlines():
        pw = PW.replace("\n", "").replace("\r", "")
        pw_list.append(pw)
        print(ip, pw)
        try:
            transport = paramiko.Transport((ip, 22))
        except Exception as e:
            print("连接不通")
            ERRORLOG.writelines(ip + ":连接不通\n")
            pw_list.remove(pw)  # TODO 剩余的密码要和该IP有关联避免无限重试密码
            break
        try:
            # print(pw)
            transport.connect(username="root", password=pw)
            print("开始检查:" + ip + "的配置:")
            # CHECKLOG.writelines("开始检查:")
            # CHECKLOG.writelines(ip + pw)
            # CHECKLOG.writelines("的配置:\n")
            ssh = paramiko.SSHClient()
            ssh._transport = transport
            # 执行命令，不要执行top之类的在不停的刷新的命令
            stdin, stdout, stderr = ssh.exec_command("cat /root/VFJ/AuthModelFrnt2/config/system.properties")
            # 获取命令结果
            res, err = stdout.read(), stderr.read()
            result = res if res else err
            CHECKLOG.writelines(f'开始检查 {ip} 的配置,密码为: {pw}\n')
            CHECKLOG.write(result.decode())
            CHECKLOG.write("\n--------------------------------------------------------------\n")
            break
        # except paramiko.ssh_exception.AuthenticationException as e:
        except Exception as e:
            print("密码不正确")
            if (count_pw := (count_pw - 1)) == 0:
                ERRORLOG.writelines(f'{ip}'":密码不能在密码表中找到\n")

CHECKLOG.close()
IP_file.close()
ERRORLOG.close()
