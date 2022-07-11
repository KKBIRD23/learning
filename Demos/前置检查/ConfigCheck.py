#!/usr/bin/env python
# visit https://tool.lu/pyc/ for more information
# Version: Python 3.6

import os
import sys
import paramiko

DIRNAME = os.path.dirname(os.path.realpath(sys.argv[0]))
ipfile = os.path.join(DIRNAME, 'IP')
pwfile = os.path.join(DIRNAME, 'PW')
logfile = os.path.join(DIRNAME, 'CHECKLOG.txt')
IP_file = open(ipfile, 'r')
CHECKLOG = open(logfile, 'w')
for IP in IP_file.readlines():
    ip = IP.replace('\n', '').replace('\r', '')
    PW_file = open(pwfile, 'r')
    for PW in PW_file.readlines():
        pw = PW.replace('\n', '').replace('\r', '')
        print(ip, pw)

        try:
            transport = paramiko.Transport((ip, 22))
        except Exception:
            e = None

            try:
                print('\xe8\xbf\x9e\xe6\x8e\xa5\xe4\xb8\x8d\xe9\x80\x9a')
                CHECKLOG.writelines(ip + ':\xe8\xbf\x9e\xe6\x8e\xa5\xe4\xb8\x8d\xe9\x80\x9a\n')
            finally:
                e = None
                del e

        try:
            transport.connect('root', pw, **('username', 'password'))
            print('\xe5\xbc\x80\xe5\xa7\x8b\xe6\xa3\x80\xe6\x9f\xa5:' + ip + '\xe7\x9a\x84\xe9\x85\x8d\xe7\xbd\xae:')
            CHECKLOG.writelines('\xe5\xbc\x80\xe5\xa7\x8b\xe6\xa3\x80\xe6\x9f\xa5:')
            CHECKLOG.writelines(ip + pw)
            CHECKLOG.writelines('\xe7\x9a\x84\xe9\x85\x8d\xe7\xbd\xae:\n')
            ssh = paramiko.SSHClient()
            ssh._transport = transport
            (stdin, stdout, stderr) = ssh.exec_command('cat /root/VFJ/AuthModelFrnt2/config/system.properties')
            res = stdout.read()
            err = stderr.read()
            result = res if res else err
            CHECKLOG.write(result.decode())
            CHECKLOG.write('\n--------------------------------------------------------------')
        continue
        except Exception:
        e = None

        try:
            print('\xe5\xaf\x86\xe7\xa0\x81\xe4\xb8\x8d\xe6\xad\xa3\xe7\xa1\xae')
            CHECKLOG.writelines(ip)
            CHECKLOG.writelines(':\xe5\xaf\x86\xe7\xa0\x81\xe4\xb8\x8d\xe6\xad\xa3\xe7\xa1\xae:')
            CHECKLOG.writelines(pw + '\n')
        finally:
            e = None
            del e

        continue

CHECKLOG.close()
