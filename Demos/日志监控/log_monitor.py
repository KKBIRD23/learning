#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import re


class LogMonitoring(object):

    def __init__(self, file_name, pattern, threshold):
        self.__file_name = file_name
        self.__pattern = pattern
        self.__threshold = threshold
        self.__count = 0

    def source(self):
        with open(self.__file_name) as f:
            f.seek(0, 2)

            while True:
                data = f.readlines()
                # 可以添加日志量的监控，如果line列表越长说明单位时间产生的日志越多
                # print(len(line))

                for line in data:
                    if line != '\n':  # 空行不处理
                        self.check(line)
                time.sleep(1)

    # 匹配规则
    def check(self, data):
        self.source()
        if re.search(f'{self.__pattern}.*?', data):
            self.call()
            return 0
        else:
            return 1

    # 报警规则
    def call(self, ):
        self.__count += 1
        if self.__count > self.__threshold:
            print('***********')


if __name__ == '__main__':
    monitor = LogMonitoring("test.log", 1, 1)
