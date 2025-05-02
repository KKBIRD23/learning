# -*- coding: utf-8 -*-
import sys
import urllib
import urllib2
import json
import codecs  # 新增导入codecs模块

# 强制标准输出使用UTF-8
reload(sys)
sys.setdefaultencoding('utf-8')
# 替换原有的open调用为codecs包装
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)  # 修正行

API_KEY = "Vfj@1234.oracle" # 和服务器一致
# SERVER_URL = "http://10.50.15.68:8023"  # 确保端口正确
SERVER_URL = "http://127.0.0.1:8023" # 本地测试

def query(sql, params=None):
    data = {
        'sql': sql,
        'params': json.dumps(params or {})
    }
    req = urllib2.Request(
        SERVER_URL,
        data=urllib.urlencode(data),
        headers={'X-Api-Key': API_KEY}
    )
    try:
        resp = urllib2.urlopen(req)
        result = json.loads(resp.read().decode('utf-8'))  # 显式解码
        return result
    except urllib2.HTTPError as e:
        print(u"错误 {}: {}".format(e.code, e.read().decode('utf-8')))
        return None

if __name__ == '__main__':
    result = query(
        u"SELECT plateNo, orderId FROM orderinfo2 WHERE plateNo LIKE :pat",
        {'pat': u'%ABM2079%'}
    )
    print(u"查询结果: {}".format(json.dumps(result, ensure_ascii=False, indent=2)))
