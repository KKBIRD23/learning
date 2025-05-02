# -*- coding: utf-8 -*-
import json
import urllib
import urllib2
import time

# 配置常量
SERVER_URL = "http://127.0.0.1:8023"
API_KEY = "Vfj@1234.oracle"
MAX_RETRIES = 3
RETRY_DELAY = 1

def query(sql, params=None, timeout=10.0):
    # 确保所有输入为 UTF-8 编码的 str 类型（Python 2 的终极解决方案）
    try:
        # 1. 预处理 SQL 语句
        if isinstance(sql, unicode):
            sql = sql.encode('utf-8')

        # 2. 预处理参数（关键修复）
        encoded_params = {}
        if params:
            for key, value in params.items():
                # 键必须是 str 类型
                if isinstance(key, unicode):
                    key = key.encode('utf-8')
                # 值可以是任意类型，但如果是字符串必须转为 UTF-8 str
                if isinstance(value, unicode):
                    value = value.encode('utf-8')
                encoded_params[key] = value

        # 3. 生成 JSON 时直接处理为 UTF-8 bytes
        # 使用 json 模块生成 unicode 字符串，再显式编码为 UTF-8
        params_json = json.dumps(encoded_params or {}, ensure_ascii=False)
        if isinstance(params_json, unicode):
            params_json = params_json.encode('utf-8')

        data = {
            'sql': sql,
            'params': params_json  # 已经是 UTF-8 bytes
        }

        # 4. 构建请求（无需再调用 urlencode，直接发送原始 JSON）
        # 因为服务端要求的是 application/x-www-form-urlencoded
        # 特殊处理：手动构建键值对
        encoded_data = urllib.urlencode({
            'sql': sql,
            'params': params_json
        })

        headers = {
            'X-Api-Key': API_KEY.encode('utf-8')
        }
        req = urllib2.Request(SERVER_URL, encoded_data, headers)

        # 5. 重试逻辑（保持不变）
        for attempt in range(MAX_RETRIES):
            try:
                response = urllib2.urlopen(req, timeout=timeout)
                response_data = response.read()
                result = json.loads(response_data)
                return {
                    "success": True,
                    "data": result.get('data', []),
                    "error": None
                }
            except urllib2.HTTPError as e:
                error_body = e.read()
                error_msg = "HTTP 错误 {}: {}".format(e.code, error_body)
                if 500 <= e.code < 600 and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return {"success": False, "data": None, "error": error_msg}
            except (urllib2.URLError, IOError) as e:
                error_msg = "网络错误: {}".format(str(e.reason))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return {"success": False, "data": None, "error": error_msg}
            except ValueError as e:
                return {"success": False, "data": None, "error": "响应解析失败: {}".format(str(e))}
            except Exception as e:
                return {"success": False, "data": None, "error": "未知错误: {}".format(str(e))}

        return {"success": False, "data": None, "error": "超出最大重试次数"}

    except Exception as e:
        return {"success": False, "data": None, "error": "客户端错误: {}".format(str(e))}

if __name__ == '__main__':
    response = query(
        u"SELECT plateNo, orderId FROM orderinfo2 WHERE plateNo = :pat",
        {'pat': u'渝ABM2079'}
    )

    if response["success"]:
        rows = response["data"]
        if len(rows) == 0:
            print u"未查询到数据".encode('utf-8')
        else:
            print u"查询到 {} 条数据:".format(len(rows)).encode('utf-8')
            print json.dumps(rows, ensure_ascii=False, indent=2)
    else:
        error_msg = response["error"]
        if isinstance(error_msg, unicode):
            error_msg = error_msg.encode('utf-8')
        print "请求失败: {}".format(error_msg)
        exit(1)
