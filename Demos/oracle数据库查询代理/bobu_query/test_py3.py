# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import urllib.parse
import json

API_KEY = "Vfj@1234.oracle"
#SERVER_URL = "http://127.0.0.1:8023"
SERVER_URL = "http://10.50.15.68:8023"

def query(sql, params=None):
    data = {
        'sql': sql,
        'params': json.dumps(params or {}, ensure_ascii=False)
    }
    encoded_data = urllib.parse.urlencode(data).encode('utf-8')
    req = urllib.request.Request(
        SERVER_URL,
        data=encoded_data,
        headers={'X-Api-Key': API_KEY}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"错误 {e.code}: {error_body}")
        return None

if __name__ == '__main__':
    result = query(
        "SELECT plateNo, orderId FROM orderinfo2 WHERE plateNo LIKE :pat",
        {'pat': '%ABM2079%'}
    )
    print(f"查询结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
