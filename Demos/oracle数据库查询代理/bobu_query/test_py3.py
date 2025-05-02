# -*- coding: utf-8 -*-
import json
import urllib.request
import urllib.error
import urllib.parse
import time
from typing import Dict, Optional, Any

# 配置常量
SERVER_URL = "http://10.50.15.68:8023"
API_KEY = "Vfj@1234.oracle"
MAX_RETRIES = 2  # 最大重试次数
RETRY_DELAY = 1  # 重试间隔（秒）

def query(
    sql: str,
    params: Optional[Dict] = None,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    执行 SQL 查询请求，返回统一格式的响应字典。

    Args:
        sql (str): 要执行的 SQL 语句。
        params (Optional[Dict]): 绑定参数的字典，默认为 None。
        timeout (float): 请求超时时间（秒），默认为 10 秒。

    Returns:
        Dict[str, Any]: 包含 success、data、error 键的响应字典。
    """
    # 参数校验
    if not isinstance(sql, str):
        return {
            "success": False,
            "data": None,
            "error": f"SQL 必须为字符串类型，实际类型为 {type(sql)}"
        }
    if params is not None and not isinstance(params, dict):
        return {
            "success": False,
            "data": None,
            "error": f"params 必须为字典类型，实际类型为 {type(params)}"
        }

    # 构建请求数据
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

    # 重试逻辑
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response = json.loads(resp.read().decode('utf-8'))
                return {
                    "success": True,
                    "data": response.get('data', []),
                    "error": None
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            error_msg = f"HTTP 错误 {e.code}: {error_body}"
            # 5xx 错误可重试
            if 500 <= e.code < 600 and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {"success": False, "data": None, "error": error_msg}
        except (urllib.error.URLError, TimeoutError) as e:
            error_msg = f"网络错误: {str(e)}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {"success": False, "data": None, "error": error_msg}
        except json.JSONDecodeError as e:
            return {"success": False, "data": None, "error": f"响应解析失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "data": None, "error": f"未知错误: {str(e)}"}

    return {"success": False, "data": None, "error": "超出最大重试次数"}

if __name__ == '__main__':
    # 示例调用
    response = query(
        "SELECT plateNo, orderId FROM orderinfo2 WHERE plateNo = :pat",
        {'pat': '渝ABM2079'}
    )

    # 统一处理响应
    if response["success"]:
        rows = response["data"]
        if len(rows) == 0:
            print("未查询到数据")
        else:
            print(f"查询到 {len(rows)} 条数据:")
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"请求失败: {response['error']}")
        exit(1)