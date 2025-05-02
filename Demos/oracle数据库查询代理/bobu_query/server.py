# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import SocketServer
import urlparse
import json
import cx_Oracle
import os
import logging
import time
import glob
import datetime
import fcntl  # 新增：用于单例锁
from logging.handlers import TimedRotatingFileHandler
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

# ---------------------------
# 配置区（需与config.json匹配）
# ---------------------------
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LISTEN_PORT = 8023  # 确保与客户端配置一致

# ---------------------------
# 初始化配置和日志
# ---------------------------
def load_config():
    """加载配置文件"""
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            return (
                config['db_user'],
                config['db_password'],
                config['db_dsn'],
                config['api_key']
            )
    except Exception as e:
        raise RuntimeError("加载配置文件失败: " + str(e))

try:
    DB_USER, DB_PASSWORD, DB_DSN, API_KEY = load_config()
except Exception as e:
    print("致命错误: " + str(e))
    exit(1)

# 初始化Oracle连接池（关键编码配置）
pool = cx_Oracle.SessionPool(
    DB_USER,
    DB_PASSWORD,
    DB_DSN,
    encoding="GBK",        # 数据库字符集为ZHS16GBK
    nencoding="UTF-8",     # 返回数据使用UTF-8
    min=2,
    max=5,
    increment=1,
    threaded=True
)

# 日志配置（含控制台输出）
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # 文件日志（按天轮转）
    file_handler = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, 'service.log'),
        when='midnight',
        backupCount=180,
        encoding='utf-8'
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # 控制台日志（简洁格式）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def clean_old_logs():
    """清理超过180天的日志文件"""
    now = time.time()
    for log_path in glob.glob(os.path.join(LOG_DIR, '*.log')):
        if os.stat(log_path).st_mtime < now - 180 * 86400:
            try:
                os.remove(log_path)
                logger.info(u"已清理旧日志文件: %s", log_path)
            except Exception as e:
                logger.warning(u"清理日志失败 %s: %s", log_path, str(e))

# ---------------------------
# HTTP请求处理器（核心修复点）
# ---------------------------
class OracleRequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, status, data):
        """发送UTF-8编码的JSON响应"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False))

    def log_message(self, format, *args):
        """重定向访问日志到logger"""
        logger.info("[Access] %s - %s", self.client_address[0], format % args)

    def do_POST(self):
        try:
            # 1. 认证检查
            if self.headers.get('X-Api-Key') != API_KEY:
                logger.warning(u"非法访问尝试 from %s", self.client_address[0])
                return self._send_response(401, {"error": "未授权"})
            
            # 2. 解析请求参数
            content_len = int(self.headers.get('content-length', 0))
            raw_data = self.rfile.read(content_len)
            params = urlparse.parse_qs(raw_data)
            
            if 'sql' not in params or not params['sql'][0]:
                return self._send_response(400, {"error": "缺少SQL参数"})
            sql = params['sql'][0].strip()
            bind_vars = json.loads(params.get('params', ['{}'])[0])
            
            # 3. 执行数据库查询
            conn = pool.acquire()
            cursor = conn.cursor()
            cursor.execute(sql, **bind_vars)
            
            # 4. 处理结果编码（关键修复）
            columns = [col[0] for col in cursor.description]
            rows = []
            for row in cursor:
                row_dict = {}
                for idx, value in enumerate(row):
                    if isinstance(value, (str, cx_Oracle.LOB)):
                        # 将LOB类型转为字符串，并处理GBK到UTF-8的转换
                        str_value = str(value) if isinstance(value, cx_Oracle.LOB) else value
                        try:
                            decoded_value = str_value.decode('gbk')  # 从GBK解码
                        except UnicodeDecodeError:
                            decoded_value = str_value  # 容错处理
                        row_dict[columns[idx]] = decoded_value
                    else:
                        row_dict[columns[idx]] = value
                rows.append(row_dict)
            
            # 5. 记录成功日志
            logger.info(
                u"成功查询 from %s | SQL: %s | 参数: %s",
                self.client_address[0],
                sql,
                json.dumps(bind_vars, ensure_ascii=False)
            )
            
            self._send_response(200, {"data": rows})
            
        except cx_Oracle.DatabaseError as e:
            error_msg = u"数据库错误: {}".format(str(e))
            logger.error(error_msg)
            self._send_response(500, {"error": error_msg})
        except ValueError as e:
            error_msg = u"参数解析错误: {}".format(str(e))
            logger.error(error_msg)
            self._send_response(400, {"error": error_msg})
        except Exception as e:
            error_msg = u"未捕获的异常: {}".format(str(e))
            logger.exception(error_msg)
            self._send_response(500, {"error": "服务器内部错误"})
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                pool.release(conn)

# ---------------------------
# 服务启动（含单例锁）
# ---------------------------
if __name__ == '__main__':
    # 单例锁防止重复启动
    lock_file = os.path.join(os.path.dirname(__file__), 'server.lock')
    try:
        lock = open(lock_file, 'w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.critical(u"服务已在运行中，禁止重复启动！")
        sys.exit(1)
    
    try:
        clean_old_logs()
        logger.info(u"=" * 50)
        logger.info(u"Oracle查询服务启动 监听端口: %d", LISTEN_PORT)
        server = HTTPServer(('0.0.0.0', LISTEN_PORT), OracleRequestHandler)
        server.serve_forever()
    except Exception as e:
        logger.critical(u"服务异常终止: %s", str(e))
        sys.exit(1)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        os.remove(lock_file)
