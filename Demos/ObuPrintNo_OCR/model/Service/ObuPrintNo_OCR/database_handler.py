# database_handler.py (FINAL CORRECTED VERSION)

import oracledb
import config
from typing import Set, Optional
import time

class DatabaseHandler:
    """
    负责处理所有与数据库的交互，包括连接池管理和数据加载。
    """
    def __init__(self, logger: any):
        self.logger = logger
        self.pool = None
        self._create_pool()

    def _create_pool(self):
        """
        在服务启动时，创建一个高效的数据库连接池。
        """
        try:
            self.logger.info("正在创建Oracle数据库连接池...")
            self.pool = oracledb.create_pool(
                user=config.DB_USERNAME,
                password=config.DB_PASSWORD,
                dsn=config.DB_DSN,
                min=2,
                max=5,
                increment=1
            )
            # ==================================================================
            self.logger.info("Oracle数据库连接池创建成功。")
        except oracledb.Error as e:
            self.logger.critical(f"数据库连接池创建失败: {e}", exc_info=True)
            self.pool = None

# database_handler.py

    def load_valid_obus(self) -> Optional[Set[str]]:
        """
        从数据库中加载所有有效的OBU码。
        """
        if not self.pool:
            self.logger.error("数据库连接池不可用，无法加载OBU码。")
            return None

        self.logger.info(f"正在从数据库表 {config.DB_TABLE_NAME} 加载有效OBU码...")

        sql = f"SELECT {config.DB_COLUMN_NAME} FROM {config.DB_TABLE_NAME}"

        try:
            with self.pool.acquire() as connection:
                with connection.cursor() as cursor:
                    # --- 【核心性能优化】设置数组大小 ---
                    # 将其设置为一个很大的值（例如100000），让它尽可能一次性地、
                    # 或者分很少几次，就将所有数据从数据库服务器拉取到应用服务器。
                    # 这将极大地减少网络延迟带来的性能损耗。
                    cursor.arraysize = 1000000

                    cursor.execute(sql)

                    t_start_fetch = time.time()
                    # 使用最高效的方式，将所有行一次性提取到列表中，然后再构建集合
                    rows = cursor.fetchall()
                    valid_obus_set = {row[0] for row in rows}
                    fetch_time = time.time() - t_start_fetch

                    self.logger.info(f"成功从数据库加载 {len(valid_obus_set)} 个有效OBU码。数据获取与处理耗时: {fetch_time:.2f}s")
                    return valid_obus_set
        except oracledb.Error as e:
            self.logger.error(f"从数据库加载OBU码时发生错误: {e}", exc_info=True)
            return None

    def close_pool(self):
        """
        在应用退出时，安全地关闭连接池。
        """
        if self.pool:
            self.logger.info("正在关闭Oracle数据库连接池...")
            self.pool.close()
            self.logger.info("Oracle数据库连接池已关闭。")