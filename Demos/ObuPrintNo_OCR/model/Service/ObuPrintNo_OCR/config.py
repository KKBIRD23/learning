# config.py (V18.1_Final_Adjudication)

import os

# --- 基础路径配置 ---
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# --- 模型路径 ---
ONNX_MODEL_PATH = os.path.join(BASE_PROJECT_DIR, "model", "yolo", "Barcode_dynamic-True_half-False.onnx")
OCR_ONNX_MODEL_PATH = os.path.join(BASE_PROJECT_DIR, "model", "ppocr-v5", "inference.onnx")
OCR_KEYS_PATH = os.path.join(BASE_PROJECT_DIR, "model", "ppocr-v5", "keys.txt")

# ================== 核心裁决引擎配置 (V18.1) ==================
# --- 预处理与净化 ---
ENABLE_OCR_CORRECTION = True
# 这个列表现在应该更保守，只包含高置信度的替换
OCR_HEURISTIC_REPLACEMENTS = {
    'S': '5',
    'B': '8',
    'I': '1',
    'O': '0',
    'D': '0',
    'Z': '2',
    'G': '6',
    'Q': '0'
}

# --- 可配置头部修正 ---
ENABLE_HEADER_CORRECTION = True
CORRECTION_HEADER_PREFIX = "5001" # 您指定的、唯一可以确信的头部规则

# --- 证据晋升与会话管理 ---
PROMOTION_THRESHOLD = 1      # OBU被确信所需的最低目击次数,当设置为1时相当于关闭确信模式直接信任
SESSION_CLEANUP_HOURS = 24   # 会话数据在内存中保留的小时数

# --- 动态批次验证器 ---
# a. “满溢纯净”规则
PURITY_CHECK_THRESHOLD = 50 # 触发“满溢纯净”规则所需的OBU数量

# b. “三点定位”与“汉明裁决”
# 汉明距离计算的开关
ENABLE_HAMMING_CHECK = True
# 汉明距离裁决的阈值，小于等于此值则通过
HAMMING_THRESHOLD = 1
# 形成一个有效号段所需的最少连号数
MIN_SEGMENT_MEMBERS = 3
# 定义号段“断裂点”的间距阈值，大于此值则认为属于不同号段
SEGMENT_GAP_THRESHOLD = 5
# “三点定位”法中，以中间值为中心，向两边扩展的范围
GUESS_RANGE = 48

# c. “混沌安全阀”
# 当识别出的独立号段数量超过此值，则判定为“混沌模式”，自动跳过汉明裁决
MAX_SEGMENTS_THRESHOLD = 5
# ============================================================

# ================== 数据库配置 ==============================
# 优先从环境变量读取，如果不存在，则使用后面的默认值。
# 这样既保证了生产环境的安全性，也方便了本地开发。
# ============本地测试使用后段，生产使用.env中配置的环境变量====================
DB_USERNAME = os.getenv("DB_USERNAME", "VFJ_CQGS")
DB_PASSWORD = os.getenv("DB_PASSWORD", "vfj_20231007")
DB_DSN = os.getenv("DB_DSN", "192.168.1.200:1521/ORCL")
DB_TABLE_NAME = "SINGCHIPOBU"
DB_COLUMN_NAME = "OBUSAMSERIALNO"

# --- 安全与同步配置 ---
REFRESH_API_KEY = os.getenv("REFRESH_API_KEY", "Vfj@1234.wq")
# ============================================================

# --- Flask 应用配置 ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# --- 日志配置 ---
LOG_LEVEL = "DEBUG" # 可选值: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_DIR = "log"
LOG_FILE = "app.log"
LOG_FILE_MAX_BYTES = 50 * 1024 * 1024   # 日志大小为50MB，超过就会新生成一个文件
LOG_FILE_BACKUP_COUNT = 5   # 保留5个备份日志文件，超过就会删除最旧的文件

# --- YOLOv8 配置 ---
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_MIN_DETECTION_AREA_PX = 2000
YOLO_MAX_DETECTION_AREA_FACTOR = 0.1
YOLO_COCO_CLASSES = ['Barcode']

# --- OCR 配置 ---
OCR_TARGET_INPUT_HEIGHT = 48
OCR_DIGIT_ROI_Y_OFFSET_FACTOR = -0.15
OCR_DIGIT_ROI_HEIGHT_FACTOR = 0.7
OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05
OCR_NUM_WORKERS = 7
SAVE_TRAINING_ROI_IMAGES = True

# --- 布局与状态管理配置 (当前主要用于调试图保存路径) ---
LAYOUT_EXPECTED_TOTAL_ROWS = 13
LAYOUT_REGULAR_ROWS_COUNT = 12
LAYOUT_REGULAR_COLS_COUNT = 4
LAYOUT_SPECIAL_ROW_COLS_COUNT = 2
LAYOUT_TOTAL_OBUS_EXPECTED = 50
LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD = 50
LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD = 400
LAYOUT_MIN_CORE_ANCHORS_FOR_STATS = 3
VALID_OBU_CODES = {} # 为兼容旧模块导入而保留

PROCESS_PHOTO_DIR = "process_photo"
SAVE_PROCESS_PHOTOS = True
PROCESS_PHOTO_JPG_QUALITY = 65

# --- 零散识别模式配置 ---
SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH = 600
SCATTERED_MODE_IMAGE_JPG_QUALITY = 75
SAVE_SCATTERED_ANNOTATED_IMAGE = True

# --- 版本号 ---
APP_VERSION = "v18.1_Final"