# config.py

import os

# --- 基础路径配置 ---
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) # 假设app.py在Service目录下

# --- 模型路径 ---
ONNX_MODEL_PATH = os.path.join(BASE_PROJECT_DIR, "model", "model", "BarCode_Detect", "Barcode_dynamic-True_half-False.onnx")
OCR_ONNX_MODEL_PATH = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_onnx", "inference.onnx")
OCR_KEYS_PATH = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_onnx", "keys.txt")





# ==================启发式规则配置区==========================
# 步骤1：启发式替换规则。在进行匹配前，会将key替换为value。
# 是否启用OCR启发式纠错功能
ENABLE_OCR_CORRECTION = True

OCR_HEURISTIC_REPLACEMENTS = {
    'S': '5',
    'B': '8',
    'I': '1',
    'O': '0',
    'D': '0',
    'Z': '2',
    'G': '6'
}
# ==================启发式规则配置区==========================

# ==================数据库配置===============================
DB_USERNAME = "VFJ_CQGS"
DB_PASSWORD = "vfj_20231007"
# Oracle的连接描述符(DSN)
DB_DSN = "192.168.1.200:1521/ORCL"
# 要查询的表名和列名
DB_TABLE_NAME = "SINGCHIPOBU"  # <--- 【请您将这里替换为真实的表名】
DB_COLUMN_NAME = "OBUSAMSERIALNO" # <--- 【请您将这里替换为真实的列名】

# --- 安全与同步配置 ---
# 用于保护 /refresh-cache 接口的API密钥
REFRESH_API_KEY = "Vfj@1234.wq" # 建议您后续修改为一个更复杂的密钥
# ==================数据库配置================================

# --- Flask 应用配置 ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16MB

# --- 日志配置 ---
LOG_DIR = "log"
LOG_FILE = "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024 # 10MB
LOG_FILE_BACKUP_COUNT = 5

# --- YOLOv8 配置 ---
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_MIN_DETECTION_AREA_PX = 2000  # 最小检测面积（像素）
YOLO_MAX_DETECTION_AREA_FACTOR = 0.1 # 最大检测面积占图像总面积的比例 (0.0 to 1.0)
YOLO_COCO_CLASSES = ['Barcode'] # 假设模型只检测一个类别

# --- OCR 配置 ---
OCR_TARGET_INPUT_HEIGHT = 48
OCR_DIGIT_ROI_Y_OFFSET_FACTOR = -0.15 # 数字区域相对于YOLO框顶部的Y偏移因子
OCR_DIGIT_ROI_HEIGHT_FACTOR = 0.7     # 数字区域高度相对于YOLO框高度的因子
OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05 # 数字区域宽度扩展因子
OCR_NUM_WORKERS = 4 # 并行OCR工作进程数 (0或1表示串行)
SAVE_TRAINING_ROI_IMAGES = True     # 是否保存在训练模式下使用的ROI切片 (预处理后，送入OCR前)

# --- 布局与状态管理配置 ---
LAYOUT_EXPECTED_TOTAL_ROWS = 13
LAYOUT_REGULAR_ROWS_COUNT = 12 # 逻辑上的常规行数 (不含特殊行)
LAYOUT_REGULAR_COLS_COUNT = 4
LAYOUT_SPECIAL_ROW_COLS_COUNT = 2 # 特殊行期望的列数
LAYOUT_TOTAL_OBUS_EXPECTED = 50 # 用于判断会话是否完成

# 用于 _analyze_layout_by_xy_clustering 的【固定像素距离】阈值
# 您需要根据实际OBU在图像中的像素间距来调整这些初始值
# 例如，如果同一行的OBU在Y方向上通常偏差在20像素内，那么Y_THRESHOLD可以设为25-30
# 如果同一列的OBU在X方向上通常偏差在80像素内，那么X_THRESHOLD可以设为90-100
LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD = 50  # Y轴方向上被认为是同一行的最大像素距离差 (建议值，请调整)
LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD = 400 # X轴方向上被认为是同一列的最大像素距离差 (建议值，请调整)

# 首次布局学习相关阈值 (来自 learn_initial_layout_from_yolo_v81 或类似函数的常量)
LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING = 5

# 用于 _learn_initial_stable_layout_params (现在主要是统计)
LAYOUT_MIN_CORE_ANCHORS_FOR_STATS = 3 # 用于统计稳定参数的最小锚点数 (替代之前的LEARNING)

# --- 过程图片保存 ---
SAVE_PROCESS_PHOTOS = True
PROCESS_PHOTO_DIR = "process_photo"
PROCESS_PHOTO_JPG_QUALITY = 65  # 过程JPEG图片质量 (0-100, 100为最高质量)

# --- 新增：零散识别模式配置 ---
SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH = 600 # 零散模式下返回的标注图宽度 (像素)
SCATTERED_MODE_IMAGE_JPG_QUALITY = 75    # 零散模式下返回的标注图JPEG质量

# --- 新增：高级标注与物料生成配置 ---
# 是否保存带有半透明颜色标注的原尺寸诊断图
SAVE_SCATTERED_ANNOTATED_IMAGE = True

# --- 有效OBU码列表 (模拟数据库) ---
VALID_OBU_CODES = {

}

# --- 版本号 ---
APP_VERSION = "v4.0_Optimized_Engine"
