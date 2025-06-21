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


# ==============================================================================
#                      模型与核心算法参数配置
# ==============================================================================
# --- YOLOv8 目标检测模型配置 ---
# ------------------------------------------------------------------------------

# YOLO模型检测的置信度阈值。
# 只有当模型认为一个区域“像条形码”的概率高于此值时，才会被接受。
# - 调高此值可以减少误检（比如把背景纹理当成条形码），但可能漏掉一些模糊的条形码。
# - 调低此值可以找到更多潜在的条形码，但会增加误检的风险。
YOLO_CONFIDENCE_THRESHOLD = 0.25

# 非极大值抑制（NMS）的IOU（交并比）阈值。
# 用于在多个重叠的检测框中，筛选出最佳的一个。
# - 值越小，抑制越强，对于紧密排列的OBU，可能会导致只剩下一个检测框。
# - 值越大，抑制越弱，可能会允许一些高度重叠的框同时存在。
YOLO_IOU_THRESHOLD = 0.45

# 允许的最小检测框面积（单位：像素）。
# 用于过滤掉那些尺寸过小的、不可能是有效OBU的检测结果。
# 可以有效避免将图像中的噪点或微小瑕疵误识别为OBU。
YOLO_MIN_DETECTION_AREA_PX = 2000

# 允许的最大检测框面积，以占总图像面积的百分比表示。
# 用于过滤掉那些尺寸异常巨大的、不可能是单个OBU的检测结果。
# 例如，0.1 表示检测框的面积不能超过总图像面积的10%。
YOLO_MAX_DETECTION_AREA_FACTOR = 0.1

# YOLO模型训练时使用的物体类别名称。
# 在我们的场景中，只有一个类别，即 'Barcode'。
YOLO_COCO_CLASSES = ['Barcode']


# --- OCR (光学字符识别) 模块配置 ---
# ------------------------------------------------------------------------------

# 在进行OCR识别前，所有从原图中裁剪出的ROI（感兴趣区域）都会被统一缩放到这个高度。
# 这个值需要与您训练OCR模型时使用的图像高度严格匹配，以获得最佳识别效果。
OCR_TARGET_INPUT_HEIGHT = 48

# 在YOLO检测框的基础上，对OBU数字部分的ROI进行垂直方向的微调。
# - 负值表示向上移动。-0.15 表示将ROI的起始点向上移动YOLO框高度的15%。
# 这是因为OBU上的数字通常位于条形码的上方。
OCR_DIGIT_ROI_Y_OFFSET_FACTOR = -0.15

# 在YOLO检测框的基础上，定义OBU数字部分ROI的高度。
# 0.7 表示数字区域的高度，大约是整个YOLO框高度的70%。
OCR_DIGIT_ROI_HEIGHT_FACTOR = 0.7

# 在YOLO检测框的基础上，对OBU数字部分ROI的宽度进行微调。
# 1.05 表示将ROI的宽度，在YOLO框宽度的基础上，再向两边各扩展5%，以确保数字的边缘不会被切掉。
OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05

# OCR并行处理的工作进程数。
# 这是提升性能的关键参数。一个好的实践是将其设置为服务器CPU的“物理核心数 - 1”。
# 例如，对于一个8核CPU的服务器，设置为7是一个非常理想的选择。
OCR_NUM_WORKERS = 7

# 是否保存在OCR识别前，经过预处理的ROI小图片。
# - 设置为 True: 会在 process_photo/training_rois/ 目录下，为每个会话生成大量的训练样本。
#   这在需要重新训练或优化模型时极其有用。
# - 设置为 False: 生产环境中建议关闭，以节省磁盘空间。
SAVE_TRAINING_ROI_IMAGES = True


# --- 【已废弃】布局与状态管理配置 ---
# ------------------------------------------------------------------------------
# 以下参数源自旧的“整版识别”逻辑，当前V19.0+的“零散识别”模式已不再使用它们。
# 保留它们是为了兼容性，但修改它们不会对当前系统产生任何影响。
LAYOUT_EXPECTED_TOTAL_ROWS = 13
LAYOUT_REGULAR_ROWS_COUNT = 12
LAYOUT_REGULAR_COLS_COUNT = 4
LAYOUT_SPECIAL_ROW_COLS_COUNT = 2
LAYOUT_TOTAL_OBUS_EXPECTED = 50
LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD = 50
LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD = 400
LAYOUT_MIN_CORE_ANCHORS_FOR_STATS = 3
VALID_OBU_CODES = {} # 为兼容旧模块导入而保留


# --- 调试图像与中间文件保存配置 ---
# ------------------------------------------------------------------------------

# 所有运行时产生的图片和文件的根目录。
PROCESS_PHOTO_DIR = "process_photo"

# 总开关：是否保存任何处理过程中的标注图像。
SAVE_PROCESS_PHOTOS = True

# 保存的标注图像的JPEG压缩质量（范围1-100，越高越清晰，文件越大）。
PROCESS_PHOTO_JPG_QUALITY = 65

# --- 零散识别模式下的特定配置 ---
# ------------------------------------------------------------------------------

# 在返回给客户端的Base64编码的标注图中，其统一的宽度。
# 将所有标注图缩放到一个固定宽度，可以方便前端展示，并减小网络传输的数据量。
SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH = 600

# 返回给客户端的Base64编码标注图的JPEG压缩质量。
# 这个值可以设得略高一些，以保证前端展示的清晰度。
SCATTERED_MODE_IMAGE_JPG_QUALITY = 75

# 是否保存每一帧处理后的、带有颜色高亮的、全尺寸的标注图到服务器磁盘。
# 文件将保存在 process_photo/scattered_annotated/ 目录下。
# - 设置为 True: 非常便于问题追溯和效果演示。
# - 设置为 False: 生产环境中如果不需要，可以关闭以节省磁盘空间。
SAVE_SCATTERED_ANNOTATED_IMAGE = True

# --- 版本号 ---
APP_VERSION = "v18.1_Final"