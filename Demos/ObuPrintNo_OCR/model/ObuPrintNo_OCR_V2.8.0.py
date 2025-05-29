# coding: utf-8
"""
OBU (车载单元) 镭标码识别与参数调优脚本
版本: v2.8.4_syntax_fix_and_comments
功能:
- [修正] 修正V2.8.3中可能存在的语法错误及日志参数记录问题。
- [注释] 增加更详细的代码注释和说明。
- [核心] 实现 PaddleOCR 参数的自动扫描与评估。
- [日志] 输出详细的CSV日志，包含每个参数组、每张图片、每个识别出的OBU的详细信息 (中文表头)。
- [显示] 在单参数组运行时，向控制台打印详细的OBU识别列表。
- [配置] 允许用户方便地配置要扫描的参数范围或运行单组黄金参数。
- [模型] 默认使用 PaddleOCR Mobile 模型进行测试。
- [筛选] 根据预设规则 (前缀、长度、纯数字) 筛选潜在的OBU码。
- [可视化] 为每个参数组合处理的每张图片生成标注了识别结果的可视化图像。
"""
import cv2
import numpy as np
import os
import time
import traceback # 用于打印详细的异常信息
import paddleocr
from itertools import product # 用于生成参数组合
import csv # 用于读写CSV文件
from datetime import datetime # 用于生成带时间戳的文件名

# --- V2.8.4 配置参数 ---
VERSION = "v2.8.4_syntax_fix_and_comments"

# --- 输入和输出路径配置 ---
IMAGE_PATHS = [
    r"./PIC/1.JPG",
    r"./PIC/2.JPG",
    r"./PIC/3.JPG"
]  # 要处理的图片路径列表
BASE_OUTPUT_DIR = "./output_v2.8_scan_results"  # 所有运行结果的根输出目录
TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S") # 当前时间戳
CURRENT_RUN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP_NOW}_{VERSION}") # 本次运行的独立输出文件夹
LOG_FILE_PATH = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"扫描日志_{VERSION}_{TIMESTAMP_NOW}.csv") # CSV日志文件名
os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True) # 创建输出目录

# --- PaddleOCR 初始化相关参数 (基础设置) ---
LANG_CFG = 'en'  # 识别语言, 'en' 表示英文
USE_TEXTLINE_ORIENTATION_CFG = False # 是否使用文本行方向分类 (OBU码通常是水平的，关闭可提速)
USE_DOC_ORIENTATION_CLASSIFY_CFG = False # 是否使用文档方向分类 (通常单张OBU图片不需要)
USE_DOC_UNWARPING_CFG = False      # 是否使用文本图像矫正 (通常单张OBU图片不需要)
OCR_VERSION_CFG = None     # 指定使用的PaddleOCR版本，确保一致性。None表示库的默认。

# --- 模型加载配置 ---
# 当前默认使用 PaddleOCR 自动下载的 Mobile 模型。
# 如果要使用手动下载的 Server 模型，请设置下面的 `_DIR_CFG` 路径，并将 `_NAME_CFG` 设为 None。
TEXT_DETECTION_MODEL_DIR_CFG = None  # 例如: r"C:\path\to\your\PP-OCRv5_server_det_infer"
TEXT_RECOGNITION_MODEL_DIR_CFG = None  # 例如: r"C:\path\to\your\PP-OCRv5_server_rec_infer"
TEXT_DETECTION_MODEL_NAME_CFG = None   # 当使用 _DIR_CFG 时，此项应为 None
TEXT_RECOGNITION_MODEL_NAME_CFG = None   # 当使用 _DIR_CFG 时，此项应为 None

# --- 参数运行模式 ---
RUN_PARAMETER_SCAN = False  # True: 执行参数扫描; False: 只运行下面的 GOLD_STANDARD_PARAM_SET

# “黄金参数组合” (当 RUN_PARAMETER_SCAN = False 时使用，基于之前测试效果较好的一组)
GOLD_STANDARD_PARAM_SET = {
    "text_det_limit_side_len": 960, # 检测时图像最长边限制
    "text_det_thresh": 0.3,         # 检测时像素级二值化阈值
    "text_det_box_thresh": 0.6,     # 检测时文本框置信度阈值
    "text_rec_score_thresh": 0.5,   # 识别结果置信度阈值 (PaddleOCR内部使用)
}

# 用于参数扫描的参数网格 (当 RUN_PARAMETER_SCAN = True 时使用)
PARAMETER_GRID_FOR_SCAN = {
    'text_det_limit_side_len': [960, 1280],       # 尝试不同的图像尺寸限制
    'text_det_thresh': [0.25, 0.3],             # 尝试不同的检测像素阈值
    'text_det_box_thresh': [0.5, 0.6],          # 尝试不同的检测框阈值
    'text_rec_score_thresh': [0.45, 0.5, 0.55], # 尝试不同的识别置信度阈值
}

# --- OBU码筛选规则 (后处理阶段) ---
OBU_CODE_PREFIX_FILTER_CFG = "5001" # OBU码期望的前缀
OBU_CODE_LENGTH_FILTER_CFG = 16     # OBU码期望的长度

# --- 全局变量 ---
paddle_ocr_engine_global = None # 全局PaddleOCR引擎实例
# CSV日志的表头 (中文)
CSV_HEADER = [
    "参数集索引", "图片名称",
    "检测边长限制", "检测像素阈值", "检测框阈值", "识别分数阈值", # 当前运行的参数
    "PaddleOCR预测耗时_秒", "原始检测文本数", "图片筛选OBU总数", # 图片整体指标
    "OBU序号_图片内", "筛选出的OBU文本", "OBU识别置信度", # 单个OBU指标
    "错误信息" # 记录处理过程中的任何错误
]
# 用于从Python参数字典的键名映射到CSV中文表头的键名
PARAM_TO_CSV_HEADER_MAP = {
    "text_det_limit_side_len": "检测边长限制",
    "text_det_thresh": "检测像素阈值",
    "text_det_box_thresh": "检测框阈值",
    "text_rec_score_thresh": "识别分数阈值",
}

# --- 函数定义 ---

def initialize_paddleocr(current_ocr_config_params):
    """
    根据给定的参数初始化或重新初始化全局的PaddleOCR引擎。
    Args:
        current_ocr_config_params (dict): 包含本次运行要使用的细粒度OCR参数的字典。
                                         例如: {"text_det_thresh": 0.25, ...}
    Returns:
        bool: True如果初始化成功, False如果失败。
    """
    global paddle_ocr_engine_global

    # 构建传递给 PaddleOCR 构造函数的基础参数字典
    init_params_for_engine = {
        'lang': LANG_CFG,
        'use_textline_orientation': USE_TEXTLINE_ORIENTATION_CFG,
        'use_doc_orientation_classify': USE_DOC_ORIENTATION_CLASSIFY_CFG,
        'use_doc_unwarping': USE_DOC_UNWARPING_CFG,
        'ocr_version': OCR_VERSION_CFG, # 例如 "PP-OCRv5"

        # 模型加载: 优先使用本地路径；如果本地路径未提供，则尝试使用模型名称。
        # 注意：PaddleOCR的参数名可能与我们配置变量名略有不同，这里用的是构造函数期望的参数名。
        'text_detection_model_dir': TEXT_DETECTION_MODEL_DIR_CFG,
        'text_recognition_model_dir': TEXT_RECOGNITION_MODEL_DIR_CFG,
        'text_detection_model_name': TEXT_DETECTION_MODEL_NAME_CFG,
        'text_recognition_model_name': TEXT_RECOGNITION_MODEL_NAME_CFG,
    }

    # 将当前参数扫描组合中的动态参数更新/覆盖到基础参数中
    if current_ocr_config_params: # current_ocr_config_params 是本次扫描用的参数
        init_params_for_engine.update(current_ocr_config_params)

    # 从最终的参数字典中移除值为None的条目，以便PaddleOCR对这些参数使用其内部默认值
    ocr_params_final_filtered = {k: v for k, v in init_params_for_engine.items() if v is not None}

    print(f"\n正在使用以下参数初始化PaddleOCR: {ocr_params_final_filtered}")
    try:
        # **关键修正**: 移除了之前错误的 show_log=False
        paddle_ocr_engine_global = paddleocr.PaddleOCR(**ocr_params_final_filtered)
        print("PaddleOCR引擎初始化/重新初始化成功。")
        # (可以保留尝试打印实际加载模型信息的代码，如果需要)
        return True
    except Exception as e:
        print(f"PaddleOCR引擎初始化失败: {e}")
        # traceback.print_exc() # 需要详细错误堆栈时取消注释
        paddle_ocr_engine_global = None
        return False

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.png"):
    """
    在图片上绘制PaddleOCR的原始检测结果和筛选后的OBU结果。
    Args:
        image (numpy.ndarray): 原始OpenCV图像 (BGR格式)。
        all_ocr_data (list): 包含所有原始提取OCR数据的列表。
                             每个元素是一个字典: {"text": str, "score": float, "box": list_of_points}
        potential_obu_data (list): 包含筛选后认为是OBU码的数据的列表，结构同上。
        output_path (str): 可视化结果图片的保存路径。
    """
    img_out = image.copy()
    _c = cv2 # 使用别名简化后续cv2的调用

    if img_out is None:
        print(f"错误: 用于绘制的输入图像为None。无法保存到 {output_path}")
        return

    # 如果没有任何OCR数据（原始或筛选后），也保存一张图片（可能是原图或空白图的副本）
    if not all_ocr_data and not potential_obu_data :
        print(f"没有OCR数据可以绘制到 {output_path}.")
        try:
            _c.imwrite(output_path, img_out)
            print(f"无OCR数据, 底图已保存到: {output_path}")
        except Exception as e_save:
            print(f"保存底图失败 {output_path}: {e_save}")
        return

    # 1. 绘制所有原始检测框 (用细的绿色线条)
    if all_ocr_data:
        for item in all_ocr_data:
            box_polygon = item.get('box') # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # 确保box_polygon是有效的点列表
            if box_polygon is not None and isinstance(box_polygon, (list, np.ndarray)) and len(box_polygon) > 0 :
                try:
                    points = np.array(box_polygon, dtype=np.int32)
                    _c.polylines(img_out, [points], isClosed=True, color=(0, 180, 0), thickness=1) # 绿色细线
                except Exception as e_draw_poly:
                    print(f"警告: 无法为检测框 {box_polygon} 绘制多边形. 错误: {e_draw_poly}")

    # 2. 绘制筛选出的Potential OBU (用粗的蓝色框，并标注识别文本)
    drawn_potential_text_count = 0
    if potential_obu_data:
        for item in potential_obu_data:
            text = item['text']
            box_polygon = item.get('box')
            if box_polygon is None or not isinstance(box_polygon, (list, np.ndarray)) or len(box_polygon) == 0:
                continue
            try:
                points = np.array(box_polygon, dtype=np.int32)
                _c.polylines(img_out, [points], isClosed=True, color=(255, 0, 0), thickness=3) # 蓝色粗线

                label = f"{text}" # 可以加上分数: f"{text} ({item['score']:.2f})"

                # 计算文本标注位置 (在框的左上角上方一点)
                text_anchor_x = points[0][0]
                text_anchor_y = points[0][1] - 10
                if text_anchor_y < 15 : text_anchor_y = points[0][1] + 25 # 防止文字超出图片顶部

                # 给文本添加背景使其更易读
                (text_width, text_height), baseline = _c.getTextSize(label, _c.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                _c.rectangle(img_out,
                             (text_anchor_x, text_anchor_y - text_height - baseline + 1),
                             (text_anchor_x + text_width, text_anchor_y + baseline -1),
                             (220,220,220), -1) # 浅灰色背景
                _c.putText(img_out, label, (text_anchor_x, text_anchor_y),
                            _c.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 0), 2) # 深蓝色文本
                drawn_potential_text_count +=1
            except Exception as e_draw_potential:
                print(f"警告: 无法为潜在OBU绘制检测框 {box_polygon}. 错误: {e_draw_potential}")

        if drawn_potential_text_count > 0:
            print(f"已在图上绘制 {drawn_potential_text_count} 个潜在OBU的文本。")

    # 保存最终的可视化图片
    try:
        _c.imwrite(output_path, img_out)
        print(f"OCR可视化结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存可视化图片失败 {output_path}: {e}")

def process_single_image_for_scan(image_path, current_run_ocr_params_dict, param_set_str_for_filename, param_set_idx_for_log):
    print(f"正在处理图片: {image_path}")
    img_filename_base = os.path.splitext(os.path.basename(image_path))[0]

    base_log_info = {"参数集索引": param_set_idx_for_log, "图片名称": img_filename_base}
    for param_key_py, param_key_csv in PARAM_TO_CSV_HEADER_MAP.items():
        base_log_info[param_key_csv] = current_run_ocr_params_dict.get(param_key_py)
    base_log_info.update({"PaddleOCR预测耗时_秒": -1.0, "原始检测文本数": 0, "图片筛选OBU总数": 0, "错误信息": "无"})

    if not paddle_ocr_engine_global:
        base_log_info["错误信息"] = "PaddleOCR引擎未初始化"; return [base_log_info]

    log_entries_for_this_image = []
    original_image = None
    all_extracted_ocr_data = [] # 在try块外部定义，确保finally或异常时也能访问（虽然这里没用finally）
    num_filtered_obus_on_image_count = 0
    potential_obu_details_for_drawing = []

    try:
        original_image = cv2.imread(image_path)
        if original_image is None: base_log_info["错误信息"] = "图片读取失败"; return [base_log_info]

        t_start_predict = time.time()
        ocr_prediction_result = paddle_ocr_engine_global.predict(original_image)
        base_log_info['PaddleOCR预测耗时_秒'] = round(time.time() - t_start_predict, 3)

        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]
            dt_polys = ocr_result_object.get('dt_polys')
            rec_texts = ocr_result_object.get('rec_texts')
            rec_scores = ocr_result_object.get('rec_scores')

            # --- 核心修改：对齐V2.6.4的解析逻辑 ---
            if rec_texts is not None and rec_scores is not None and dt_polys is not None:
                num_rec_texts = len(rec_texts)
                print(f"  图片 '{img_filename_base}': PaddleOCR原始返回 len(rec_texts)={num_rec_texts}, len(rec_scores)={len(rec_scores)}, len(dt_polys)={len(dt_polys)}")

                # 以 rec_texts 的长度为基准进行迭代，并确保不会对 rec_scores 和 dt_polys 造成索引越界
                # 这是V2.6.4能成功处理的关键（假设 rec_texts 和 rec_scores 长度一致）
                if num_rec_texts > 0 and num_rec_texts == len(rec_scores): # 确保识别文本和分数数量一致
                    # 并且确保我们不会因为 dt_polys 更短而越界 (虽然通常 dt_polys >= rec_texts)
                    max_items_to_process = min(num_rec_texts, len(dt_polys) if dt_polys is not None else 0)
                    if max_items_to_process < num_rec_texts:
                         print(f"  警告: 检测框数量 ({len(dt_polys) if dt_polys is not None else 0}) 少于识别文本数量 ({num_rec_texts})。将按较小值处理。")

                    for i in range(max_items_to_process): # 使用确保安全的迭代次数
                        all_extracted_ocr_data.append({
                            "text": str(rec_texts[i]),
                            "score": float(rec_scores[i]),
                            "box": dt_polys[i]
                        })
                elif num_rec_texts == 0:
                    print(f"  图片 '{img_filename_base}': 未识别到任何文本 (rec_texts is empty or None).")
                else: # rec_texts 和 rec_scores 长度不一致，这是个问题
                    warning_msg = (f"警告(图片: {img_filename_base}): rec_texts({num_rec_texts}) 和 "
                                   f"rec_scores({len(rec_scores) if rec_scores is not None else 'None'}) 长度不匹配！无法安全组合。")
                    print(warning_msg)
                    if base_log_info["错误信息"] == "无": base_log_info["错误信息"] = warning_msg
            else:
                print(f"警告(图片: {img_filename_base}): rec_texts, rec_scores, 或 dt_polys 为空或None。")
                if base_log_info["错误信息"] == "无": base_log_info["错误信息"] = "rec_texts, rec_scores, 或 dt_polys 为空或None"

        base_log_info["原始检测文本数"] = len(all_extracted_ocr_data)

        if all_extracted_ocr_data:
            for data_item in all_extracted_ocr_data:
                text = data_item['text'].strip()
                score = data_item['score']
                if text.startswith(OBU_CODE_PREFIX_FILTER_CFG) and \
                   len(text) == OBU_CODE_LENGTH_FILTER_CFG and \
                   text.isdigit():
                    num_filtered_obus_on_image_count += 1
                    obu_log_entry = {**base_log_info, "图片筛选OBU总数": 0, "OBU序号_图片内": num_filtered_obus_on_image_count, "筛选出的OBU文本": text, "OBU识别置信度": round(score, 4)}
                    log_entries_for_this_image.append(obu_log_entry)
                    potential_obu_details_for_drawing.append(data_item)

        for entry in log_entries_for_this_image: entry["图片筛选OBU总数"] = num_filtered_obus_on_image_count
        print(f"  图片 '{img_filename_base}': 原始有效文本数={base_log_info['原始检测文本数']}, 筛选后OBU数={num_filtered_obus_on_image_count}, 耗时={base_log_info['PaddleOCR预测耗时_秒']:.2f}s")
        if not log_entries_for_this_image: no_obu_entry = {**base_log_info, "图片筛选OBU总数":0, "OBU序号_图片内": 0, "筛选出的OBU文本": "无", "OBU识别置信度": 0.0}; log_entries_for_this_image.append(no_obu_entry)

        viz_filename = f"output_{img_filename_base}_{param_set_str_for_filename}_{VERSION}.png"; visualization_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, viz_filename)
        if original_image is not None : draw_ocr_results_refined(original_image, all_extracted_ocr_data, potential_obu_details_for_drawing, visualization_path)

        return log_entries_for_this_image
    except Exception as e:
        print(f"处理图片 {image_path} 时发生严重错误: {e}"); traceback.print_exc(); base_log_info["错误信息"] = f"严重错误: {str(e)}"
        if not log_entries_for_this_image: return [base_log_info]
        else:
            for entry in log_entries_for_this_image:
                if entry.get("错误信息", "无") == "无": entry["错误信息"] = f"后续处理错误: {str(e)}"
            return log_entries_for_this_image


# --- 主程序 ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"--- OBU识别与参数调优工具 {VERSION} ---")
    print(f"输出目录: {os.path.abspath(CURRENT_RUN_OUTPUT_DIR)}")
    print(f"日志文件: {LOG_FILE_PATH}")

    # 根据 RUN_PARAMETER_SCAN 的设置，决定是进行参数扫描还是单次运行黄金参数
    if RUN_PARAMETER_SCAN:
        param_grid_to_use = PARAMETER_GRID_FOR_SCAN
        print("模式: 参数扫描已启用。")
    else:
        param_grid_to_use = {k: [v] for k, v in GOLD_STANDARD_PARAM_SET.items()} # 将单组参数转为扫描器期望的格式
        print(f"模式: 单参数组运行 (黄金参数): {GOLD_STANDARD_PARAM_SET}")

    # 获取所有参数的名称 (用于后续构建字典)
    param_names_for_scan = list(param_grid_to_use.keys())
    # 生成所有参数值的组合
    param_value_combinations = list(product(*(param_grid_to_use[val] for val in param_names_for_scan)))
    total_param_sets_to_run = len(param_value_combinations)
    print(f"待评估的参数组合总数: {total_param_sets_to_run}")

    all_csv_rows_to_write = [] # 用于存储所有将写入CSV的行数据

    # 打开CSV文件准备写入 (使用追加模式 'a' 可能更安全，但每次运行新日志用 'w' 合理)
    try:
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER, extrasaction='ignore')
            writer.writeheader() # 首先写入表头

            # 外层循环：遍历每一种参数组合
            for i_param_set, current_param_values_tuple in enumerate(param_value_combinations):
                # 将当前参数值元组与参数名对应，构建字典
                current_ocr_params_for_init_and_log = dict(zip(param_names_for_scan, current_param_values_tuple))

                # 为可视化文件名生成简洁的参数字符串
                param_set_label_for_file = "params"
                for name, val in current_ocr_params_for_init_and_log.items():
                    short_name = name.split('_')[-1] if '_' in name else name # 取参数名最后一部分
                    param_set_label_for_file += f"_{short_name[:4]}{val}" # 例如 _lsl960_t0.3...

                print(f"\n===== 正在运行参数集 {i_param_set+1}/{total_param_sets_to_run}: {current_ocr_params_for_init_and_log} =====")

                # 为当前参数集初始化/重新初始化OCR引擎
                if not initialize_paddleocr(current_ocr_config_params=current_ocr_params_for_init_and_log):
                    print(f"严重错误: PaddleOCR引擎初始化失败。跳过此参数集。")
                    # 记录初始化失败到CSV
                    error_log_entry = {
                        "参数集索引": i_param_set + 1,
                        "图片名称": "初始化失败",
                        **{PARAM_TO_CSV_HEADER_MAP.get(k,k):v for k,v in current_ocr_params_for_init_and_log.items()}, # 记录参数
                        "错误信息": "PaddleOCR引擎初始化失败"
                    }
                    all_csv_rows_to_write.append(error_log_entry) # 也加入总列表
                    writer.writerow({h: error_log_entry.get(h, "") for h in CSV_HEADER}) # 写入CSV
                    continue

                detailed_obu_output_for_console_this_set = [] # 用于单次运行时屏幕打印

                # 内层循环：使用当前参数集处理所有图片
                for image_path_to_process in IMAGE_PATHS:
                    if not os.path.exists(image_path_to_process):
                        print(f"警告: 图片路径未找到，跳过: {image_path_to_process}")
                        file_not_found_entry = {
                            "参数集索引": i_param_set + 1,
                            "图片名称": os.path.basename(image_path_to_process),
                            **{PARAM_TO_CSV_HEADER_MAP.get(k,k):v for k,v in current_ocr_params_for_init_and_log.items()},
                            "错误信息": "图片文件未找到"
                        }
                        all_csv_rows_to_write.append(file_not_found_entry)
                        writer.writerow({h: file_not_found_entry.get(h, "") for h in CSV_HEADER})
                        continue

                    # 调用处理单张图片的函数，它会返回一个日志条目列表
                    log_entries_from_image = process_single_image_for_scan(
                        image_path_to_process,
                        current_ocr_params_for_init_and_log, # 传递当前参数用于日志记录
                        param_set_label_for_file,
                        i_param_set + 1 # 传递参数集索引
                    )
                    all_csv_rows_to_write.extend(log_entries_from_image) # 加入总列表
                    for entry in log_entries_from_image: # 逐条写入CSV
                        writer.writerow({h: entry.get(h, "") for h in CSV_HEADER})

                    # 如果是单参数组运行模式 (非扫描)，收集详细OBU用于后续控制台打印
                    if not RUN_PARAMETER_SCAN:
                        for entry in log_entries_from_image:
                            # 确保只添加包含实际OBU文本的条目
                            if entry.get("筛选出的OBU文本", "无") not in ["无", "N/A"] and entry.get("错误信息", "无") == "无":
                                 detailed_obu_output_for_console_this_set.append(
                                     f"  图片: {entry['图片名称']}, OBU序号: {entry['OBU序号_图片内']}, "
                                     f"文本: '{entry['筛选出的OBU文本']}', 置信度: {entry['OBU识别置信度']:.4f}"
                                 )

                # 如果是单参数组运行模式，并且有详细OBU结果，则在处理完该参数集的所有图片后打印
                if not RUN_PARAMETER_SCAN and detailed_obu_output_for_console_this_set:
                    print(f"\n--- 参数集 {i_param_set+1} ({GOLD_STANDARD_PARAM_SET}) 的详细OBU识别结果 ---")
                    for line in detailed_obu_output_for_console_this_set:
                        print(line)

        print(f"\n详细扫描结果已保存到CSV: {LOG_FILE_PATH}")

    except IOError as e_io:
        print(f"写入CSV文件时发生IO错误: {LOG_FILE_PATH}. 错误: {e_io}")
    except Exception as e_main:
        print(f"主程序发生未预料的错误: {e_main}")
        traceback.print_exc()


    # --- 控制台打印总结报告 ---
    print("\n\n--- 参数扫描总体总结 (详情请查阅CSV日志) ---")
    if not all_csv_rows_to_write: # 检查是否有任何数据被处理
        print("没有参数集被成功运行或没有结果可供总结。")
    else:
        summary_by_param_set = {} # 用于聚合每个参数集的结果
        # 从 all_csv_rows_to_write (包含了所有图片的细粒度日志) 中聚合数据
        for row_data in all_csv_rows_to_write:
            param_idx = row_data.get("参数集索引", "未知参数集")

            # 为总结报告正确提取参数值 (使用PARAM_TO_CSV_HEADER_MAP的反向查找或直接用param_names_for_scan)
            current_params_for_summary_display = {}
            if param_names_for_scan: # 确保 param_names_for_scan 已定义
                for py_param_name in param_names_for_scan:
                    csv_header_name = PARAM_TO_CSV_HEADER_MAP.get(py_param_name)
                    if csv_header_name and csv_header_name in row_data:
                         current_params_for_summary_display[py_param_name] = row_data[csv_header_name]
                    elif py_param_name in row_data: # Fallback if mapping is missing for some reason
                         current_params_for_summary_display[py_param_name] = row_data[py_param_name]


            if param_idx not in summary_by_param_set:
                summary_by_param_set[param_idx] = {
                    "params_display_str": str(current_params_for_summary_display), # 用于显示的参数字符串
                    "params_dict": current_params_for_summary_display, # 参数字典
                    "processed_image_names": set(), # 用集合统计处理了多少不同的图片
                    "total_filtered_obus_for_set": 0,
                    "init_or_processing_error_count": 0
                }

            # 检查是否有错误
            error_message = row_data.get("错误信息", "无")
            if error_message != "无" and error_message != "N/A" and error_message != "":
                summary_by_param_set[param_idx]["init_or_processing_error_count"] += 1
                # 如果是初始化失败的特殊标记行，图片名可能是 "初始化失败" 或 "N/A_INIT_FAIL"
                if row_data.get("图片名称") not in ["初始化失败", "N/A_INIT_FAIL"]:
                     summary_by_param_set[param_idx]["processed_image_names"].add(row_data["图片名称"]) # 仍然算作处理过
                continue # 有错误的行不计入OBU数统计

            # 统计成功处理的图片和OBU
            summary_by_param_set[param_idx]["processed_image_names"].add(row_data["图片名称"])
            if row_data.get("筛选出的OBU文本", "无") not in ["无", "N/A"]: # 确保是有效的OBU条目
                 summary_by_param_set[param_idx]["total_filtered_obus_for_set"] += 1

        # 打印聚合后的总结
        sorted_param_indices = sorted(summary_by_param_set.keys())
        best_avg_obus_console = -1.0 # 使用浮点数
        best_param_set_info_console = None

        for param_idx_key in sorted_param_indices:
            set_data = summary_by_param_set[param_idx_key]
            params_display_str = set_data["params_display_str"]
            print(f"\n参数集 {param_idx_key}: {params_display_str}")

            if set_data["init_or_processing_error_count"] > 0 and not set_data["processed_image_names"]:
                print(f"  错误: PaddleOCR引擎初始化失败或所有图片处理均失败。")
                continue

            num_unique_images_processed = len(set_data["processed_image_names"])
            total_obus_in_set = set_data["total_filtered_obus_for_set"]

            if num_unique_images_processed > 0:
                avg_obus = total_obus_in_set / num_unique_images_processed
                print(f"  成功处理图片数: {num_unique_images_processed}, 共识别OBU: {total_obus_in_set}, 平均每图OBU: {avg_obus:.2f}")
                # 更新最佳参数集 (确保所有预期的图片都被处理了)
                if avg_obus > best_avg_obus_console and num_unique_images_processed == len(IMAGE_PATHS):
                    best_avg_obus_console = avg_obus
                    best_param_set_info_console = {
                        "params": set_data["params_dict"], # 使用字典形式的参数
                        "avg_obus": avg_obus,
                        "total_obus":total_obus_in_set,
                        "img_count":num_unique_images_processed
                    }
            else:
                 print("  此参数集未能成功处理任何图片或未识别到OBU（可能所有图片都有错误）。")

        if best_param_set_info_console:
            print("\n--- 基于平均OBU数的推荐参数集 (仅供参考) ---")
            print(f"  参数: {best_param_set_info_console['params']}")
            print(f"  平均OBU数: {best_param_set_info_console['avg_obus']:.2f} (总计: {best_param_set_info_console['total_obus']} 来自 {best_param_set_info_console['img_count']} 张图片)")
            print("提醒: 真实最佳参数需结合可视化图片进行人工判断。")
        else:
            print("\n未能找到在所有图片上都表现良好的推荐参数集。")

    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。")
    print(f"-------------------------------------------------")