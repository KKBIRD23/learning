# coding: utf-8
"""
OBU (On-Board Unit) 镭标码区域识别脚本 - 直接使用PaddleOCR方案
版本: v2.6.4_tuning_and_docs
功能:
- 完善脚本说明和参数注释。
- 明确当前版本PaddleOCR (3.0.0) 通过构造函数参数加载非默认模型（如Server版）的行为。
- 提供参数配置区，方便对 Mobile 模型 (当前默认加载) 进行调优。
- 直接使用 PaddleOCR 对整张输入图片进行文本检测和识别。
- 根据预设规则筛选潜在的OBU码。
- 可视化原始检测结果和筛选后的OBU码结果。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr

# --- V2.6.4 配置参数 ---
VERSION = "v2.6.4_tuning_and_docs"
IMAGE_NAME = r"..\..\DATA\PIC\1pic\2-1.jpg" # 指向包含多个OBU的图片路径

# --- PaddleOCR 初始化相关参数 ---
# 参考文档: https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html#3

# 基础设置
LANG = 'en' # 识别语言
USE_TEXTLINE_ORIENTATION = False # 是否使用文本行方向分类 (通常OBU码是水平的)
USE_DOC_ORIENTATION_CLASSIFY = False # 是否使用文档方向分类 (通常单张OBU图不需要)
USE_DOC_UNWARPING = False # 是否使用文本图像矫正 (通常单张OBU图不需要)

# 模型选择:
# 注意: 根据测试 (PaddleOCR v3.0.0)，直接在构造函数中设置以下两个名称参数
# 可能仍会导致PaddleOCR加载其默认的Mobile系列模型。
# 要强制使用Server模型，更可靠的方式是手动下载Server模型，
# 然后通过 det_model_dir 和 rec_model_dir 参数指定它们的本地路径 (见下方注释掉的参数)。
TEXT_DETECTION_MODEL_NAME_CFG = "PP-OCRv5_mobile_det" # "PP-OCRv5_server_det" # 尝试Server版名称 (当前测试显示可能无效)
TEXT_RECOGNITION_MODEL_NAME_CFG = "PP-OCRv5_mobile_rec" # "PP-OCRv5_server_rec" # 尝试Server版名称 (当前测试显示可能无效)

# 如果已手动下载Server模型，请取消注释并设置以下路径:
# DET_MODEL_DIR_LOCAL_CFG = r"D:/PaddleOCR_Models/PP-OCRv5_server_det" # 示例路径
# REC_MODEL_DIR_LOCAL_CFG = r"D:/PaddleOCR_Models/PP-OCRv5_server_rec" # 示例路径
DET_MODEL_DIR_LOCAL_CFG = None
REC_MODEL_DIR_LOCAL_CFG = None

# 检测相关参数 (这些参数对于当前默认加载的Mobile模型是有效的)
# 详情请查阅 PaddleOCR 文档中关于检测模型参数的说明 (如DBNet的参数)
TEXT_DET_LIMIT_SIDE_LEN_CFG = 960 # 输入图像送入检测模型前的最长边限制。
                                  # 默认960。增大此值有助于检测小目标，但会增加推理时间。
                                  # 如果原图尺寸远大于此值，会被等比例缩放。
TEXT_DET_THRESH_CFG = 0.3         # 文本检测的像素级阈值 (0.0 ~ 1.0)。较低的值会检测更多区域。
TEXT_DET_BOX_THRESH_CFG = 0.6     # 文本检测的框级别阈值 (0.0 ~ 1.0)。较低的值会保留更多候选框。

# 识别相关参数
TEXT_REC_SCORE_THRESH_CFG = 0.5   # 文本识别结果的置信度阈值 (0.0 ~ 1.0)。
                                  # PaddleOCR引擎会只返回高于此阈值的识别结果。

# --- OBU码筛选规则 (后处理阶段) ---
OBU_CODE_PREFIX_FILTER = "5001" # OBU码期望的前缀
OBU_CODE_LENGTH_FILTER = 16     # OBU码期望的长度
# (如果 TEXT_REC_SCORE_THRESH_CFG 设得较低, 可以在这里再加一道后处理置信度筛选)
# MIN_OCR_CONFIDENCE_POST_FILTER = 0.6

# --- PaddleOCR引擎初始化 ---
paddle_ocr_engine = None
try:
    print("Initializing PaddleOCR engine globally...")
    ocr_init_params = {
        'lang': LANG,
        'use_textline_orientation': USE_TEXTLINE_ORIENTATION,
        'use_doc_orientation_classify': USE_DOC_ORIENTATION_CLASSIFY,
        'use_doc_unwarping': USE_DOC_UNWARPING,

        # 模型路径优先于模型名称
        'det_model_dir': DET_MODEL_DIR_LOCAL_CFG,
        'rec_model_dir': REC_MODEL_DIR_LOCAL_CFG,
        'text_detection_model_name': TEXT_DETECTION_MODEL_NAME_CFG, # 仅当 _model_dir 为None时可能生效
        'text_recognition_model_name': TEXT_RECOGNITION_MODEL_NAME_CFG, # 仅当 _model_dir 为None时可能生效

        'text_det_limit_side_len': TEXT_DET_LIMIT_SIDE_LEN_CFG, # 对应 det_limit_side_len (旧版)
        'text_det_thresh': TEXT_DET_THRESH_CFG,                 # 对应 det_db_thresh (旧版DBNet参数)
        'text_det_box_thresh': TEXT_DET_BOX_THRESH_CFG,         # 对应 det_db_box_thresh (旧版DBNet参数)
        'text_rec_score_thresh': TEXT_REC_SCORE_THRESH_CFG,     # 识别结果置信度阈值
    }
    # 移除值为None的参数，让PaddleOCR使用其内部默认值或基于其他参数的逻辑
    ocr_params_filtered = {k: v for k, v in ocr_init_params.items() if v is not None}

    print(f"PaddleOCR effective initialization parameters: {ocr_params_filtered}")
    paddle_ocr_engine = paddleocr.PaddleOCR(**ocr_params_filtered)
    print("PaddleOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR engine globally: {e}")
    traceback.print_exc()
    paddle_ocr_engine = None

timing_profile = {}

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.png"):
    # (此函数与V2.6.3版本一致)
    img_out = image.copy()
    if not all_ocr_data and not potential_obu_data :
        print("No OCR data to draw.")
        if img_out is not None:
            try:
                cv2.imwrite(output_path, img_out)
                print(f"Empty OCR data, base image saved to: {output_path}")
            except Exception as e_save:
                 print(f"Error saving base image: {e_save}")
        return
    if all_ocr_data:
        for item in all_ocr_data:
            box_polygon = item.get('box')
            if box_polygon is not None and isinstance(box_polygon, (list, np.ndarray)):
                points = np.array(box_polygon, dtype=np.int32)
                cv2.polylines(img_out, [points], isClosed=True, color=(0, 180, 0), thickness=1) # 原始检测为绿色细线
    drawn_potential_text_count = 0
    if potential_obu_data:
        for item in potential_obu_data:
            text = item['text']
            box_polygon = item.get('box')
            if box_polygon is None or not isinstance(box_polygon, (list, np.ndarray)): continue
            points = np.array(box_polygon, dtype=np.int32)
            cv2.polylines(img_out, [points], isClosed=True, color=(255, 0, 0), thickness=3) # 筛选出的OBU为蓝色粗线
            label = f"{text}"
            text_anchor_x = points[0][0]
            text_anchor_y = points[0][1] - 10
            if text_anchor_y < 15 : text_anchor_y = points[0][1] + 25
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_out, (text_anchor_x, text_anchor_y - text_height - baseline + 1),
                          (text_anchor_x + text_width, text_anchor_y + baseline -1), (220,220,220), -1) # 文字背景
            cv2.putText(img_out, label, (text_anchor_x, text_anchor_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 0), 2) # 文字颜色
            drawn_potential_text_count +=1
        print(f"Drawn {drawn_potential_text_count} potential OBU texts on the image.")
    try:
        if img_out is not None:
            cv2.imwrite(output_path, img_out)
            print(f"OCR visualization saved to: {output_path}")
        else:
            print("Error: img_out became None before saving visualization.")
    except Exception as e:
        print(f"Error saving visualization image to {output_path}: {e}")


# --- V2.6.4 主程序 ---
if __name__ == "__main__":
    t_start_overall = time.time()
    timing_profile['0_total_script_execution'] = 0
    print(f"--- OBU 识别工具 {VERSION} ---")

    if not paddle_ocr_engine:
        print("错误: PaddleOCR 引擎未能初始化。程序退出。")
        exit()
    if not os.path.exists(IMAGE_NAME):
        print(f"错误: 图片未找到: {IMAGE_NAME}")
        exit()

    try:
        t_start_img_read = time.time()
        original_image = cv2.imread(IMAGE_NAME)
        timing_profile['1_image_reading'] = time.time() - t_start_img_read
        if original_image is None: raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")
        orig_img_h, orig_img_w = original_image.shape[:2]
        print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['1_image_reading']:.2f} 秒读取)")

        print(f"\n--- 开始使用 PaddleOCR 对整张图片进行文本检测和识别 ---")
        t_start_paddleocr = time.time()
        ocr_prediction_result = paddle_ocr_engine.predict(original_image)
        timing_profile['2_paddleocr_prediction'] = time.time() - t_start_paddleocr
        print(f"PaddleOCR predict() 完成 ({timing_profile['2_paddleocr_prediction']:.3f} 秒)")

        all_extracted_ocr_data = []
        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]
            dt_polys = ocr_result_object.get('dt_polys')
            rec_texts = ocr_result_object.get('rec_texts')
            rec_scores = ocr_result_object.get('rec_scores')

            if rec_texts is not None and rec_scores is not None and dt_polys is not None:
                print(f"从PaddleOCR结果中提取到 {len(rec_texts)} 条原始文本 (置信度高于或等于 TEXT_REC_SCORE_THRESH_CFG: {TEXT_REC_SCORE_THRESH_CFG})。")
                for i in range(len(rec_texts)):
                    all_extracted_ocr_data.append({
                        "text": str(rec_texts[i]),
                        "score": float(rec_scores[i]),
                        "box": dt_polys[i]
                    })
            else: print("未能从 OCRResult 对象中提取到有效的 rec_texts, rec_scores 或 dt_polys。")
        else: print("PaddleOCR predict() 未返回有效结果或结果为空。")

        potential_obu_list = []
        if all_extracted_ocr_data:
            print(f"\n--- 开始根据内容规则筛选潜在的 OBU 码 (前缀='{OBU_CODE_PREFIX_FILTER}', 长度={OBU_CODE_LENGTH_FILTER}, 纯数字) ---")
            for data_item in all_extracted_ocr_data:
                text = data_item['text'].strip()
                # 此时的 score 已经是经过 PaddleOCR 内部 TEXT_REC_SCORE_THRESH_CFG 筛选过的
                if text.startswith(OBU_CODE_PREFIX_FILTER) and \
                   len(text) == OBU_CODE_LENGTH_FILTER and \
                   text.isdigit():
                    potential_obu_list.append(data_item)
            print(f"筛选完毕，找到 {len(potential_obu_list)} 个符合条件的潜在OBU码。")

        if potential_obu_list:
            print(f"\n--- 最终筛选出的 OBU 码列表 ({len(potential_obu_list)} 条) ---")
            # 按在图像中的位置（例如，从上到下，从左到右）排序，方便查看和后续矩阵映射
            # 这里用box的左上角y坐标，然后是x坐标作为排序依据
            potential_obu_list.sort(key=lambda item: (item['box'][0][1], item['box'][0][0]))
            for i, obu_data in enumerate(potential_obu_list):
                 print(f"  OBU {i+1}: Text='{obu_data['text']}', Score: {obu_data['score']:.4f}, Box (TL_y, TL_x): ({obu_data['box'][0][1]}, {obu_data['box'][0][0]})")
        else: print("没有筛选出符合所有条件的OBU码。")

        output_fn_base = os.path.splitext(os.path.basename(IMAGE_NAME))[0]
        visualization_path = f"output_{output_fn_base}_{VERSION}.png"
        if original_image is not None:
            draw_ocr_results_refined(original_image, all_extracted_ocr_data, potential_obu_list, visualization_path)
        else:
            print("错误：original_image 未加载，无法进行可视化。")

    except FileNotFoundError as e: print(f"文件未找到错误: {e}")
    except Exception as e: print(f"发生未处理的错误: {e}"); traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 ({VERSION}) ---")
        sorted_timing_keys = sorted(timing_profile.keys(), key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 99)
        for key in sorted_timing_keys: print(f"  {key}: {timing_profile[key]:.3f} 秒")
        print(f"------------------------------")