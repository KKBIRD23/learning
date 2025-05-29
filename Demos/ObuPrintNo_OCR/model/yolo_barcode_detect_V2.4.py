import cv2
import numpy as np
import onnxruntime
import os
import time
import traceback
import paddle
import paddleocr

# --- PaddleOCR引擎初始化 ---
paddle_ocr_engine = None
try:
    print("Initializing PaddleOCR engine globally...")
    paddle_ocr_engine = paddleocr.PaddleOCR(lang='en')
    print("PaddleOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR engine globally: {e}")
    print("PaddleOCR features will be disabled.")
    paddle_ocr_engine = None

# --- V2.4 配置参数 ---
VERSION = "v2.4.4_digit_roi_upper_half"
ONNX_MODEL_PATH = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx" # <--- 新模型路径
IMAGE_NAME = r"./PIC/1.JPG"
CONFIDENCE_THRESHOLD = 0.25 # 可以根据新模型调整，0.25是一个不错的起点
IOU_THRESHOLD = 0.45      # NMS的IOU阈值

# --- 切块逻辑配置 ---
ENABLE_TILING = False # <--- 默认禁用切块，优先使用新模型的整图推理
FIXED_TILING_GRID = None
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5
TILE_OVERLAP_RATIO = 0.2

# --- 检测结果面积筛选配置 ---
MIN_DETECTION_AREA = 2000 # 根据新模型的检测框大小调整 (之前是9000)
MAX_DETECTION_AREA = 0.1 # 占图像总面积的比例 (之前是0.01，可以适当放宽)

# --- V2.4.7 数字ROI裁剪精调参数 ---
# 数字ROI的上边缘相对于YOLO检测框上边缘的偏移因子 (以YOLO框高度为单位)
# 负值表示向上偏移 (我们期望数字在条形码上方，所以通常是负值)
# 例如：-0.2 表示数字ROI的上边缘比YOLO框的上边缘还要高出YOLO框高度的20%
#        0.0 表示数字ROI的上边缘与YOLO框上边缘对齐
#        0.1 表示数字ROI的上边缘在YOLO框上边缘下方10%的位置 (不太可能用于我们的场景)
DIGIT_ROI_Y_OFFSET_FACTOR = -0.2  # <--- 初始尝试值：向上偏移YOLO框高度的20%

# 数字ROI的高度相对于YOLO检测框总高度的比例因子
DIGIT_ROI_HEIGHT_FACTOR = 0.6    # <--- 初始尝试值：数字区域高度是YOLO框高度的60%

# 数字ROI的宽度相对于YOLO检测框宽度的扩展因子
DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05

# 选择送给PaddleOCR的预处理类型: "color_digit", "gray_digit", "binary_otsu_digit", "binary_adaptive_digit"
# 或者设置为 "all" 来尝试所有，或者一个列表 ["binary_otsu_digit", "gray_digit"]
OCR_PREPROCESS_TYPE_TO_USE = "binary_otsu_digit" # <--- 新增：优先使用OTSU二值化

COCO_CLASSES = ['Barcode'] # 明确我们的模型是单类别

timing_profile = {}
process_photo_dir = "process_photo" # 过程图片保存目录

# --- 清理过程图片文件夹的函数 ---
def clear_process_photo_directory(directory="process_photo"):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            except Exception as e: print(f'Failed to delete {file_path}. Reason: {e}')
    else: os.makedirs(directory, exist_ok=True)

# --- 原V2.2/V2.3的切块相关函数 (保留但不一定调用) ---
def preprocess_image_data_for_tiling(img_data, input_shape_hw): # 重命名以区分
    # ... (与您V2.3中的 preprocess_image_data 内容一致) ...
    img = img_data
    if img is None: raise ValueError("输入图像数据为空 (tiling)")
    img_height, img_width = img.shape[:2]
    input_height, input_width = input_shape_hw
    ratio = min(input_width / img_width, input_height / img_height)
    new_width, new_height = int(img_width * ratio), int(img_height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 128, dtype=np.uint8)
    x_pad, y_pad = (input_width - new_width) // 2, (input_height - new_height) // 2
    canvas[y_pad:y_pad + new_height, x_pad:x_pad + new_width] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), ratio, x_pad, y_pad

def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold_tile, model_output_channels_param_ignored): # 重命名conf_threshold
    # ... (与您V2.3中的 postprocess_detections_from_tile 内容一致, 使用 actual_model_output_channels) ...
    predictions_raw = np.squeeze(outputs[0])
    if predictions_raw.ndim != 2: return np.array([]), np.array([]), np.array([])
    actual_model_output_channels = predictions_raw.shape[0]
    if not isinstance(actual_model_output_channels, int): return np.array([]), np.array([]), np.array([])
    transposed_predictions = predictions_raw.transpose()
    boxes_tile_local_scaled, scores_tile_local, class_ids_tile_local = [], [], []
    for pred_data in transposed_predictions:
        if len(pred_data) != actual_model_output_channels: continue
        cx, cy, w, h = pred_data[:4]; confidence, class_id = 0.0, -1
        if actual_model_output_channels == 6: confidence = pred_data[4]; class_id = int(pred_data[5])
        elif actual_model_output_channels == 5: confidence = pred_data[4]; class_id = 0
        elif actual_model_output_channels > 4 :
            class_scores = pred_data[4:]
            if class_scores.size > 0: confidence = np.max(class_scores); class_id = np.argmax(class_scores)
            else: continue
        else: continue
        if confidence >= conf_threshold_tile:
            x1, y1, x2, y2 = (cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)
            boxes_tile_local_scaled.append([x1, y1, x2, y2]); scores_tile_local.append(confidence); class_ids_tile_local.append(class_id)
    if not boxes_tile_local_scaled: return np.array([]), np.array([]), np.array([])
    final_boxes_tile_original_coords = []
    tile_h_orig, tile_w_orig = tile_original_shape_hw
    for box in boxes_tile_local_scaled:
        b_x1,b_y1,b_x2,b_y2 = box[0]-preprocessing_pad_x,box[1]-preprocessing_pad_y,box[2]-preprocessing_pad_x,box[3]-preprocessing_pad_y
        if preprocessing_ratio == 0: continue
        ot_x1,ot_y1,ot_x2,ot_y2 = b_x1/preprocessing_ratio,b_y1/preprocessing_ratio,b_x2/preprocessing_ratio,b_y2/preprocessing_ratio
        ot_x1,ot_y1 = np.clip(ot_x1,0,tile_w_orig),np.clip(ot_y1,0,tile_h_orig); ot_x2,ot_y2 = np.clip(ot_x2,0,tile_w_orig),np.clip(ot_y2,0,tile_h_orig)
        final_boxes_tile_original_coords.append([ot_x1,ot_y1,ot_x2,ot_y2])
    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold): # 重命名以区分
    # ... (与您V2.3中的 non_max_suppression 内容一致) ...
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    if not isinstance(scores, np.ndarray) or scores.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]; areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i);_ = order.size;order = order[1:]
        if _ == 1: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order]);xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h;ovr=inter/(areas[i]+areas[order]-inter)
        inds=np.where(ovr<=iou_threshold)[0];order=order[inds]
    return keep

# --- 新的预处理和后处理函数 (适配新ONNX模型，基于test_onnx.py) ---
def preprocess_onnx_for_main(img_data, target_shape_hw):
    # ... (与 test_onnx.py 中的 preprocess_onnx 内容一致) ...
    img_height_orig, img_width_orig = img_data.shape[:2]; target_h, target_w = target_shape_hw
    ratio = min(target_w / img_width_orig, target_h / img_height_orig); new_w, new_h = int(img_width_orig * ratio), int(img_height_orig * ratio)
    resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8); pad_x = (target_w - new_w) // 2; pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    input_tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0; input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, ratio, pad_x, pad_y

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold,
                                   original_shape_hw, model_input_shape_hw,
                                   ratio_preproc, pad_x_preproc, pad_y_preproc,
                                   num_classes=1):
    # ... (与 test_onnx.py 中修正后的 postprocess_yolo_onnx 内容一致, 处理5属性输出) ...
    raw_output_tensor = np.squeeze(outputs_onnx[0]);
    if raw_output_tensor.ndim != 2: print(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []
    expected_attributes = 4 + 1
    for i_pred, pred_data in enumerate(predictions_to_iterate):
        if len(pred_data) != expected_attributes:
            if i_pred == 0: print(f"错误: Main 每个预测的属性数量 ({len(pred_data)}) 与期望值 ({expected_attributes}) 不符。")
            continue
        box_coords_raw = pred_data[:4]; final_confidence = float(pred_data[4]); class_id = 0
        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = box_coords_raw; x1,y1,x2,y2 = cx-w/2,cy-h/2,cx+w/2,cy+h/2
            boxes_candidate.append([x1,y1,x2,y2]); scores_candidate.append(final_confidence); class_ids_candidate.append(class_id)
    if not boxes_candidate: return []
    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold) # 使用全局NMS
    final_detections = []; orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx); box_model_coords = boxes_candidate[idx]; score = scores_candidate[idx]; class_id_val = class_ids_candidate[idx]
        box_no_pad_x1,box_no_pad_y1 = box_model_coords[0]-pad_x_preproc,box_model_coords[1]-pad_y_preproc
        box_no_pad_x2,box_no_pad_y2 = box_model_coords[2]-pad_x_preproc,box_model_coords[3]-pad_y_preproc
        if ratio_preproc == 0: continue
        orig_x1,orig_y1 = box_no_pad_x1/ratio_preproc,box_no_pad_y1/ratio_preproc; orig_x2,orig_y2 = box_no_pad_x2/ratio_preproc,box_no_pad_y2/ratio_preproc
        final_x1,final_y1 = np.clip(orig_x1,0,orig_w),np.clip(orig_y1,0,orig_h); final_x2,final_y2 = np.clip(orig_x2,0,orig_w),np.clip(orig_y2,0,orig_h)
        final_detections.append([int(final_x1),int(final_y1),int(final_x2),int(final_y2),score,class_id_val])
    return final_detections


# draw_detections 函数 (与V2.3.1一致，可以绘制ROI索引和OCR文本)
def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    # ... (与您V2.3.1中的 draw_detections 内容一致) ...
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out

# --- V2.4.7 主程序 (修正 class_names 定义 和 图像读取检查逻辑) ---
if __name__ == "__main__":
    t_start_overall = time.time()
    timing_profile['0_total_script_execution'] = 0
    # 使用脚本顶部的 VERSION 变量
    print(f"--- OBU 检测与识别工具 {VERSION} ---") # VERSION 来自全局配置

    clear_process_photo_directory(process_photo_dir) # process_photo_dir 来自全局配置
    print(f"'{process_photo_dir}' 文件夹已清理。")

    if not os.path.exists(ONNX_MODEL_PATH): print(f"错误: 模型未找到: {ONNX_MODEL_PATH}"); exit() # ONNX_MODEL_PATH 来自全局配置
    if not os.path.exists(IMAGE_NAME): print(f"错误: 图片未找到: {IMAGE_NAME}"); exit() # IMAGE_NAME 来自全局配置

    actual_max_area_threshold_px = None
    try:
        print(f"--- 初始化与模型加载 ---")
        t_start = time.time()
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        timing_profile['1_model_loading'] = time.time() - t_start
        print(f"ONNX模型加载完成 ({timing_profile['1_model_loading']:.2f} 秒)")

        input_cfg = session.get_inputs()[0]
        input_name = input_cfg.name
        input_shape_onnx = input_cfg.shape

        model_input_h_ref, model_input_w_ref = 640, 640
        if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int):
            model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]
        else:
            print(f"警告: 模型输入维度包含符号名称: {input_shape_onnx}. 使用参考尺寸 H={model_input_h_ref}, W={model_input_w_ref}")

        # class_names 从脚本顶部的 COCO_CLASSES (应为 ['Barcode']) 获取
        class_names = COCO_CLASSES
        print(f"模型输入: {input_name} {input_shape_onnx}. 类别设置为: {class_names}")

        t_start_img_read = time.time() # 重命名 t_start 以免混淆
        original_image = cv2.imread(IMAGE_NAME)
        timing_profile['2_image_reading'] = time.time() - t_start_img_read

        if original_image is None:
            print(f"错误: 无法读取图片: {IMAGE_NAME} (耗时: {timing_profile['2_image_reading']:.2f} 秒)")
            raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")

        orig_img_h, orig_img_w = original_image.shape[:2]
        print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['2_image_reading']:.2f} 秒读取)")

        # --- 后续代码与V2.4.6版本一致 ---
        # (MAX_DETECTION_AREA 计算, apply_tiling 判断, 切块或整图处理逻辑, 面积筛选, OCR循环, 绘图, 时间分析)
        # ... (我将直接粘贴V2.4.6中这部分的代码)

        if MAX_DETECTION_AREA is not None:
            if isinstance(MAX_DETECTION_AREA, float) and 0<MAX_DETECTION_AREA<=1.0: actual_max_area_threshold_px = (orig_img_h*orig_img_w)*MAX_DETECTION_AREA; print(f"MAX_DETECTION_AREA设为总面积{MAX_DETECTION_AREA*100:.1f}%, 阈值: {actual_max_area_threshold_px:.0f} px².")
            elif isinstance(MAX_DETECTION_AREA, (int,float)) and MAX_DETECTION_AREA > 1: actual_max_area_threshold_px = float(MAX_DETECTION_AREA); print(f"MAX_DETECTION_AREA设为绝对值: {actual_max_area_threshold_px:.0f} px².")
            else: actual_max_area_threshold_px = None
        apply_tiling = ENABLE_TILING; use_fixed_grid_tiling = False
        if apply_tiling and FIXED_TILING_GRID is not None and isinstance(FIXED_TILING_GRID, tuple) and len(FIXED_TILING_GRID) == 2 and all(isinstance(n, int) and n > 0 for n in FIXED_TILING_GRID): use_fixed_grid_tiling = True; print(f"切块处理: 固定网格 {FIXED_TILING_GRID}. 重叠率: {TILE_OVERLAP_RATIO*100}%")
        elif apply_tiling: apply_tiling = (orig_img_w > model_input_w_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING or orig_img_h > model_input_h_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING); print(f"切块处理: {'动态切块' if apply_tiling else '禁用 (尺寸未达动态切块阈值)'}。参考模型输入: {model_input_h_ref}x{model_input_w_ref}, 重叠率: {TILE_OVERLAP_RATIO*100}%")
        else: print(f"切块处理禁用 (全局配置)。")

        aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []
        if apply_tiling:
            t_start_tiling_loop = time.time(); total_inference_time,total_tile_preprocessing_time,total_tile_postprocessing_time,num_tiles_processed = 0,0,0,0
            if use_fixed_grid_tiling:
                num_cols, num_rows = FIXED_TILING_GRID; nominal_tile_w = orig_img_w/num_cols; nominal_tile_h = orig_img_h/num_rows; overlap_w_px = int(nominal_tile_w*TILE_OVERLAP_RATIO); overlap_h_px = int(nominal_tile_h*TILE_OVERLAP_RATIO)
                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        num_tiles_processed+=1; stride_x=nominal_tile_w if num_cols==1 else (nominal_tile_w-overlap_w_px); stride_y=nominal_tile_h if num_rows==1 else (nominal_tile_h-overlap_h_px); current_tile_x0=int(c_idx*stride_x); current_tile_y0=int(r_idx*stride_y); current_tile_x1=int(current_tile_x0+nominal_tile_w); current_tile_y1=int(current_tile_y0+nominal_tile_h); tile_crop_x0=max(0,current_tile_x0); tile_crop_y0=max(0,current_tile_y0); tile_crop_x1=min(orig_img_w,current_tile_x1); tile_crop_y1=min(orig_img_h,current_tile_y1); tile_data=original_image[tile_crop_y0:tile_crop_y1,tile_crop_x0:tile_crop_x1]; tile_h_curr,tile_w_curr=tile_data.shape[:2]
                        if tile_h_curr==0 or tile_w_curr==0 or tile_h_curr<model_input_h_ref*0.1 or tile_w_curr<model_input_w_ref*0.1: continue
                        t_s=time.time(); tensor,ratio,pad_x,pad_y = preprocess_image_data_for_tiling(tile_data,(model_input_h_ref,model_input_w_ref)); total_tile_preprocessing_time+=time.time()-t_s; t_s=time.time(); outputs=session.run(None,{input_name:tensor}); total_inference_time+=time.time()-t_s; t_s=time.time(); boxes_np,scores_np,c_ids_np = postprocess_detections_from_tile(outputs,(tile_h_curr,tile_w_curr),(model_input_h_ref,model_input_w_ref),ratio,pad_x,pad_y,CONFIDENCE_THRESHOLD,0); total_tile_postprocessing_time+=time.time()-t_s
                        if boxes_np.shape[0]>0:
                            for i_box in range(boxes_np.shape[0]): b=boxes_np[i_box]; aggregated_boxes.append([b[0]+tile_crop_x0,b[1]+tile_crop_y0,b[2]+tile_crop_x0,b[3]+tile_crop_y0]); aggregated_scores.append(scores_np[i_box]); aggregated_class_ids.append(c_ids_np[i_box])
            else: # Dynamic Tiling
                tile_w_dyn,tile_h_dyn=model_input_h_ref,model_input_h_ref; overlap_w_dyn,overlap_h_dyn=int(tile_w_dyn*TILE_OVERLAP_RATIO),int(tile_h_dyn*TILE_OVERLAP_RATIO); stride_w_dyn,stride_h_dyn=tile_w_dyn-overlap_w_dyn,tile_h_dyn-overlap_h_dyn
                for y0_dyn in range(0,orig_img_h,stride_h_dyn):
                    for x0_dyn in range(0,orig_img_w,stride_w_dyn):
                        num_tiles_processed+=1; x1_dyn,y1_dyn=min(x0_dyn+tile_w_dyn,orig_img_w),min(y0_dyn+tile_h_dyn,orig_img_h); tile_data=original_image[y0_dyn:y1_dyn,x0_dyn:x1_dyn]; tile_h_curr,tile_w_curr=tile_data.shape[:2]
                        if tile_h_curr==0 or tile_w_curr==0 or tile_h_curr<model_input_h_ref*0.1 or tile_w_curr<model_input_w_ref*0.1: continue
                        t_s=time.time(); tensor,ratio,pad_x,pad_y = preprocess_image_data_for_tiling(tile_data,(model_input_h_ref,model_input_w_ref)); total_tile_preprocessing_time+=time.time()-t_s; t_s=time.time(); outputs=session.run(None,{input_name:tensor}); total_inference_time+=time.time()-t_s; t_s=time.time(); boxes_np,scores_np,c_ids_np = postprocess_detections_from_tile(outputs,(tile_h_curr,tile_w_curr),(model_input_h_ref,model_input_w_ref),ratio,pad_x,pad_y,CONFIDENCE_THRESHOLD,0); total_tile_postprocessing_time+=time.time()-t_s
                        if boxes_np.shape[0]>0:
                            for i_box in range(boxes_np.shape[0]): b=boxes_np[i_box]; aggregated_boxes.append([b[0]+x0_dyn,b[1]+y0_dyn,b[2]+x0_dyn,b[3]+y0_dyn]); aggregated_scores.append(scores_np[i_box]); aggregated_class_ids.append(c_ids_np[i_box])
            timing_profile['3a_tiling_loop_total (incl_all_tiles_pre_inf_post)'] = time.time()-t_start_tiling_loop; timing_profile['3b_tiling_total_tile_preprocessing']=total_tile_preprocessing_time; timing_profile['3c_tiling_total_tile_inference']=total_inference_time; timing_profile['3d_tiling_total_tile_postprocessing']=total_tile_postprocessing_time; print(f"切块检测完成 (处理 {num_tiles_processed} 个图块)。")
            if len(aggregated_boxes)>0: t_start_nms=time.time(); keep_indices=non_max_suppression_global(np.array(aggregated_boxes),np.array(aggregated_scores),IOU_THRESHOLD); timing_profile['4a_global_nms']=time.time()-t_start_nms; aggregated_boxes=[aggregated_boxes[i] for i in keep_indices]; aggregated_scores=[aggregated_scores[i] for i in keep_indices]; aggregated_class_ids=[aggregated_class_ids[i] for i in keep_indices]; print(f"全局NMS完成 ({timing_profile['4a_global_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else: timing_profile['4a_global_nms']=0; aggregated_boxes,aggregated_scores,aggregated_class_ids=[],[],[]; print("切块后未检测到聚合对象，或NMS后无剩余对象。")
        else:
            print("--- 开始整图检测 (使用新ONNX模型适配的预处理和后处理) ---"); t_s = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s; t_s = time.time(); outputs_main = session.run(None, {input_name: input_tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s;
            detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(class_names)); timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s
            aggregated_boxes = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]; aggregated_scores = [d[4] for d in detections_result_list]; aggregated_class_ids = [d[5] for d in detections_result_list]; num_tiles_processed = 1; print(f"整图处理与后处理完成。找到了 {len(aggregated_boxes)} 个框。")

        if len(aggregated_boxes) > 0 and ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or actual_max_area_threshold_px is not None):
            t_start_area_filter=time.time(); filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids=[],[],[]; initial_box_count_before_area_filter=len(aggregated_boxes)
            for i,box in enumerate(aggregated_boxes):
                b_w,b_h=box[2]-box[0],box[3]-box[1]; area=b_w*b_h; valid_area=True
                if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA: valid_area=False
                if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid_area=False
                if valid_area: filtered_by_area_boxes.append(box); filtered_by_area_scores.append(aggregated_scores[i]); filtered_by_area_ids.append(aggregated_class_ids[i])
            aggregated_boxes,aggregated_scores,aggregated_class_ids=filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids; timing_profile['5_area_filtering']=time.time()-t_start_area_filter; print(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框).")
        else:
            timing_profile['5_area_filtering']=0
            if len(aggregated_boxes)>0: print("面积筛选未启用或不适用。")

        ocr_texts_for_drawing = []
        recognized_obu_data_list = []
        if len(aggregated_boxes) > 0:
            print(f"--- 最终检测到 {len(aggregated_boxes)} 个OBU的YOLO框, 开始精确裁剪数字区域并进行OCR ---"); t_ocr_total_start=time.time()
            for i,yolo_box_coords in enumerate(aggregated_boxes):
                class_id=int(aggregated_class_ids[i]); class_name_str=class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
                x1_yolo,y1_yolo,x2_yolo,y2_yolo=[int(c) for c in yolo_box_coords]; h_yolo=y2_yolo-y1_yolo; w_yolo=x2_yolo-x1_yolo
                y1_digit_ideal=y1_yolo; h_digit_ideal=int(h_yolo*DIGIT_ROI_HEIGHT_FACTOR); y2_digit_ideal=y1_yolo+h_digit_ideal # DIGIT_ROI_HEIGHT_FACTOR 来自全局
                w_digit_expanded=int(w_yolo*DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_yolo=x1_yolo+w_yolo/2; x1_digit_ideal=int(cx_yolo-w_digit_expanded/2); x2_digit_ideal=int(cx_yolo+w_digit_expanded/2) # DIGIT_ROI_WIDTH_EXPAND_FACTOR 来自全局
                y1_digit_clipped=max(0,y1_digit_ideal); y2_digit_clipped=min(orig_img_h,y2_digit_ideal); x1_digit_clipped=max(0,x1_digit_ideal); x2_digit_clipped=min(orig_img_w,x2_digit_ideal)
                current_box_info={"roi_index":i+1,"class":class_name_str,"bbox_yolo":[x1_yolo,y1_yolo,x2_yolo,y2_yolo],"bbox_digit_ocr":[x1_digit_clipped,y1_digit_clipped,x2_digit_clipped,y2_digit_clipped],"confidence_yolo":float(aggregated_scores[i]),"ocr_text_color_digit":"N/A","ocr_text_gray_digit":"N/A","ocr_text_binary_otsu_digit":"N/A","ocr_text_binary_adaptive_digit":"N/A","ocr_final_text":"N/A","ocr_confidence":0.0}
                print(f"  OBU ROI {current_box_info['roi_index']} (YOLO Box): 类别='{current_box_info['class']}', 边界框={current_box_info['bbox_yolo']}, YOLO置信度={current_box_info['confidence_yolo']:.2f}"); print(f"    Calculated Digit ROI for OCR: {current_box_info['bbox_digit_ocr']}")
                ocr_text_to_draw="N/A"
                if paddle_ocr_engine:
                    dx1,dy1,dx2,dy2=current_box_info['bbox_digit_ocr']
                    if dx2>dx1 and dy2>dy1:
                        digit_roi_color=original_image[dy1:dy2,dx1:dx2]; cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_color.png"),digit_roi_color)
                        digit_roi_gray=cv2.cvtColor(digit_roi_color,cv2.COLOR_BGR2GRAY); cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_gray.png"),digit_roi_gray)
                        _,digit_roi_binary_otsu=cv2.threshold(digit_roi_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_binary_otsu.png"),digit_roi_binary_otsu)
                        digit_roi_binary_adaptive=cv2.adaptiveThreshold(digit_roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2); cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_binary_adaptive.png"),digit_roi_binary_adaptive)
                        images_to_ocr_map={"color_digit":digit_roi_color,"gray_digit":digit_roi_gray,"binary_otsu_digit":digit_roi_binary_otsu,"binary_adaptive_digit":digit_roi_binary_adaptive}
                        ocr_inputs_to_try_cfg=[];
                        if OCR_PREPROCESS_TYPE_TO_USE=="all": ocr_inputs_to_try_cfg=list(images_to_ocr_map.keys()) # OCR_PREPROCESS_TYPE_TO_USE 来自全局
                        elif isinstance(OCR_PREPROCESS_TYPE_TO_USE,str) and OCR_PREPROCESS_TYPE_TO_USE in images_to_ocr_map: ocr_inputs_to_try_cfg=[OCR_PREPROCESS_TYPE_TO_USE]
                        elif isinstance(OCR_PREPROCESS_TYPE_TO_USE,list): ocr_inputs_to_try_cfg=[key for key in OCR_PREPROCESS_TYPE_TO_USE if key in images_to_ocr_map]
                        if not ocr_inputs_to_try_cfg: ocr_inputs_to_try_cfg=["binary_otsu_digit"]; print(f"警告: OCR_PREPROCESS_TYPE_TO_USE 配置无效，默认尝试 'binary_otsu_digit'")
                        best_ocr_text_for_this_roi="N/A"; highest_ocr_confidence=0.0
                        for preprocess_type_key in ocr_inputs_to_try_cfg:
                            image_to_process_single_channel=images_to_ocr_map[preprocess_type_key]
                            if image_to_process_single_channel.ndim==2: image_to_process_bgr=cv2.cvtColor(image_to_process_single_channel,cv2.COLOR_GRAY2BGR)
                            elif image_to_process_single_channel.ndim==3 and image_to_process_single_channel.shape[2]==3: image_to_process_bgr=image_to_process_single_channel
                            else: print(f"错误: 无法处理图像 {preprocess_type_key} 的通道数: {image_to_process_single_channel.shape}"); continue
                            print(f"    Attempting OCR with {preprocess_type_key} for Digit ROI {current_box_info['roi_index']}...")
                            try:
                                ocr_result_list=paddle_ocr_engine.ocr(image_to_process_bgr)
                                if ocr_result_list and isinstance(ocr_result_list,list) and len(ocr_result_list)>0:
                                    if ocr_result_list[0] is None: print(f"      PaddleOCR ({preprocess_type_key}): Result is None.")
                                    elif isinstance(ocr_result_list[0],dict):
                                        image_result_dict=ocr_result_list[0]; extracted_texts_list=image_result_dict.get('rec_texts'); rec_scores_list=image_result_dict.get('rec_scores')
                                        if isinstance(extracted_texts_list,list) and isinstance(rec_scores_list,list) and len(extracted_texts_list)==len(rec_scores_list) and extracted_texts_list:
                                            full_recognized_text="".join(extracted_texts_list).replace(" ",""); ocr_confidence=rec_scores_list[0] if rec_scores_list else 0.0; print(f"      PaddleOCR ({preprocess_type_key}) Result: '{full_recognized_text}' (Conf: {ocr_confidence:.2f})")
                                            current_box_info[f"ocr_text_{preprocess_type_key}"]=full_recognized_text
                                            if best_ocr_text_for_this_roi=="N/A" or ocr_confidence>highest_ocr_confidence: best_ocr_text_for_this_roi=full_recognized_text; highest_ocr_confidence=ocr_confidence
                                        elif extracted_texts_list is None or rec_scores_list is None: print(f"      PaddleOCR ({preprocess_type_key}): 'rec_texts' or 'rec_scores' key not found.")
                                        elif not extracted_texts_list: print(f"      PaddleOCR ({preprocess_type_key}): 'rec_texts' is empty.")
                                        else: print(f"      PaddleOCR ({preprocess_type_key}): Texts/Scores mismatch.")
                                    else: print(f"      PaddleOCR ({preprocess_type_key}): Result[0] not a dict.")
                                else: print(f"      PaddleOCR ({preprocess_type_key}): No valid result list.");
                            except Exception as ocr_e: print(f"      Error during PaddleOCR ({preprocess_type_key}): {ocr_e}")
                        current_box_info["ocr_final_text"]=best_ocr_text_for_this_roi; current_box_info["ocr_confidence"]=highest_ocr_confidence; ocr_text_to_draw=best_ocr_text_for_this_roi
                    else: print(f"    Skipping OCR for invalid/zero-size Digit ROI.")
                else: print("    PaddleOCR engine not available.")
                recognized_obu_data_list.append(current_box_info); ocr_texts_for_drawing.append(ocr_text_to_draw); print("-"*30)
            timing_profile['7_ocr_processing_total']=time.time()-t_ocr_total_start; print(f"--- 所有ROI的OCR处理完成 ({timing_profile['7_ocr_processing_total']:.3f} 秒) ---")
            print("\n--- 初步识别结果列表 (未映射到矩阵) ---");
            for obu_data in recognized_obu_data_list:
                if obu_data["ocr_final_text"]!="N/A": print(f"ROI {obu_data['roi_index']} (YOLO BBox: {obu_data['bbox_yolo']}): Digit OCR Text = {obu_data['ocr_final_text']}")
            print("---------------------------------------\n")
            t_start_drawing=time.time(); output_img_to_draw_on=original_image.copy(); output_img_to_draw_on=draw_detections(output_img_to_draw_on,np.array(aggregated_boxes),np.array(aggregated_scores),np.array(aggregated_class_ids),class_names,ocr_texts=ocr_texts_for_drawing,roi_indices=[item["roi_index"] for item in recognized_obu_data_list]); timing_profile['8_drawing_results_final']=time.time()-t_start_drawing; output_fn_base=os.path.splitext(os.path.basename(IMAGE_NAME))[0]; output_fn=f"output_{output_fn_base}_{VERSION}_ocr.png"; cv2.imwrite(output_fn,output_img_to_draw_on); print(f"最终结果图已保存到: {output_fn} ({timing_profile['8_drawing_results_final']:.2f} 秒用于绘图)")
        else:
            print("最终未检测到任何OBU ROI，无法进行OCR。"); timing_profile['7_ocr_processing_total']=0; timing_profile['8_drawing_results_final']=0
    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"发生错误: {e}"); traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 ({VERSION}) ---"); sorted_timing_profile = {k: timing_profile[k] for k in sorted(timing_profile.keys())}
        for stage, duration in sorted_timing_profile.items(): print(f"  {stage}: {duration:.3f} 秒")
        print(f"------------------------------")
