# layout_and_state_manager.py
import cv2
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import os
import logging

from config import (
    LAYOUT_EXPECTED_TOTAL_ROWS,
    LAYOUT_REGULAR_COLS_COUNT,
    LAYOUT_SPECIAL_ROW_COLS_COUNT,
    LAYOUT_MIN_CORE_ANCHORS_FOR_STATS,
    LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD,
    LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD,
    VALID_OBU_CODES
)

def _analyze_layout_by_xy_clustering(
    yolo_detections: List[Dict[str, Any]],
    y_pixel_threshold: float, # 直接使用配置的固定像素阈值
    x_pixel_threshold: float, # 直接使用配置的固定像素阈值
    logger: Any,
    session_id_for_log: str = "N/A"
) -> List[Dict[str, Any]]:
    log_prefix = f"会话 {session_id_for_log} (_analyze_xy_clustering V3.0_P0_FixedThresh):" # 版本更新
    logger.info(f"{log_prefix} 开始通过XY独立聚类分析布局，输入锚点数: {len(yolo_detections)}")

    if not yolo_detections:
        logger.warning(f"{log_prefix} 无YOLO检测结果，无法进行聚类。")
        return []

    detections_with_rc = [det.copy() for det in yolo_detections]

    # --- Y轴聚类 (确定 frame_r) ---
    y_data_for_sort = [{'original_list_idx': i, 'cy': det['cy']} for i, det in enumerate(detections_with_rc)]
    y_data_for_sort.sort(key=lambda item: item['cy'])

    logger.info(f"{log_prefix} Y轴分组固定像素阈值: {y_pixel_threshold:.1f}")

    y_groups_indices: List[List[int]] = []
    if y_data_for_sort:
        current_y_group_indices = [y_data_for_sort[0]['original_list_idx']]
        current_group_ref_y = y_data_for_sort[0]['cy']

        for i in range(1, len(y_data_for_sort)):
            item_curr = y_data_for_sort[i]
            # 使用组内所有元素的平均Y值与新元素比较，更鲁棒
            current_group_avg_cy = np.mean([detections_with_rc[idx]['cy'] for idx in current_y_group_indices])
            if abs(item_curr['cy'] - current_group_avg_cy) < y_pixel_threshold:
                current_y_group_indices.append(item_curr['original_list_idx'])
            else:
                y_groups_indices.append(list(current_y_group_indices))
                current_y_group_indices = [item_curr['original_list_idx']]
                current_group_ref_y = item_curr['cy'] # 新组的参考Y更新为第一个元素
        y_groups_indices.append(list(current_y_group_indices))

    for r_idx, group_indices in enumerate(y_groups_indices):
        avg_y_of_group = np.mean([detections_with_rc[original_idx]['cy'] for original_idx in group_indices])
        logger.debug(f"{log_prefix} Y组 {r_idx} (avg_cy={avg_y_of_group:.0f}): 包含detections_with_rc索引 {group_indices}")
        for original_idx in group_indices:
            detections_with_rc[original_idx]['frame_r'] = r_idx
            detections_with_rc[original_idx]['frame_r_avg_cy'] = avg_y_of_group

    logger.info(f"{log_prefix} Y轴聚类完成，得到 {len(y_groups_indices)} 个单帧逻辑行组。")

    # --- X轴聚类 (确定 frame_c) ---
    x_data_for_sort = [{'original_list_idx': i, 'cx': det['cx']} for i, det in enumerate(detections_with_rc)]
    x_data_for_sort.sort(key=lambda item: item['cx'])

    logger.info(f"{log_prefix} X轴分组固定像素阈值: {x_pixel_threshold:.1f}")

    x_groups_indices: List[List[int]] = []
    if x_data_for_sort:
        current_x_group_indices = [x_data_for_sort[0]['original_list_idx']]
        current_group_ref_x = x_data_for_sort[0]['cx']
        for i in range(1, len(x_data_for_sort)):
            item_curr = x_data_for_sort[i]
            current_group_avg_cx = np.mean([detections_with_rc[idx]['cx'] for idx in current_x_group_indices])
            if abs(item_curr['cx'] - current_group_avg_cx) < x_pixel_threshold:
                current_x_group_indices.append(item_curr['original_list_idx'])
            else:
                x_groups_indices.append(list(current_x_group_indices))
                current_x_group_indices = [item_curr['original_list_idx']]
                current_group_ref_x = item_curr['cx']
        x_groups_indices.append(list(current_x_group_indices))

    for c_idx, group_indices in enumerate(x_groups_indices):
        avg_x_of_group = np.mean([detections_with_rc[original_idx]['cx'] for original_idx in group_indices])
        logger.debug(f"{log_prefix} X组 {c_idx} (avg_cx={avg_x_of_group:.0f}): 包含detections_with_rc索引 {group_indices}")
        for original_idx in group_indices:
            detections_with_rc[original_idx]['frame_c'] = c_idx
            detections_with_rc[original_idx]['frame_c_avg_cx'] = avg_x_of_group

    logger.info(f"{log_prefix} X轴聚类完成，得到 {len(x_groups_indices)} 个单帧逻辑列组。")

    for i, det in enumerate(detections_with_rc):
        if 'frame_r' not in det:
            det['frame_r'] = -1
        if 'frame_c' not in det:
            det['frame_c'] = -1

    return detections_with_rc

def _create_frame_layout_stats_and_identify_special_row(
    yolo_detections_with_frame_rc: List[Dict[str, Any]],
    session_config: Dict[str, Any],
    logger: Any,
    session_id_for_log: str = "N/A"
) -> Dict[str, Any]:
    log_prefix = f"会话 {session_id_for_log} (_create_frame_stats_special_row V3.3_P0):"
    logger.info(f"{log_prefix} 基于XY聚类结果统计参数并识别特殊行。输入锚点数: {len(yolo_detections_with_frame_rc)}")

    stats = {
        "median_obu_w_frame": 100.0, "median_obu_h_frame": 40.0,
        "avg_physical_row_height_frame": 60.0,
        "row_y_means_from_clustering": {},
        "col_x_means_from_clustering": {},
        "is_special_row_identified": False,
        "identified_special_row_frame_r": -1,
        "special_row_at_logical_top": False
    }

    if not yolo_detections_with_frame_rc or len(yolo_detections_with_frame_rc) < LAYOUT_MIN_CORE_ANCHORS_FOR_STATS:
        logger.warning(f"{log_prefix} 带frame_r/c的锚点数量不足。返回默认统计。")
        return stats

    all_ws = [a['w'] for a in yolo_detections_with_frame_rc if a.get('w', 0) > 0]
    all_hs = [a['h'] for a in yolo_detections_with_frame_rc if a.get('h', 0) > 0]
    if all_ws: stats["median_obu_w_frame"] = np.median(all_ws)
    if all_hs: stats["median_obu_h_frame"] = np.median(all_hs)
    logger.info(f"{log_prefix} 当前帧OBU中位数尺寸: W={stats['median_obu_w_frame']:.1f}, H={stats['median_obu_h_frame']:.1f}")

    row_cy_collections = defaultdict(list)
    col_cx_collections = defaultdict(list)
    for det in yolo_detections_with_frame_rc:
        fr, fc = det.get('frame_r', -1), det.get('frame_c', -1)
        if fr != -1: row_cy_collections[fr].append(det['cy'])
        if fc != -1: col_cx_collections[fc].append(det['cx'])

    for r, cys in row_cy_collections.items():
        if cys: stats["row_y_means_from_clustering"][r] = np.mean(cys)
    for c, cxs in col_cx_collections.items():
        if cxs: stats["col_x_means_from_clustering"][c] = np.mean(cxs)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{log_prefix}   行Y均值: { {r:int(y) for r,y in stats['row_y_means_from_clustering'].items()} }")
        logger.debug(f"{log_prefix}   列X均值: { {c:int(x) for c,x in stats['col_x_means_from_clustering'].items()} }")

    sorted_unique_row_y_means = sorted(list(set(stats["row_y_means_from_clustering"].values())))
    if len(sorted_unique_row_y_means) >= 2:
        row_gaps_y = np.abs(np.diff(sorted_unique_row_y_means))
        if row_gaps_y.size > 0:
            median_row_gap = np.median(row_gaps_y)
            if median_row_gap > stats["median_obu_h_frame"] * 0.3:
                stats["avg_physical_row_height_frame"] = median_row_gap
    logger.info(f"{log_prefix} 当前帧平均物理行高估算: {stats['avg_physical_row_height_frame']:.1f}像素")

    # --- 特殊行识别逻辑 (P0阶段简化版，主要针对底部特殊行) ---
    expected_special_cols = session_config.get("special_row_cols_count", LAYOUT_SPECIAL_ROW_COLS_COUNT)
    expected_reg_cols = session_config.get("regular_cols_count", LAYOUT_REGULAR_COLS_COUNT)

    rows_summary = defaultdict(lambda: {"col_indices": set(), "avg_y": 0.0, "count":0, "frame_cs": []})
    for det in yolo_detections_with_frame_rc:
        fr, fc = det.get('frame_r', -1), det.get('frame_c', -1)
        if fr != -1 and fc != -1:
            rows_summary[fr]["col_indices"].add(fc)
            rows_summary[fr]["avg_y"] += det['cy']
            rows_summary[fr]["count"] += 1
            rows_summary[fr]["frame_cs"].append(fc) # 记录该行所有的frame_c

    candidate_special_rows_bottom = [] # (frame_r, num_cols_in_row, avg_y_of_row)
    if rows_summary:
        # 找到物理最底部的行 (frame_r 值最大，或者 avg_y 最大)
        # 我们用 avg_y 来判断物理上的底部
        sorted_rows_by_phys_y = sorted(rows_summary.items(), key=lambda item: item[1]["avg_y"]/item[1]["count"] if item[1]["count"]>0 else 0, reverse=True)

        if sorted_rows_by_phys_y:
            bottom_most_fr, bottom_row_data = sorted_rows_by_phys_y[0]
            num_cols_in_bottom_row = len(bottom_row_data["col_indices"])

            if num_cols_in_bottom_row == expected_special_cols:
                # 检查是否大致居中 (对于4列转2列的情况)
                is_centered = False
                if expected_reg_cols == 4 and expected_special_cols == 2:
                    # 期望特殊行的 frame_c 是中间的两个，例如1和2 (如果X聚类准确的话)
                    # 或者说，这两个frame_c的平均值应该接近总列数的一半减0.5
                    # ( (expected_reg_cols/2 -1) + (expected_reg_cols/2) ) / 2 = expected_reg_cols/2 - 0.5
                    # 例如4列时，期望是 (1+2)/2 = 1.5. 总列数是0,1,2,3.
                    # 这是一个比较强的假设，依赖X聚类的稳定性
                    # 简化：只要列数对，就认为是候选
                    is_centered = True # 暂时简化
                elif num_cols_in_bottom_row == expected_special_cols : # 其他列数配置
                    is_centered = True

                if is_centered:
                    # 检查其上一行是否像常规行
                    is_prev_row_regular = True # 默认可信
                    if len(sorted_rows_by_phys_y) > 1:
                        prev_row_fr, prev_row_data = sorted_rows_by_phys_y[1] # 物理上的上一行
                        num_cols_in_prev_row = len(prev_row_data["col_indices"])
                        if not (expected_reg_cols -1 <= num_cols_in_prev_row <= expected_reg_cols +1) :
                             is_prev_row_regular = False

                    if is_prev_row_regular:
                        stats["is_special_row_identified"] = True
                        stats["identified_special_row_frame_r"] = bottom_most_fr
                        stats["special_row_at_logical_top"] = False # 因为我们找的是底部的
                        logger.info(f"{log_prefix} 识别到底部特殊行: frame_r={bottom_most_fr}")

    if not stats["is_special_row_identified"]:
        logger.warning(f"{log_prefix} 未能明确识别出特殊行（特别是底部）。")

    logger.info(f"{log_prefix} 帧布局统计与特殊行识别完成。")
    return stats

class LayoutStateManager:
    def __init__(self, config_params: Dict[str, Any], logger: Any):
        self.logger = logger
        self.logger.warning("LayoutStateManager 已被加载，但其核心功能在当前版本中已废弃。")

    def analyze_frame_layout_and_get_params(self, *args, **kwargs):
        self.logger.debug("调用了已废弃的 analyze_frame_layout_and_get_params 方法。")
        return None, []

    def determine_y_axis_anchor(self, *args, **kwargs):
        self.logger.debug("调用了已废弃的 determine_y_axis_anchor 方法。")
        return {}, []

    def draw_stable_layout_on_image(self,
                                     yolo_detections_with_frame_rc: List[Dict[str, Any]],
                                     image_wh: Tuple[int, int],
                                     session_id: str,
                                     frame_num: int,
                                     y_anchor_info: Optional[Dict[str,Any]],
                                     current_frame_layout_stats: Optional[Dict[str, Any]]
                                     ):
        log_prefix = f"会话 {session_id} (draw_layout_from_cluster F{frame_num} V3.2_P0_Corrected):"
        self.logger.critical(f"{log_prefix} CRITICAL_LOG: 进入绘制【单帧逻辑投射图】函数。")

        img_w, img_h = image_wh
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        self.logger.info(f"{log_prefix} 画布尺寸: W={img_w}, H={img_h}")

        if not yolo_detections_with_frame_rc:
            self.logger.warning(f"{log_prefix} 无带frame_r/c的YOLO检测结果，无法绘制投射图。")
            # 即使没有检测结果，也尝试保存一张空白图，以便确认函数被调用
            # (或者根据需要决定是否保存空图)
            # return

        avg_w_draw = 100.0; avg_h_draw = 40.0
        if current_frame_layout_stats:
            avg_w_draw = current_frame_layout_stats.get("median_obu_w_frame", avg_w_draw)
            avg_h_draw = current_frame_layout_stats.get("median_obu_h_frame", avg_h_draw)

        delta_r_draw = y_anchor_info.get("delta_r", 0) if y_anchor_info else 0
        font_scale = 0.7; font_thickness = 1

        # 绘制由 _analyze_layout_by_xy_clustering 直接找到的坑位
        # 并在其物理中心绘制标记，标注其全局逻辑坐标
        # 我们需要统计每个 (frame_r, frame_c) 的平均物理中心
        logical_to_physical_centers = defaultdict(lambda: {"cx_sum": 0, "cy_sum": 0, "count": 0})
        for det in yolo_detections_with_frame_rc: # 确保这里迭代的是带frame_r/c的列表
            fr = det.get('frame_r', -1)
            fc = det.get('frame_c', -1)
            if fr != -1 and fc != -1:
                # 使用全局逻辑坐标作为key来聚合中心点
                key = (fr + delta_r_draw, fc)
                logical_to_physical_centers[key]["cx_sum"] += det['cx']
                logical_to_physical_centers[key]["cy_sum"] += det['cy']
                logical_to_physical_centers[key]["count"] += 1

        for (global_r_draw, global_c_draw), data in logical_to_physical_centers.items():
            if data["count"] == 0: continue
            center_x = int(data["cx_sum"] / data["count"])
            center_y = int(data["cy_sum"] / data["count"])

            is_drawn_as_special = False
            _expected_total_rows = self.config_params.get("LAYOUT_EXPECTED_TOTAL_ROWS", 13)
            _reg_cols = self.config_params.get("LAYOUT_REGULAR_COLS_COUNT", 4)
            _spec_cols = self.config_params.get("LAYOUT_SPECIAL_ROW_COLS_COUNT", 2)

            # 假设特殊行在底部 (这个判断需要更全局的 special_row_at_logical_top 信息)
            # 我们从 current_frame_layout_stats 中获取（如果它被正确设置了）
            is_special_row_top_from_stats = False # 默认不在顶部
            if current_frame_layout_stats:
                 is_special_row_top_from_stats = current_frame_layout_stats.get("special_row_at_logical_top", False)

            expected_special_global_r = 0 if is_special_row_top_from_stats else _expected_total_rows - 1

            if global_r_draw == expected_special_global_r:
                if _reg_cols == 4 and _spec_cols == 2: # 针对4列转2列的特殊行
                    # 特殊行的有效全局逻辑列通常是1和2
                    if global_c_draw == 1 or global_c_draw == 2:
                        is_drawn_as_special = True
                elif global_c_draw < _spec_cols: # 其他情况，只要在特殊行列数内
                     is_drawn_as_special = True

            slot_color = (0, 100, 150) if is_drawn_as_special else (0, 150, 0)
            text_color = (255, 255, 255)
            text_to_draw = f"({global_r_draw},{global_c_draw})"

            half_w, half_h = int(avg_w_draw / 2), int(avg_h_draw / 2)
            pt1 = (max(0, center_x - half_w), max(0, center_y - half_h))
            pt2 = (min(img_w -1, center_x + half_w), min(img_h -1, center_y + half_h))

            if pt1[0] >= pt2[0] or pt1[1] >= pt2[1]: continue

            cv2.rectangle(canvas, pt1, pt2, slot_color, -1)
            cv2.rectangle(canvas, pt1, pt2, (200,200,200), 1)

            (tw, th), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.putText(canvas, text_to_draw, (center_x - tw // 2, center_y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

        output_dir = os.path.join(self.config_params.get("PROCESS_PHOTO_DIR", "process_photo"))
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

        map_filename = f"frame_layout_projection_s{session_id[:8]}_f{frame_num}.png"
        output_path = os.path.join(output_dir, map_filename)
        try:
            save_success = cv2.imwrite(output_path, canvas)
            if save_success: self.logger.critical(f"{log_prefix} CRITICAL_LOG: 【单帧逻辑投射图】已成功保存到: {output_path}")
            else: self.logger.critical(f"{log_prefix} CRITICAL_LOG: cv2.imwrite未能保存单帧逻辑投射图到 {output_path} (返回False)")
        except Exception as e_save: self.logger.critical(f"{log_prefix} CRITICAL_LOG: 保存单帧逻辑投射图失败 {output_path}: {e_save}", exc_info=True)

    def update_session_state_with_reference_logic(self, *args, **kwargs):
        self.logger.debug("调用了已废弃的 update_session_state_with_reference_logic 方法。")
        return [], {}, []

# END OF LayoutStateManager CLASS