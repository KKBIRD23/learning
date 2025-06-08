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
        self.config_params = config_params
        self.logger = logger
        # 从传入的config_params字典中获取配置，如果获取不到则使用全局config模块的默认值
        self.expected_total_rows = self.config_params.get(
            "LAYOUT_EXPECTED_TOTAL_ROWS",
            LAYOUT_EXPECTED_TOTAL_ROWS # Fallback to global config import
        )
        self.regular_cols_count = self.config_params.get(
            "LAYOUT_REGULAR_COLS_COUNT",
            LAYOUT_REGULAR_COLS_COUNT
        )
        self.special_row_cols_count = self.config_params.get(
            "LAYOUT_SPECIAL_ROW_COLS_COUNT",
            LAYOUT_SPECIAL_ROW_COLS_COUNT
        )
        self.logger.info(f"LayoutStateManager initialized with: "
                         f"ExpectedRows={self.expected_total_rows}, "
                         f"RegularCols={self.regular_cols_count}, "
                         f"SpecialCols={self.special_row_cols_count}")

    def analyze_frame_layout_and_get_params(self,
                                           yolo_detections: List[Dict[str, Any]],
                                           image_wh: Tuple[int, int],
                                           session_config_from_session: Dict[str, Any], # 这个参数现在主要用来传递期望行列数等
                                           session_id: str
                                           ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        log_prefix = f"会话 {session_id} (analyze_frame_layout_entry V3.2_P0_Corrected):"
        self.logger.info(f"{log_prefix} 开始分析帧布局并获取参数...")

        if not yolo_detections:
            self.logger.warning(f"{log_prefix} 无YOLO检测结果，无法分析布局。")
            return None, []

        # 从 self.config_params (初始化时从全局config传入) 获取固定像素阈值
        y_pixel_thresh = self.config_params.get("LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD", 50.0)
        x_pixel_thresh = self.config_params.get("LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD", 400.0)

        detections_with_frame_rc = _analyze_layout_by_xy_clustering(
            yolo_detections,
            y_pixel_thresh,
            x_pixel_thresh,
            self.logger, # 传递自身的logger实例
            session_id
        )
        if not detections_with_frame_rc:
            # 检查是否所有检测框都未能分配frame_r或frame_c
            all_failed_clustering = True
            for det_check in detections_with_frame_rc: # detections_with_frame_rc 可能是空列表
                if det_check.get('frame_r', -1) != -1 and det_check.get('frame_c', -1) != -1:
                    all_failed_clustering = False
                    break
            if all_failed_clustering and yolo_detections: # 如果有输入但聚类全失败
                 self.logger.error(f"{log_prefix} XY独立聚类未能为任何锚点分配有效的单帧逻辑坐标。")
            # 返回原始detections，让后续步骤知道聚类失败（通过检查有无frame_r/c）
            # 或者返回一个空列表表示没有带rc的结果，这取决于后续函数的期望
            # 为了与之前的返回类型一致，如果聚类有部分成功，还是返回detections_with_frame_rc
            # 如果聚类函数本身返回空列表（例如输入为空），则这里也返回空
            if not detections_with_frame_rc and yolo_detections: # 输入非空，但聚类输出为空
                return None, [] # 表示聚类过程有问题，无法进行下一步统计

        # 使用 session_config_from_session (它应该包含了正确的期望行列数等)
        current_frame_layout_stats = _create_frame_layout_stats_and_identify_special_row(
            detections_with_frame_rc,
            session_config_from_session, # 传递从会话中获取的布局配置
            self.logger,
            session_id
        )

        return current_frame_layout_stats, detections_with_frame_rc

    def determine_y_axis_anchor(self,
                                current_frame_verified_obus_with_rc: List[Dict[str, Any]], # 已包含frame_r
                                obu_evidence_pool: Dict[str, Any],
                                current_frame_layout_stats: Optional[Dict[str, Any]],
                                session_id: str,
                                current_frame_num: int,
                                session_config: Dict[str, Any]
                                ) -> Tuple[Optional[Dict[str, Any]], List[float]]:
        log_prefix = f"会话 {session_id} (determine_y_axis_anchor F{current_frame_num} V3.2_P0_Corrected):"

        y_anchor_info = {
            "delta_r": 0,
            "is_anchored": False,
            "is_first_frame_anchor_failed": False,
            "anchor_type": "none"
        }

        _expected_total_rows_drawing = session_config.get("expected_total_rows", self.expected_total_rows)
        estimated_row_y_for_drawing = [0.0] * _expected_total_rows_drawing
        avg_row_h_for_drawing = 100.0
        base_y_for_drawing = 100.0

        if current_frame_layout_stats:
            avg_row_h_for_drawing = current_frame_layout_stats.get("avg_physical_row_height_frame", avg_row_h_for_drawing)
            row_y_means = current_frame_layout_stats.get("row_y_means_from_clustering", {})
            if row_y_means: # 如果有聚类的行Y均值
                # 找到最小的 frame_r (聚类出的最顶行) 及其对应的平均Y
                min_fr_for_base_y = min(row_y_means.keys()) if row_y_means else 0
                base_y_for_drawing_at_min_fr = row_y_means.get(min_fr_for_base_y, avg_row_h_for_drawing * 0.5)
                # 将 base_y_for_drawing 校准为 frame_r=0 时的估算物理Y
                base_y_for_drawing = base_y_for_drawing_at_min_fr - min_fr_for_base_y * avg_row_h_for_drawing
            else: # 如果没有聚类行信息，使用一个非常粗略的默认值
                 base_y_for_drawing = avg_row_h_for_drawing * 0.5


        for i in range(_expected_total_rows_drawing):
            estimated_row_y_for_drawing[i] = base_y_for_drawing + i * avg_row_h_for_drawing

        # --- 第一帧特殊处理：基于特殊行强制锚定 ---
        if current_frame_num == 1:
            self.logger.info(f"{log_prefix} 首帧处理，尝试基于特殊行锚定。")
            if current_frame_layout_stats and current_frame_layout_stats.get("is_special_row_identified"):
                special_row_fr = current_frame_layout_stats["identified_special_row_frame_r"]
                # 我们强制特殊行在底部
                if not current_frame_layout_stats.get("special_row_at_logical_top", True): # 确保学习到的是底部特殊行
                    expected_global_special_r = session_config.get("expected_total_rows", self.expected_total_rows) - 1
                    delta_r = expected_global_special_r - special_row_fr
                    y_anchor_info.update({
                        "delta_r": delta_r,
                        "is_anchored": True,
                        "anchor_type": "first_frame_special_row_bottom"
                    })
                    self.logger.info(f"{log_prefix} 首帧：通过识别到的底部特殊行 (frame_r={special_row_fr}) "
                                     f"锚定成功。计算得到 delta_r = {delta_r}。")
                else:
                    y_anchor_info["is_first_frame_anchor_failed"] = True
                    self.logger.warning(f"{log_prefix} 首帧：识别到特殊行，但其位置不符合底部锚定要求（或逻辑未完全实现）。标记为锚定失败。")
            else:
                y_anchor_info["is_first_frame_anchor_failed"] = True
                self.logger.error(f"{log_prefix} 首帧：未能识别到特殊行！根据行政规定，此为错误。delta_r保持0。")

            if y_anchor_info["is_anchored"]:
                # 更新绘图用的Y估算 (基于第一帧的delta_r)
                # 假设 frame_r=0 对应的物理Y是 base_y_for_drawing
                # global_r = frame_r + delta_r
                # 当 global_r = 0 时, frame_r = -delta_r
                # y_for_global_0 = base_y_for_drawing + (-delta_r) * avg_row_h_for_drawing
                y_for_global_0 = base_y_for_drawing - y_anchor_info["delta_r"] * avg_row_h_for_drawing
                for i_global_logical in range(_expected_total_rows_drawing):
                    estimated_row_y_for_drawing[i_global_logical] = y_for_global_0 + i_global_logical * avg_row_h_for_drawing
                self.logger.info(f"  首帧锚定后，用于绘图的行Y估算: {[int(y) for y in estimated_row_y_for_drawing]}")

            return y_anchor_info, estimated_row_y_for_drawing

        # --- 后续帧的Y轴锚定逻辑 ---
        historical_overlaps_in_current_frame = []
        for obu_cur in current_frame_verified_obus_with_rc:
            if obu_cur["text"] in obu_evidence_pool:
                hist_entry = obu_evidence_pool[obu_cur["text"]]
                if hist_entry.get("logical_coord") is not None and obu_cur.get("frame_r", -1) != -1:
                    historical_overlaps_in_current_frame.append({
                        "text": obu_cur["text"],
                        "current_physical_anchor": obu_cur["physical_anchor"],
                        "historical_logical_coord": hist_entry["logical_coord"],
                        "current_frame_r": obu_cur["frame_r"]
                    })
        self.logger.info(f"{log_prefix} 在当前帧找到 {len(historical_overlaps_in_current_frame)} 个有效的历史重叠OBU用于Y轴锚定。")
        if not historical_overlaps_in_current_frame:
            self.logger.error(f"{log_prefix} 检测到漏帧！当前帧与历史无有效重叠OBU。delta_r将为0。")
            y_anchor_info["is_skipped_due_to_no_overlap"] = True
            return y_anchor_info, estimated_row_y_for_drawing
        min_y_historical_obu_for_anchor = min(
            historical_overlaps_in_current_frame, key=lambda o: o["current_physical_anchor"]["cy"]
        )
        y_ref_current_phys = min_y_historical_obu_for_anchor["current_physical_anchor"]["cy"]
        l_ref_global_hist = min_y_historical_obu_for_anchor["historical_logical_coord"][0]
        frame_r_anchor_curr = min_y_historical_obu_for_anchor["current_frame_r"]
        delta_r = l_ref_global_hist - frame_r_anchor_curr
        y_anchor_info.update({
            "ref_obu_text": min_y_historical_obu_for_anchor['text'],
            "ref_global_logical_row": l_ref_global_hist,
            "ref_frame_logical_row": frame_r_anchor_curr,
            "ref_physical_y_current": y_ref_current_phys,
            "delta_r": delta_r, "is_anchored": True, "anchor_type": "historical_obu"
        })
        self.logger.info(f"{log_prefix} Y轴锚定成功。参照OBU='{y_anchor_info['ref_obu_text']}', "
                         f"历史全局行L={l_ref_global_hist}, 当前帧聚类行R={frame_r_anchor_curr}, delta_r={delta_r}")

        # 更新用于绘图的 estimated_row_y_for_drawing
        # 假设 frame_r_anchor_curr 对应的物理Y是 y_ref_current_phys
        # 那么 frame_r = 0 对应的物理Y大约是 y_ref_current_phys - frame_r_anchor_curr * avg_row_h_for_drawing
        # (这个 base_y_for_drawing 是基于当前帧聚类出的 frame_r=0 的行的物理Y)
        # 我们需要的是全局行0的物理Y
        # global_r = frame_r + delta_r => frame_r = global_r - delta_r
        # 当 global_r = 0 时, frame_r_for_global_0 = -delta_r
        # y_for_global_0 = base_y_for_drawing_of_frame_r0 - (-delta_r) * avg_row_h_for_drawing (如果delta_r是负的)
        # 这里的 base_y_for_drawing 是指当前帧 frame_r=0 的物理Y
        # 所以，全局行0的物理Y = (当前帧frame_r=0的物理Y) - delta_r * avg_row_h_for_drawing
        y_for_global_0 = base_y_for_drawing - delta_r * avg_row_h_for_drawing
        for i_global_logical in range(_expected_total_rows_drawing):
            estimated_row_y_for_drawing[i_global_logical] = y_for_global_0 + i_global_logical * avg_row_h_for_drawing

        self.logger.info(f"  更新后（理论上的）动态行Y估算(用于绘图): {[int(y) for y in estimated_row_y_for_drawing]}")

        return y_anchor_info, estimated_row_y_for_drawing

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

    def update_session_state_with_reference_logic(
        self,
        session_data: Dict[str, Any],
        current_frame_yolo_detections_with_rc: List[Dict[str, Any]],
        current_frame_ocr_results: List[Dict[str, Any]],
        y_anchor_info: Optional[Dict[str, Any]],
        session_id: str,
        current_frame_num: int,
        session_config: Dict[str, Any]
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], str], List[Dict[str, Any]]]:
        log_prefix = f"会话 {session_id} (update_state_ref_logic V3.3_P0_Corrected F{current_frame_num}):"
        self.logger.info(f"{log_prefix} 开始核心状态更新 (基于XY聚类和Y轴锚定)...")

        obu_evidence_pool = session_data["obu_evidence_pool"]
        logical_matrix = session_data["logical_matrix"]
        recognized_texts_map = session_data["recognized_texts_map"]
        warnings = []

        _expected_total_rows = session_config.get("expected_total_rows", self.expected_total_rows)
        _regular_cols = session_config.get("regular_cols_count", self.regular_cols_count)
        _special_cols = session_config.get("special_row_cols_count", self.special_row_cols_count)

        delta_r = y_anchor_info.get("delta_r", 0) if y_anchor_info else 0
        self.logger.info(f"{log_prefix} 使用的行偏移 delta_r: {delta_r}")

        map_ocr_results_by_yolo_idx = {ocr.get("original_index"): ocr for ocr in current_frame_ocr_results if ocr}

        for det_with_rc in current_frame_yolo_detections_with_rc:
            original_yolo_idx = det_with_rc["original_index"]
            ocr_item = map_ocr_results_by_yolo_idx.get(original_yolo_idx)
            if not ocr_item: continue
            ocr_text = ocr_item.get("ocr_final_text", "")
            if ocr_text in VALID_OBU_CODES:
                frame_r, frame_c = det_with_rc.get('frame_r', -1), det_with_rc.get('frame_c', -1)
                if frame_r == -1 or frame_c == -1:
                    self.logger.warning(f"{log_prefix} OBU '{ocr_text}' (YOLO原始索引 {original_yolo_idx}) 因单帧聚类失败，无法处理。")
                    continue
                current_global_r = frame_r + delta_r
                current_global_c = frame_c
                phys_anchor = ocr_item.get("yolo_anchor_details")
                if not phys_anchor:
                    self.logger.warning(f"{log_prefix} OBU '{ocr_text}' 缺少yolo_anchor_details，无法更新证据池。")
                    continue
                ocr_conf = ocr_item.get("ocr_confidence", 0.0)
                if ocr_text in obu_evidence_pool:
                    obu_evidence_pool[ocr_text]["physical_anchors"] = [phys_anchor]
                    obu_evidence_pool[ocr_text]["ocr_confidence"] = max(obu_evidence_pool[ocr_text].get("ocr_confidence",0.0), ocr_conf)
                    obu_evidence_pool[ocr_text]["last_seen_frame"] = current_frame_num
                    historical_coord = obu_evidence_pool[ocr_text]['logical_coord']
                    self.logger.debug(f"{log_prefix} 历史OBU '{ocr_text}' 更新。当前推断全局({current_global_r},{current_global_c}), 历史全局: {historical_coord}")
                    if historical_coord and (abs(historical_coord[0] - current_global_r) > 1 or abs(historical_coord[1] - current_global_c) > 0) :
                         self.logger.warning(f"{log_prefix} 历史OBU '{ocr_text}' 当前推断({current_global_r},{current_global_c}) 与历史 {historical_coord} 差异大。仍以历史为准。")
                else:
                    obu_evidence_pool[ocr_text] = {
                        "physical_anchors": [phys_anchor], "ocr_confidence": ocr_conf,
                        "logical_coord": (current_global_r, current_global_c),
                        "first_seen_frame": current_frame_num, "last_seen_frame": current_frame_num
                    }
                    self.logger.info(f"{log_prefix} 新OBU '{ocr_text}' 定位到全局 ({current_global_r},{current_global_c})")
        self.logger.info(f"{log_prefix} OBU证据池更新完毕。总数: {len(obu_evidence_pool)}")

        for r_init in range(_expected_total_rows):
            for c_init in range(_regular_cols):
                if logical_matrix[r_init][c_init] != -1:
                    logical_matrix[r_init][c_init] = 0
                    if (r_init, c_init) in recognized_texts_map:
                        del recognized_texts_map[(r_init, c_init)]

        stable_params_ref = session_data.get("stable_layout_parameters")
        if stable_params_ref:
            is_special_row_top_from_stable = stable_params_ref.get("special_row_at_logical_top", False)
            global_special_row_idx = -1

            # 对于第一帧，特殊行的全局位置是其 frame_r + delta_r
            # 对于后续帧，我们应该依赖第一帧学习到的全局特殊行位置（如果存了的话）
            # 或者，如果每帧都重新判断特殊行，那么也用 frame_r + delta_r
            # 当前的 stable_params_ref 存储的是第一帧的统计，包含 identified_special_row_frame_r

            identified_special_fr_from_stable = stable_params_ref.get("identified_special_row_frame_r", -1)
            if identified_special_fr_from_stable != -1 : # 如果第一帧识别到了特殊行
                # 那么它的全局行号应该是 identified_special_fr_from_stable + (第一帧的delta_r)
                # 对于后续帧，如果delta_r变了，这个全局特殊行号的计算需要小心
                # 简化：我们假设特殊行的全局逻辑位置是固定的（0或12）
                # 并通过 is_special_row_top_from_stable 来决定是哪个
                global_special_row_idx = 0 if is_special_row_top_from_stable else _expected_total_rows - 1
                self.logger.info(f"{log_prefix} 从stable_params获取特殊行位置：全局行 {global_special_row_idx} (基于is_top={is_special_row_top_from_stable})")
            else: # 如果第一帧没有识别到特殊行，后续帧也无法强制标记-1
                 self.logger.warning(f"{log_prefix} stable_params中未包含明确的特殊行识别信息。跳过强制-1标记。")


            if 0 <= global_special_row_idx < _expected_total_rows and \
               _regular_cols == 4 and _special_cols == 2:
                self.logger.info(f"{log_prefix} 应用特殊行-1标记规则到全局逻辑行 {global_special_row_idx}。")
                logical_matrix[global_special_row_idx][0] = -1
                logical_matrix[global_special_row_idx][3] = -1
        else:
            self.logger.warning(f"{log_prefix} stable_layout_parameters 为空，无法应用特殊行-1标记规则。")

        sorted_evidence = sorted(
            [(text, evi) for text, evi in obu_evidence_pool.items() if evi.get("logical_coord") is not None],
            key=lambda item: item[1].get("ocr_confidence", 0.0),
            reverse=True
        )
        for obu_text_fill, evidence_fill in sorted_evidence:
            final_logical_coord = evidence_fill.get("logical_coord")
            if not final_logical_coord: continue
            r_fill, c_fill = final_logical_coord
            if 0 <= r_fill < _expected_total_rows and 0 <= c_fill < _regular_cols:
                if logical_matrix[r_fill][c_fill] == 0:
                    logical_matrix[r_fill][c_fill] = 1
                    recognized_texts_map[(r_fill, c_fill)] = obu_text_fill
                elif logical_matrix[r_fill][c_fill] == 1:
                    if recognized_texts_map.get((r_fill,c_fill)) != obu_text_fill:
                        self.logger.warning(f"{log_prefix} 矩阵填充冲突: ({r_fill},{c_fill}) 已被 '{recognized_texts_map.get((r_fill,c_fill))}' 占据，无法放入 '{obu_text_fill}'。")
            else:
                 self.logger.warning(f"{log_prefix} OBU '{obu_text_fill}' 的最终逻辑坐标 ({r_fill},{c_fill}) 超出矩阵范围 ({_expected_total_rows}x{_regular_cols})。")

        num_filled_final = sum(1 for r_val in logical_matrix for status in r_val if status == 1)
        self.logger.info(f"{log_prefix} 最终矩阵构建完成，共填充 {num_filled_final} 个OBU。")

        for det_with_rc in current_frame_yolo_detections_with_rc:
            original_yolo_idx = det_with_rc["original_index"]
            ocr_item = map_ocr_results_by_yolo_idx.get(original_yolo_idx)
            is_ocr_valid_and_in_db = False
            if ocr_item:
                ocr_text_check = ocr_item.get("ocr_final_text", "")
                if ocr_text_check in VALID_OBU_CODES:
                    is_ocr_valid_and_in_db = True
            if not is_ocr_valid_and_in_db:
                frame_r, frame_c = det_with_rc.get('frame_r', -1), det_with_rc.get('frame_c', -1)
                if frame_r != -1 and frame_c != -1:
                    global_r_fail = frame_r + delta_r
                    global_c_fail = frame_c
                    if 0 <= global_r_fail < _expected_total_rows and \
                       0 <= global_c_fail < _regular_cols and \
                       logical_matrix[global_r_fail][global_c_fail] == 0:
                        logical_matrix[global_r_fail][global_c_fail] = 2

        self.logger.info(f"{log_prefix} 核心状态更新完成。")
        return logical_matrix, recognized_texts_map, warnings

# END OF LayoutStateManager CLASS