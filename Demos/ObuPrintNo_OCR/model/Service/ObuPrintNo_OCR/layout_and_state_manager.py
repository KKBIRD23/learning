# layout_and_state_manager.py
import cv2
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import os

from config import (
    LAYOUT_EXPECTED_TOTAL_ROWS,
    LAYOUT_REGULAR_COLS_COUNT,
    LAYOUT_SPECIAL_ROW_COLS_COUNT,
    LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING, # 用于首次学习的最小锚点数
    # LAYOUT_MIN_VALID_ROWS_FOR_LEARNING, # 可能不再直接使用，由聚类结果决定
    # LAYOUT_MIN_ANCHORS_PER_RELIABLE_ROW, # 同上
    LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR,
    LAYOUT_COL_GROUP_X_THRESHOLD_FACTOR, # 新增
    VALID_OBU_CODES
)

# 新的核心函数：通过XY独立聚类分析布局，并为每个YOLO框附加单帧逻辑行列号
def _analyze_layout_by_xy_clustering(
    yolo_detections: List[Dict[str, Any]], # 包含 cx,cy,w,h
    avg_obu_w: float, # 从所有检测中统计得到的平均OBU宽度
    avg_obu_h: float, # 从所有检测中统计得到的平均OBU高度
    config_threshold_factors: Dict[str, float], # 包含行和列的分组阈值因子
    logger: Any,
    session_id_for_log: str = "N/A"
) -> List[Dict[str, Any]]:
    """
    通过对YOLO检测结果的X、Y坐标独立进行阈值分组，为每个检测框附加单帧内的相对逻辑行列号。
    Args:
        yolo_detections: YOLO检测结果列表。
        avg_obu_w: 平均OBU宽度，用于计算X轴分组阈值。
        avg_obu_h: 平均OBU高度，用于计算Y轴分组阈值。
        config_threshold_factors: 包含 'row_y_factor' 和 'col_x_factor'。
        logger: 日志记录器。
    Returns:
        带有 'frame_r' 和 'frame_c' 属性的YOLO检测结果列表。
    """
    log_prefix = f"会话 {session_id_for_log} (_analyze_xy_clustering):"
    logger.info(f"{log_prefix} 开始通过XY独立聚类分析布局，输入锚点数: {len(yolo_detections)}")

    if not yolo_detections:
        return []

    # --- Y轴聚类 (确定 frame_r) ---
    y_coords = np.array([det['cy'] for det in yolo_detections])
    sorted_y_indices = np.argsort(y_coords)

    y_threshold = avg_obu_h * config_threshold_factors.get("row_y_factor", LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR)
    logger.info(f"{log_prefix} Y轴分组阈值: {y_threshold:.1f} (avg_h={avg_obu_h:.1f})")

    y_groups = []
    if sorted_y_indices.size > 0:
        current_y_group = [sorted_y_indices[0]]
        for i in range(1, len(sorted_y_indices)):
            idx_curr = sorted_y_indices[i]
            idx_prev_in_group = current_y_group[-1]
            if abs(y_coords[idx_curr] - y_coords[idx_prev_in_group]) < y_threshold:
                current_y_group.append(idx_curr)
            else:
                y_groups.append(list(current_y_group))
                current_y_group = [idx_curr]
        y_groups.append(list(current_y_group))

    # 为每个YOLO检测结果赋予 frame_r
    detections_with_frame_r = [det.copy() for det in yolo_detections] # 创建副本以修改
    for r_idx, group_indices in enumerate(y_groups):
        group_avg_y = np.mean([y_coords[i] for i in group_indices])
        logger.debug(f"{log_prefix} Y组 {r_idx} (avg_y={group_avg_y:.0f}): 包含 {len(group_indices)} 个锚点")
        for yolo_original_list_idx in group_indices:
            detections_with_frame_r[yolo_original_list_idx]['frame_r'] = r_idx

    logger.info(f"{log_prefix} Y轴聚类完成，得到 {len(y_groups)} 个逻辑行组。")

    # --- X轴聚类 (确定 frame_c) ---
    x_coords = np.array([det['cx'] for det in detections_with_frame_r]) # 使用已附加frame_r的列表
    sorted_x_indices = np.argsort(x_coords)

    x_threshold = avg_obu_w * config_threshold_factors.get("col_x_factor", LAYOUT_COL_GROUP_X_THRESHOLD_FACTOR)
    logger.info(f"{log_prefix} X轴分组阈值: {x_threshold:.1f} (avg_w={avg_obu_w:.1f})")

    x_groups = []
    if sorted_x_indices.size > 0:
        current_x_group = [sorted_x_indices[0]]
        for i in range(1, len(sorted_x_indices)):
            idx_curr = sorted_x_indices[i]
            idx_prev_in_group = current_x_group[-1]
            if abs(x_coords[idx_curr] - x_coords[idx_prev_in_group]) < x_threshold:
                current_x_group.append(idx_curr)
            else:
                x_groups.append(list(current_x_group))
                current_x_group = [idx_curr]
        x_groups.append(list(current_x_group))

    # 为每个YOLO检测结果赋予 frame_c
    # 注意：detections_with_frame_r 已经是副本，可以直接修改
    for c_idx, group_indices in enumerate(x_groups):
        group_avg_x = np.mean([x_coords[i] for i in group_indices])
        logger.debug(f"{log_prefix} X组 {c_idx} (avg_x={group_avg_x:.0f}): 包含 {len(group_indices)} 个锚点")
        for yolo_original_list_idx in group_indices:
            detections_with_frame_r[yolo_original_list_idx]['frame_c'] = c_idx
            # 同时记录一下这个分组的平均X，可能用于后续统计col_x_estimates
            detections_with_frame_r[yolo_original_list_idx]['frame_c_avg_x'] = group_avg_x


    logger.info(f"{log_prefix} X轴聚类完成，得到 {len(x_groups)} 个逻辑列组。")

    # 检查是否有未分配 frame_r 或 frame_c 的情况 (理论上不应发生，除非输入为空)
    for i, det in enumerate(detections_with_frame_r):
        if 'frame_r' not in det:
            logger.warning(f"{log_prefix} 锚点 {i} (cx={det['cx']}, cy={det['cy']}) 未分配 frame_r！")
            det['frame_r'] = -1 # 标记为无效
        if 'frame_c' not in det:
            logger.warning(f"{log_prefix} 锚点 {i} (cx={det['cx']}, cy={det['cy']}) 未分配 frame_c！")
            det['frame_c'] = -1 # 标记为无效

    return detections_with_frame_r


# _learn_initial_stable_layout_params 现在基于 _analyze_layout_by_xy_clustering 的结果进行统计
def _learn_initial_stable_layout_params(
    yolo_detections_with_frame_rc: List[Dict[str, Any]], # 已包含 frame_r, frame_c
    image_wh: Tuple[int, int],
    session_config: Dict[str, Any],
    logger: Any,
    session_id_for_log: str = "N/A"
) -> Optional[Dict[str, Any]]:
    log_prefix = f"会话 {session_id_for_log} (_learn_initial_stable_layout V2.0_XYClusterBased):"
    logger.critical(f"{log_prefix} CRITICAL_LOG: 基于XY聚类结果学习稳定布局。输入锚点数: {len(yolo_detections_with_frame_rc)}")

    if not yolo_detections_with_frame_rc:
        logger.critical(f"{log_prefix} CRITICAL_LOG: 无带frame_r/c的锚点输入。学习失败，返回None。")
        return None

    # 1. 统计平均OBU尺寸 (从原始YOLO检测中获取，因为frame_rc不影响尺寸)
    avg_obu_w = np.median([a['w'] for a in yolo_detections_with_frame_rc if a.get('w', 0) > 0])
    avg_obu_h = np.median([a['h'] for a in yolo_detections_with_frame_rc if a.get('h', 0) > 0])
    if not (avg_obu_w and avg_obu_h and avg_obu_w > 5 and avg_obu_h > 5):
        logger.critical(f"{log_prefix} CRITICAL_LOG: 统计得到的OBU平均尺寸 (W:{avg_obu_w}, H:{avg_obu_h}) 无效。学习失败，返回None。")
        return None
    logger.info(f"{log_prefix} OBU平均像素尺寸 (中位数): W={avg_obu_w:.1f}, H={avg_obu_h:.1f}")

    # 2. 统计各逻辑行/列的平均物理坐标，并计算平均行高/列间距
    row_physics_y_map: Dict[int, List[float]] = {} # frame_r -> list of cy
    col_physics_x_map: Dict[int, List[float]] = {} # frame_c -> list of cx

    for det in yolo_detections_with_frame_rc:
        frame_r, frame_c = det.get('frame_r', -1), det.get('frame_c', -1)
        if frame_r != -1:
            row_physics_y_map.setdefault(frame_r, []).append(det['cy'])
        if frame_c != -1:
            col_physics_x_map.setdefault(frame_c, []).append(det['cx'])

    if not row_physics_y_map or not col_physics_x_map:
        logger.critical(f"{log_prefix} CRITICAL_LOG: 未能从聚类结果中统计出行/列物理坐标。学习失败，返回None。")
        return None

    # 计算每行的平均Y (row_y_estimates_initial_guess)
    # 和每列的平均X (col_x_estimates_regular)
    num_frame_rows = max(row_physics_y_map.keys()) + 1 if row_physics_y_map else 0
    num_frame_cols = max(col_physics_x_map.keys()) + 1 if col_physics_x_map else 0

    # 我们期望的逻辑行/列数
    expected_total_rows = session_config.get("expected_total_rows", LAYOUT_EXPECTED_TOTAL_ROWS)
    expected_regular_cols = session_config.get("regular_cols_count", LAYOUT_REGULAR_COLS_COUNT)

    row_y_estimates_initial_guess = [np.mean(row_physics_y_map.get(r, [0])) for r in range(num_frame_rows)]
    # 如果聚类出的行数少于期望，需要填充或外推 (简化：暂时只用聚类出的)
    # TODO: 如果 num_frame_rows < expected_total_rows，需要更智能的填充/外推逻辑
    if num_frame_rows < expected_total_rows and num_frame_rows > 0:
        logger.warning(f"{log_prefix} 聚类行数 {num_frame_rows} 少于期望 {expected_total_rows}。Y估算可能不完整。")
        # 简单填充：用最后一个有效行的Y加上平均行高（如果能算出来）
        last_valid_y = row_y_estimates_initial_guess[-1]
        # 计算一个临时的平均行高用于填充
        temp_avg_row_h_for_fill = avg_obu_h * 1.2
        if len(row_y_estimates_initial_guess) >=2:
            gaps = np.diff(sorted(list(set(row_y_estimates_initial_guess)))) # 去重并排序后计算差值
            if gaps.size > 0: temp_avg_row_h_for_fill = np.median(gaps)

        for i_fill in range(num_frame_rows, expected_total_rows):
            row_y_estimates_initial_guess.append(last_valid_y + (i_fill - num_frame_rows + 1) * temp_avg_row_h_for_fill)


    col_x_estimates_regular = [np.mean(col_physics_x_map.get(c, [0])) for c in range(num_frame_cols)]
    # TODO: 如果 num_frame_cols < expected_regular_cols，需要填充/外推
    if num_frame_cols < expected_regular_cols and num_frame_cols > 0:
        logger.warning(f"{log_prefix} 聚类列数 {num_frame_cols} 少于期望 {expected_regular_cols}。X估算可能不完整。")
        last_valid_x = col_x_estimates_regular[-1]
        temp_avg_col_w_for_fill = avg_obu_w * 1.1 # 使用OBU宽度估算列间距
        if len(col_x_estimates_regular) >= 2:
            gaps_x = np.diff(sorted(list(set(col_x_estimates_regular))))
            if gaps_x.size > 0 : temp_avg_col_w_for_fill = np.median(gaps_x)

        for i_fill_c in range(num_frame_cols, expected_regular_cols):
            col_x_estimates_regular.append(last_valid_x + (i_fill_c - num_frame_cols + 1) * temp_avg_col_w_for_fill)


    logger.info(f"{log_prefix} 初始行Y估算 (基于聚类均值): {[int(y) for y in row_y_estimates_initial_guess if y is not None]}")
    logger.info(f"{log_prefix} 初始列X估算 (基于聚类均值): {[int(x) for x in col_x_estimates_regular if x is not None]}")

    # 计算平均物理行高 (基于聚类得到的行Y估算)
    avg_physical_row_height = avg_obu_h * 1.2 # Fallback
    if len(row_y_estimates_initial_guess) >= 2:
        valid_row_y_sorted = sorted([y for y in row_y_estimates_initial_guess if y is not None])
        if len(valid_row_y_sorted) >=2:
            row_gaps_y = np.abs(np.diff(valid_row_y_sorted))
            if row_gaps_y.size > 0:
                median_row_gap = np.median(row_gaps_y)
                if median_row_gap > avg_obu_h * 0.5:
                    avg_physical_row_height = median_row_gap
    logger.info(f"{log_prefix} 平均物理行高估算 (基于聚类行Y): {avg_physical_row_height:.1f}像素")

    # 判断特殊行位置 (基于聚类结果)
    special_row_at_logical_top = False
    # 简化版：如果最顶部的聚类行 (frame_r=0) 的列数等于special_row_cols，且下一行的列数接近常规列数，则认为特殊行在顶部
    # (这个逻辑需要更细化，当前聚类结果的行列号是临时的，还未映射到13x4的最终逻辑)
    # 暂时先用一个简单的默认值或基于整体形状的猜测
    # TODO: 强化特殊行判断逻辑，结合聚类出的行列数量和位置
    logger.info(f"{log_prefix} 判断特殊行在逻辑顶部 (暂用默认): {special_row_at_logical_top}")


    stable_params = {
        "avg_physical_row_height": avg_physical_row_height,
        "col_x_estimates_regular": col_x_estimates_regular, # 长度可能不足expected_regular_cols
        "avg_obu_w": avg_obu_w,
        "avg_obu_h": avg_obu_h,
        "special_row_at_logical_top": special_row_at_logical_top,
        "row_y_estimates_initial_guess": row_y_estimates_initial_guess # 长度可能不足expected_total_rows
    }
    logger.critical(f"{log_prefix} CRITICAL_LOG: 初始稳定布局参数学习成功 (基于XY聚类)。")
    return stable_params


class LayoutStateManager:
    def __init__(self, config_params: Dict[str, Any], logger: Any):
        self.config_params = config_params
        self.logger = logger
        self.expected_total_rows = self.config_params.get("LAYOUT_EXPECTED_TOTAL_ROWS", 13)
        self.regular_cols_count = self.config_params.get("LAYOUT_REGULAR_COLS_COUNT", 4)
        self.special_row_cols_count = self.config_params.get("LAYOUT_SPECIAL_ROW_COLS_COUNT", 2)

    def learn_initial_stable_layout(self,
                                    yolo_detections_for_calib: List[Dict[str, Any]],
                                    image_wh: Tuple[int, int],
                                    session_config_from_session: Dict[str, Any],
                                    session_id: str) -> Optional[Dict[str, Any]]:
        """
        学习初始稳定布局参数。
        首先通过XY独立聚类为YOLO检测框附加单帧逻辑行列号 (frame_r, frame_c)。
        然后基于这些带单帧逻辑坐标的检测框，统计生成稳定的布局参数。
        """
        log_prefix = f"会话 {session_id} (learn_initial_stable_layout_entry):"
        self.logger.info(f"{log_prefix} 开始学习初始稳定布局参数...")

        if not yolo_detections_for_calib:
            self.logger.warning(f"{log_prefix} 无YOLO检测结果用于学习初始布局。")
            return None

        # 先统计一次全局的平均OBU尺寸，用于后续聚类阈值计算
        all_ws = [d['w'] for d in yolo_detections_for_calib if d.get('w', 0) > 0]
        all_hs = [d['h'] for d in yolo_detections_for_calib if d.get('h', 0) > 0]
        avg_w_for_clustering = np.median(all_ws) if all_ws else 100.0
        avg_h_for_clustering = np.median(all_hs) if all_hs else 40.0

        if avg_w_for_clustering <=5 or avg_h_for_clustering <=5:
             self.logger.warning(f"{log_prefix} 统计的初始平均OBU尺寸过小 (W:{avg_w_for_clustering}, H:{avg_h_for_clustering})，可能影响聚类。")
             # 可以选择返回None或使用更硬编码的fallback

        # 步骤1: XY独立聚类，为每个检测框附加 frame_r, frame_c
        detections_with_frame_rc = _analyze_layout_by_xy_clustering(
            yolo_detections_for_calib,
            avg_w_for_clustering,
            avg_h_for_clustering,
            { # 从self.config_params获取阈值因子
                "row_y_factor": self.config_params.get("LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR", 0.4),
                "col_x_factor": self.config_params.get("LAYOUT_COL_GROUP_X_THRESHOLD_FACTOR", 0.6)
            },
            self.logger,
            session_id
        )

        if not detections_with_frame_rc:
            self.logger.error(f"{log_prefix} XY独立聚类未能为任何锚点分配单帧逻辑坐标。")
            return None

        # 步骤2: 基于带frame_r/c的检测框，学习（统计）稳定的布局参数
        stable_params = _learn_initial_stable_layout_params(
            detections_with_frame_rc, # 传递已处理过的列表
            image_wh,
            session_config_from_session,
            self.logger,
            session_id
        )
        return stable_params

    # ... (determine_y_axis_anchor, _map_single_anchor_to_logical_using_params,
    #      draw_stable_layout_on_image, update_session_state_with_reference_logic 方法保持不变，
    #      但请确保它们内部对 self.config_params 的访问是正确的，例如获取阈值因子等)
    def determine_y_axis_anchor(self,
                                current_frame_verified_obus: List[Dict[str, Any]],
                                obu_evidence_pool: Dict[str, Any],
                                stable_layout_params: Dict[str, Any],
                                session_id: str,
                                current_frame_num: int
                                ) -> Tuple[Optional[Dict[str, Any]], List[float], bool]:
        # ... (此方法逻辑与上一版相同) ...
        log_prefix = f"会话 {session_id} (determine_y_axis_anchor F{current_frame_num}):"
        is_frame_skipped = False
        y_anchor_info = None
        current_dynamic_row_y_estimates = list(stable_layout_params.get("row_y_estimates_initial_guess", []))
        if not current_dynamic_row_y_estimates:
            self.logger.warning(f"{log_prefix} stable_layout_params中无row_y_estimates_initial_guess，生成临时Y估算。")
            img_h_temp = self.config_params.get("IMAGE_FALLBACK_HEIGHT_FOR_LAYOUT", 5000)
            current_dynamic_row_y_estimates = [(r + 0.5) * (img_h_temp / self.expected_total_rows) for r in range(self.expected_total_rows)]
        historical_overlaps_in_current_frame = []
        for obu_cur in current_frame_verified_obus:
            if obu_cur["text"] in obu_evidence_pool:
                hist_entry = obu_evidence_pool[obu_cur["text"]]
                if hist_entry.get("logical_coord") is not None: # 确保历史条目有已确定的逻辑坐标
                    historical_overlaps_in_current_frame.append({
                        "text": obu_cur["text"],
                        "current_physical_anchor": obu_cur["physical_anchor"],
                        "historical_logical_coord": hist_entry["logical_coord"],
                        # 新增：记录当前帧通过XY聚类得到的单帧逻辑行号，用于后续delta_r计算
                        "current_frame_r": obu_cur.get("frame_r", -1)
                    })
        self.logger.info(f"{log_prefix} 在当前帧找到 {len(historical_overlaps_in_current_frame)} 个有效的历史重叠OBU用于Y轴锚定。")
        if current_frame_num > 1 and not historical_overlaps_in_current_frame:
            self.logger.error(f"{log_prefix} 检测到漏帧！当前帧与历史无有效重叠OBU。")
            is_frame_skipped = True
            return None, current_dynamic_row_y_estimates, is_frame_skipped
        min_y_historical_obu_for_anchor = None
        if historical_overlaps_in_current_frame:
            # 筛选出那些在当前帧也被成功赋予了 frame_r 的历史OBU
            valid_anchor_candidates = [o for o in historical_overlaps_in_current_frame if o.get("current_frame_r", -1) != -1]
            if valid_anchor_candidates:
                min_y_historical_obu_for_anchor = min(
                    valid_anchor_candidates,
                    key=lambda o: o["current_physical_anchor"]["cy"]
                )
        if min_y_historical_obu_for_anchor:
            y_ref_current = min_y_historical_obu_for_anchor["current_physical_anchor"]["cy"]
            l_ref_global = min_y_historical_obu_for_anchor["historical_logical_coord"][0]
            frame_r_anchor = min_y_historical_obu_for_anchor["current_frame_r"]

            # 计算 delta_r (全局逻辑行号与当前帧聚类得到的行号的偏移)
            delta_r = l_ref_global - frame_r_anchor
            self.logger.info(f"{log_prefix} Y轴锚定参照: OBU='{min_y_historical_obu_for_anchor['text']}', "
                             f"历史全局行L={l_ref_global}, 当前帧聚类行R={frame_r_anchor}, delta_r={delta_r}")

            avg_row_h = stable_layout_params.get("avg_physical_row_height")
            if avg_row_h is None or avg_row_h <= 1:
                self.logger.warning(f"{log_prefix} 稳定参数中的平均行高无效 ({avg_row_h})，无法精确Y轴锚定。")
            else:
                # 更新动态行Y估算：基于Y轴锚定点的物理Y，和稳定的平均行高，以及计算出的delta_r
                # (或者说，我们不再强依赖一个完整的 current_dynamic_row_y_estimates 列表，
                #  而是直接在需要时用 delta_r 和 frame_r 来计算全局逻辑行)
                # 但为了绘图和某些回退逻辑，还是生成一个
                # 假设 frame_r=0 对应的物理Y可以通过 y_ref_current - frame_r_anchor * avg_row_h 估算
                estimated_y_for_frame_r0 = y_ref_current - frame_r_anchor * avg_row_h
                current_dynamic_row_y_estimates = [
                     estimated_y_for_frame_r0 + (i_global - delta_r) * avg_row_h for i_global in range(self.expected_total_rows)
                ]
                y_anchor_info = {
                    "ref_obu_text": min_y_historical_obu_for_anchor['text'],
                    "ref_global_logical_row": l_ref_global,
                    "ref_frame_logical_row": frame_r_anchor,
                    "ref_physical_y_current": y_ref_current,
                    "delta_r": delta_r # 保存这个重要的偏移量
                }
                self.logger.info(f"{log_prefix} Y轴锚定成功。Delta_r: {delta_r}")
                self.logger.info(f"  更新后（理论上的）动态行Y估算: {[int(y) for y in current_dynamic_row_y_estimates]}")
        elif current_frame_num > 1:
            self.logger.warning(f"{log_prefix} 后续帧未能找到Y轴锚定参照物。将使用基于稳定参数的初始Y估算，delta_r=0。")
            y_anchor_info = {"delta_r": 0} # 假设无偏移
        else: # 第一帧
            y_anchor_info = {"delta_r": 0} # 第一帧无偏移

        return y_anchor_info, current_dynamic_row_y_estimates, is_frame_skipped

    def _map_single_anchor_to_logical_using_params(
        self, anchor_to_map: Dict[str, Any],
        row_y_estimates_map: List[float],
        col_x_estimates_regular_map: List[float],
        stable_layout_params_map: Dict[str, Any],
        logger_map: Any,
        session_id_for_log_map: str,
        log_prefix_map: str = "(_map_single_anchor)"
    ) -> Optional[Tuple[int, int]]:
        # ... (此方法逻辑与上一版相同) ...
        if not row_y_estimates_map or not col_x_estimates_regular_map or \
           any(x is None for x in col_x_estimates_regular_map):
            return None
        avg_row_h = stable_layout_params_map.get("avg_physical_row_height", 50)
        avg_obu_w = stable_layout_params_map.get("avg_obu_w", 100)
        y_match_threshold = avg_row_h * self.config_params.get("LAYOUT_Y_MATCH_THRESHOLD_FACTOR", 0.85)
        x_match_threshold = avg_obu_w * self.config_params.get("LAYOUT_X_MATCH_THRESHOLD_FACTOR", 0.85)
        special_on_top = stable_layout_params_map.get("special_row_at_logical_top", False)
        cand_r = -1; min_y_d_sq = float('inf')
        for r_idx, est_y in enumerate(row_y_estimates_map):
            dist_y_sq = (anchor_to_map['cy'] - est_y)**2
            if dist_y_sq < min_y_d_sq and dist_y_sq < y_match_threshold**2:
                min_y_d_sq = dist_y_sq; cand_r = r_idx
        if cand_r == -1: return None
        is_special_row = (cand_r == (self.expected_total_rows - 1) and not special_on_top) or \
                         (cand_r == 0 and special_on_top)
        cols_to_match_xs = []
        if is_special_row and self.special_row_cols_count == 2 and self.regular_cols_count == 4:
            if len(col_x_estimates_regular_map) == 4:
                cols_to_match_xs = [col_x_estimates_regular_map[1], col_x_estimates_regular_map[2]]
        else:
            cols_to_match_xs = col_x_estimates_regular_map
        if not cols_to_match_xs or not all(isinstance(x, (int, float)) for x in cols_to_match_xs):
            return None
        cand_c_in_options = -1; min_x_d_sq = float('inf')
        for c_opt_idx, est_x in enumerate(cols_to_match_xs):
            dist_x_sq = (anchor_to_map['cx'] - est_x)**2
            if dist_x_sq < min_x_d_sq and dist_x_sq < x_match_threshold**2:
                min_x_d_sq = dist_x_sq; cand_c_in_options = c_opt_idx
        if cand_c_in_options != -1:
            final_c = cand_c_in_options
            if is_special_row and self.special_row_cols_count == 2 and self.regular_cols_count == 4:
                final_c = cand_c_in_options + 1
            if 0 <= final_c < self.regular_cols_count:
                return (cand_r, final_c)
        return None

    def draw_stable_layout_on_image(self,
                                     stable_layout_params: Dict[str, Any],
                                     image_wh: Tuple[int, int],
                                     session_id: str,
                                     frame_num: int):
        # ... (此方法逻辑与上一版相同) ...
        log_prefix = f"会话 {session_id} (draw_stable_layout F{frame_num}):"
        self.logger.critical(f"{log_prefix} CRITICAL_LOG: 进入绘制稳定布局图函数。")
        img_w, img_h = image_wh
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        self.logger.info(f"{log_prefix} 画布尺寸: W={img_w}, H={img_h}")
        row_y_estimates = stable_layout_params.get("row_y_estimates_initial_guess", [])
        col_x_estimates_reg = stable_layout_params.get("col_x_estimates_regular", [])
        avg_w = stable_layout_params.get("avg_obu_w", 100)
        avg_h = stable_layout_params.get("avg_obu_h", 40)
        special_on_top = stable_layout_params.get("special_row_at_logical_top", False)
        if not row_y_estimates or not col_x_estimates_reg or \
           not all(isinstance(x, (int,float)) for x_idx, x in enumerate(col_x_estimates_reg) if x is not None and x_idx < self.regular_cols_count ):
            self.logger.critical(f"{log_prefix} CRITICAL_LOG: 稳定布局参数不完整或类型错误，无法绘制坑位图。Rows: {len(row_y_estimates)}, Cols: {col_x_estimates_reg}")
            return
        font_scale = 0.5; font_thickness = 1
        for r_log in range(self.expected_total_rows):
            for c_log_visual in range(self.regular_cols_count):
                is_special_row_current = (r_log == 0 and special_on_top) or \
                                         (r_log == self.expected_total_rows - 1 and not special_on_top)
                actual_c_log_for_data = c_log_visual
                is_placeholder_slot = False
                if is_special_row_current and self.special_row_cols_count == 2 and self.regular_cols_count == 4:
                    if c_log_visual == 0 or c_log_visual == 3:
                        is_placeholder_slot = True
                    else:
                        actual_c_log_for_data = c_log_visual -1
                slot_color = (50, 50, 50); text_color = (100, 100, 100); text_to_draw = "N/A"
                if not is_placeholder_slot:
                    if not (0 <= actual_c_log_for_data < len(col_x_estimates_reg) and \
                            col_x_estimates_reg[actual_c_log_for_data] is not None):
                        continue
                    slot_color = (0, 150, 0) if not is_special_row_current else (0, 100, 150)
                    text_color = (255, 255, 255); text_to_draw = f"({r_log},{c_log_visual})"
                try:
                    center_y = int(row_y_estimates[r_log])
                    if is_special_row_current and self.special_row_cols_count == 2 and self.regular_cols_count == 4:
                        if c_log_visual == 0: center_x = int(col_x_estimates_reg[0])
                        elif c_log_visual == 1: center_x = int(col_x_estimates_reg[1])
                        elif c_log_visual == 2: center_x = int(col_x_estimates_reg[2])
                        elif c_log_visual == 3: center_x = int(col_x_estimates_reg[3])
                        else: continue
                    else:
                        if 0 <= c_log_visual < len(col_x_estimates_reg) and col_x_estimates_reg[c_log_visual] is not None:
                            center_x = int(col_x_estimates_reg[c_log_visual])
                        else: continue
                except (IndexError, TypeError) as e_coord:
                    self.logger.error(f"{log_prefix} 绘制坑位图时获取坐标错误: r={r_log}, c_visual={c_log_visual}, Error: {e_coord}")
                    continue
                half_w, half_h = int(avg_w / 2), int(avg_h / 2)
                pt1 = (max(0, center_x - half_w), max(0, center_y - half_h))
                pt2 = (min(img_w -1, center_x + half_w), min(img_h -1, center_y + half_h))
                if pt1[0] >= pt2[0] or pt1[1] >= pt2[1]: continue
                cv2.rectangle(canvas, pt1, pt2, slot_color, -1 if not is_placeholder_slot else 1)
                if not is_placeholder_slot: cv2.rectangle(canvas, pt1, pt2, (200,200,200), 1)
                (tw, th), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.putText(canvas, text_to_draw, (center_x - tw // 2, center_y + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
        output_dir = os.path.join(self.config_params.get("PROCESS_PHOTO_DIR", "process_photo"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        map_filename = f"stable_layout_map_s{session_id[:8]}_f{frame_num}.png"
        output_path = os.path.join(output_dir, map_filename)
        try:
            save_success = cv2.imwrite(output_path, canvas)
            if save_success:
                self.logger.critical(f"{log_prefix} CRITICAL_LOG: 稳定布局可用坑位图已成功保存到: {output_path}")
            else:
                self.logger.critical(f"{log_prefix} CRITICAL_LOG: cv2.imwrite未能保存稳定布局图到 {output_path} (返回False)")
        except Exception as e_save:
            self.logger.critical(f"{log_prefix} CRITICAL_LOG: 保存稳定布局坑位图失败 {output_path}: {e_save}", exc_info=True)

    def update_session_state_with_reference_logic(
        self,
        session_data: Dict[str, Any],
        current_frame_yolo_detections_with_frame_rc: List[Dict[str, Any]], # 已包含 frame_r, frame_c
        current_frame_ocr_results: List[Dict[str, Any]],
        y_anchor_info: Optional[Dict[str, Any]], # 包含 delta_r
        # current_dynamic_row_y_estimates: List[float], # 这个可能不再直接使用，因为有delta_r
        stable_layout_params: Dict[str, Any],
        session_id: str,
        current_frame_num: int
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], str], List[Dict[str, Any]]]:
        log_prefix = f"会话 {session_id} (update_state_ref_logic V2.0 F{current_frame_num}):"
        self.logger.info(f"{log_prefix} 开始核心状态更新 (基于XY聚类和Y轴锚定)...")

        obu_evidence_pool = session_data["obu_evidence_pool"]
        logical_matrix = session_data["logical_matrix"]
        recognized_texts_map = session_data["recognized_texts_map"]
        warnings = []

        # 获取 delta_r，如果Y轴锚定失败或首帧，则为0
        delta_r = 0
        if y_anchor_info and "delta_r" in y_anchor_info:
            delta_r = y_anchor_info["delta_r"]
        self.logger.info(f"{log_prefix} 使用的行偏移 delta_r: {delta_r}")

        # 1. 更新 obu_evidence_pool，并为当前帧OBU计算初步的全局逻辑坐标
        #    (历史OBU的logical_coord保持不变)
        map_ocr_results_by_yolo_idx = {ocr.get("original_index"): ocr for ocr in current_frame_ocr_results if ocr}

        for det_with_rc in current_frame_yolo_detections_with_frame_rc:
            original_yolo_idx = det_with_rc["original_index"]
            ocr_item = map_ocr_results_by_yolo_idx.get(original_yolo_idx)
            if not ocr_item: continue

            ocr_text = ocr_item.get("ocr_final_text", "")
            if ocr_text in VALID_OBU_CODES:
                frame_r, frame_c = det_with_rc.get('frame_r', -1), det_with_rc.get('frame_c', -1)
                if frame_r == -1 or frame_c == -1: # 聚类失败的跳过
                    self.logger.warning(f"{log_prefix} OBU '{ocr_text}' (YOLO idx {original_yolo_idx}) 因单帧聚类失败，无法处理。")
                    continue

                # 计算当前帧推断的全局逻辑坐标
                current_global_r = frame_r + delta_r
                current_global_c = frame_c # X轴暂时不加偏移

                phys_anchor = ocr_item.get("yolo_anchor_details") # 从OCR结果中获取匹配的YOLO锚点细节
                ocr_conf = ocr_item.get("ocr_confidence", 0.0)

                if text in obu_evidence_pool: # 历史OBU
                    obu_evidence_pool[text]["physical_anchors"] = [phys_anchor]
                    obu_evidence_pool[text]["ocr_confidence"] = max(obu_evidence_pool[text].get("ocr_confidence",0.0), ocr_conf)
                    obu_evidence_pool[text]["last_seen_frame"] = current_frame_num
                    # logical_coord 保持不变，不被当前帧的推断覆盖 (100%信任历史)
                    self.logger.debug(f"{log_prefix} 历史OBU '{text}' 更新物理信息。保持历史逻辑坐标: {obu_evidence_pool[text]['logical_coord']}")
                else: # 新OBU
                    obu_evidence_pool[text] = {
                        "physical_anchors": [phys_anchor], "ocr_confidence": ocr_conf,
                        "logical_coord": (current_global_r, current_global_c), # 直接使用当前帧推断的全局坐标
                        "first_seen_frame": current_frame_num, "last_seen_frame": current_frame_num
                    }
                    self.logger.info(f"{log_prefix} 新OBU '{text}' 初步定位到全局 ({current_global_r},{current_global_c})")

        self.logger.info(f"{log_prefix} OBU证据池更新完毕。总数: {len(obu_evidence_pool)}")

        # 2. 构建最终输出矩阵 (基于 obu_evidence_pool 中所有已确定 logical_coord 的OBU)
        # 清空矩阵 (保留-1)
        for r_init in range(self.expected_total_rows):
            for c_init in range(self.regular_cols_count):
                if logical_matrix[r_init][c_init] != -1: # 只有非永久不可用才清零
                    logical_matrix[r_init][c_init] = 0
                    if (r_init, c_init) in recognized_texts_map:
                        del recognized_texts_map[(r_init, c_init)]

        # 填充矩阵，解决冲突 (OCR分高的优先)
        sorted_evidence = sorted(
            [(text, evi) for text, evi in obu_evidence_pool.items() if evi.get("logical_coord") is not None],
            key=lambda item: item[1].get("ocr_confidence", 0.0),
            reverse=True
        )

        for obu_text_fill, evidence_fill in sorted_evidence:
            r_fill, c_fill = evidence_fill["logical_coord"]
            # 确保行列号在13x4的有效范围内
            if 0 <= r_fill < self.expected_total_rows and 0 <= c_fill < self.regular_cols_count:
                if logical_matrix[r_fill][c_fill] == 0:
                    logical_matrix[r_fill][c_fill] = 1
                    recognized_texts_map[(r_fill, c_fill)] = obu_text_fill
                elif logical_matrix[r_fill][c_fill] == 1:
                    # 如果一个高分OBU想覆盖一个低分OBU（因为我们是按分数排序的，所以这里应该是低分想覆盖高分）
                    # 或者不同OBU映射到同一位置
                    if recognized_texts_map.get((r_fill,c_fill)) != obu_text_fill: # 确保不是同一个OBU的重复（理论上不应发生）
                        self.logger.warning(f"{log_prefix} 矩阵填充冲突: 坑位 ({r_fill},{c_fill}) 已被 "
                                         f"'{recognized_texts_map.get((r_fill,c_fill))}' (分更高或先到) 占据，无法放入 '{obu_text_fill}'。")
            else:
                 self.logger.warning(f"{log_prefix} OBU '{obu_text_fill}' 的最终逻辑坐标 ({r_fill},{c_fill}) 超出矩阵范围。")

        num_filled_final = sum(1 for r_val in logical_matrix for status in r_val if status == 1)
        self.logger.info(f"{log_prefix} 最终矩阵构建完成，共填充 {num_filled_final} 个OBU。")

        # 3. 标记OCR失败或无效的格子
        #    对于当前帧检测到但未成功识别并放入obu_evidence_pool的YOLO框，
        #    如果它们通过XY聚类得到了单帧逻辑坐标(frame_r, frame_c)，
        #    并且转换到全局坐标后，该坑位为空，则标记为2.
        for det_with_rc in current_frame_yolo_detections_with_frame_rc:
            original_yolo_idx = det_with_rc["original_index"]
            ocr_item = map_ocr_results_by_yolo_idx.get(original_yolo_idx)

            is_ocr_valid_and_in_db = False
            if ocr_item:
                ocr_text_check = ocr_item.get("ocr_final_text", "")
                if ocr_text_check in VALID_OBU_CODES:
                    is_ocr_valid_and_in_db = True

            if not is_ocr_valid_and_in_db: # 如果OCR失败或无效
                frame_r, frame_c = det_with_rc.get('frame_r', -1), det_with_rc.get('frame_c', -1)
                if frame_r != -1 and frame_c != -1:
                    global_r_fail = frame_r + delta_r
                    global_c_fail = frame_c

                    if 0 <= global_r_fail < self.expected_total_rows and \
                       0 <= global_c_fail < self.regular_cols_count and \
                       logical_matrix[global_r_fail][global_c_fail] == 0: # 只标记之前是“未知”的
                        logical_matrix[global_r_fail][global_c_fail] = 2

        self.logger.info(f"{log_prefix} 核心状态更新完成。")
        return logical_matrix, recognized_texts_map, warnings