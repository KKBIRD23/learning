# layout_and_state_manager.py
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

# 从 config 导入一些布局相关的常量，如果它们没有通过参数传入
from config import (
    LAYOUT_EXPECTED_TOTAL_ROWS,
    LAYOUT_REGULAR_COLS_COUNT,
    LAYOUT_SPECIAL_ROW_COLS_COUNT,
    LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING,
    LAYOUT_MIN_VALID_ROWS_FOR_LEARNING,
    LAYOUT_MIN_ANCHORS_PER_RELIABLE_ROW,
    LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR,
    VALID_OBU_CODES # 用于最终矩阵构建时的校验（虽然主要校验已在OCR结果整合后）
)
# 从 image_utils 导入，如果需要
# from image_utils import ...

# 辅助函数：从YOLO锚点学习初始稳定布局参数
# 这个函数是基于您1688行代码中的 _learn_layout_parameters_from_anchors_v9 进行调整和适配的
# 主要目标是学习一套基准的行高、列X估算、OBU平均尺寸、特殊行位置
def _learn_initial_stable_layout_params(
    yolo_detections_for_calib: List[Dict[str, Any]], # 包含 cx,cy,w,h,score
    image_wh: Tuple[int, int],
    session_config: Dict[str, Any], # 包含 expected_total_rows, regular_cols_count 等
    logger: Any,
    session_id_for_log: str = "N/A"
) -> Optional[Dict[str, Any]]:
    """
    从一组高质量的YOLO检测结果中学习初始的、稳定的布局参数。
    Args:
        yolo_detections_for_calib: 用于校准的YOLO检测结果列表。
        image_wh: 图像的 (宽度, 高度)。
        session_config: 当前会话的布局配置。
        logger: 日志记录器。
        session_id_for_log: 用于日志的会话ID。
    Returns:
        一个包含稳定布局参数的字典，如果学习失败则返回None。
        参数字典包含:
            "avg_physical_row_height": float
            "col_x_estimates_regular": List[float] (长度为常规列数)
            "avg_obu_w": float
            "avg_obu_h": float
            "special_row_at_logical_top": bool
            "row_y_estimates_initial_guess": List[float] (基于学习到的参数的初始Y估算)
    """
    log_prefix = f"会话 {session_id_for_log} (_learn_initial_stable_layout):"
    logger.info(f"{log_prefix} 开始学习初始稳定布局参数...")

    img_w, img_h = image_wh
    expected_total_rows = session_config.get("expected_total_rows", LAYOUT_EXPECTED_TOTAL_ROWS)
    expected_regular_cols = session_config.get("regular_cols_count", LAYOUT_REGULAR_COLS_COUNT)
    special_row_cols = session_config.get("special_row_cols_count", LAYOUT_SPECIAL_ROW_COLS_COUNT)

    if not yolo_detections_for_calib or len(yolo_detections_for_calib) < LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING:
        logger.warning(f"{log_prefix} 用于校准的YOLO锚点数量 ({len(yolo_detections_for_calib)}) 过少。")
        return None

    # 1. 行分组 (与您之前代码中的 _learn_layout_parameters_from_anchors_v9 类似)
    anchors_sorted_by_y = sorted(yolo_detections_for_calib, key=lambda a: (a['cy'], a['cx']))
    physical_rows_grouped = []
    # 使用中位数高度进行分组阈值计算，更鲁棒
    median_h_all = np.median([a['h'] for a in anchors_sorted_by_y if a.get('h', 0) > 0])
    if not median_h_all or median_h_all <=0: median_h_all = 30 # Fallback

    y_threshold_for_grouping = median_h_all * LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR

    _current_row_group = [anchors_sorted_by_y[0]]
    for i in range(1, len(anchors_sorted_by_y)):
        if abs(anchors_sorted_by_y[i]['cy'] - _current_row_group[-1]['cy']) < y_threshold_for_grouping:
            _current_row_group.append(anchors_sorted_by_y[i])
        else:
            physical_rows_grouped.append(sorted(_current_row_group, key=lambda a: a['cx']))
            _current_row_group = [anchors_sorted_by_y[i]]
    if _current_row_group:
        physical_rows_grouped.append(sorted(_current_row_group, key=lambda a: a['cx']))

    if not physical_rows_grouped:
        logger.warning(f"{log_prefix} 行分组失败。")
        return None

    reliable_physical_rows = [row for row in physical_rows_grouped if len(row) >= LAYOUT_MIN_ANCHORS_PER_RELIABLE_ROW]
    if len(reliable_physical_rows) < LAYOUT_MIN_VALID_ROWS_FOR_LEARNING:
        logger.warning(f"{log_prefix} 可靠物理行数量 ({len(reliable_physical_rows)}) 过少。")
        return None
    logger.info(f"{log_prefix} 分组得到 {len(physical_rows_grouped)} 个物理行，其中 {len(reliable_physical_rows)} 个可靠。")

    # 2. 学习OBU平均像素尺寸 (使用中位数更鲁棒)
    avg_obu_w = np.median([a['w'] for a in yolo_detections_for_calib if a.get('w', 0) > 0])
    avg_obu_h = np.median([a['h'] for a in yolo_detections_for_calib if a.get('h', 0) > 0])
    if not (avg_obu_w and avg_obu_h and avg_obu_w > 5 and avg_obu_h > 5): # 简单合理性检查
        logger.warning(f"{log_prefix} 学习到的OBU平均尺寸 (W:{avg_obu_w}, H:{avg_obu_h}) 无效。")
        return None
    logger.info(f"{log_prefix} OBU平均像素尺寸 (中位数): W={avg_obu_w:.1f}, H={avg_obu_h:.1f}")

    # 3. 学习平均物理行高 (基于可靠行间距的中位数)
    avg_physical_row_height = avg_obu_h * 1.2 # Fallback
    if len(reliable_physical_rows) >= 2:
        row_centers_y = [np.mean([a['cy'] for a in row]) for row in reliable_physical_rows]
        row_gaps_y = np.abs(np.diff(row_centers_y))
        if row_gaps_y.size > 0:
            median_row_gap = np.median(row_gaps_y)
            if median_row_gap > avg_obu_h * 0.5: # 确保行高比OBU自身高
                avg_physical_row_height = median_row_gap
    logger.info(f"{log_prefix} 平均物理行高估算: {avg_physical_row_height:.1f}像素")


    # 4. 判断特殊行位置 (逻辑顶部或底部)
    special_row_at_logical_top = False # 默认在底部
    # (与您之前代码中 _learn_layout_parameters_from_anchors_v9 的判断逻辑类似)
    # 简化版：检查物理顶部和底部的可靠行是否符合特殊行特征
    if reliable_physical_rows:
        first_reliable_row_len = len(reliable_physical_rows[0])
        last_reliable_row_len = len(reliable_physical_rows[-1])

        # 推断常规列数 (从非特殊行中学习)
        non_special_like_rows_for_cols = [r for r in reliable_physical_rows if len(r) != special_row_cols]
        inferred_reg_cols = expected_regular_cols
        if non_special_like_rows_for_cols:
            col_counts = Counter(len(r) for r in non_special_like_rows_for_cols)
            if col_counts: inferred_reg_cols = col_counts.most_common(1)[0][0]

        is_first_special_like = (first_reliable_row_len == special_row_cols)
        is_last_special_like = (last_reliable_row_len == special_row_cols)

        if is_first_special_like and not is_last_special_like:
            # 如果只有顶部像特殊行，且后续行像常规行
            if len(reliable_physical_rows) > 1 and abs(len(reliable_physical_rows[1]) - inferred_reg_cols) <= 1:
                special_row_at_logical_top = True
        elif is_last_special_like and not is_first_special_like:
            # 如果只有底部像特殊行，且其前一行像常规行
             if len(reliable_physical_rows) > 1 and abs(len(reliable_physical_rows[-2]) - inferred_reg_cols) <= 1:
                special_row_at_logical_top = False # 明确在底部
        # 如果顶部和底部都像，或者都不像，或者只有一行，则默认在底部 (或需要更复杂逻辑)
        # 此处简化：如果顶部符合且底部不符合，则认为在顶部，否则在底部。
    logger.info(f"{log_prefix} 判断特殊行在逻辑顶部: {special_row_at_logical_top}")

    # 5. 学习常规列的X坐标估算 (col_x_estimates_regular)
    # (与您之前代码中 _learn_layout_parameters_from_anchors_v9 的列学习逻辑类似，但目标是得到稳定的基准)
    col_x_estimates_regular = [None] * expected_regular_cols

    # 尝试从符合常规列数的可靠行中学习
    standard_rows_for_x_learning = [row for row in reliable_physical_rows if len(row) == expected_regular_cols]
    if not standard_rows_for_x_learning: # 如果没有完美匹配的，放宽到接近的
        standard_rows_for_x_learning = [row for row in reliable_physical_rows if abs(len(row) - expected_regular_cols) <=1 and len(row) > 1]

    if standard_rows_for_x_learning:
        temp_cols_x_accumulator = [[] for _ in range(expected_regular_cols)]
        for row in standard_rows_for_x_learning:
            # 如果行内元素少于expected_regular_cols，我们假设它们是左对齐的（或需要更复杂的对齐逻辑）
            # 为简化，这里只用那些列数正好等于expected_regular_cols的行来精确学习
            if len(row) == expected_regular_cols:
                for c_idx, anchor in enumerate(row):
                    temp_cols_x_accumulator[c_idx].append(anchor['cx'])

        for c_idx in range(expected_regular_cols):
            if temp_cols_x_accumulator[c_idx]:
                col_x_estimates_regular[c_idx] = np.median(temp_cols_x_accumulator[c_idx]) # 使用中位数更鲁棒

    # Fallback填充未学习到的列X (与之前讨论的逻辑类似)
    num_learned_cols = sum(1 for x in col_x_estimates_regular if x is not None)
    if num_learned_cols < expected_regular_cols:
        logger.info(f"{log_prefix} 需要对 {expected_regular_cols - num_learned_cols} 个常规列X进行Fallback填充。")
        # (此处可以复用之前版本中更详细的Fallback逻辑，基于已知列和avg_obu_w外推，或均匀分布)
        # 简化版Fallback：如果至少有一个已知，则基于它和avg_obu_w推算
        if num_learned_cols > 0:
            first_known_idx = -1
            for idx, val in enumerate(col_x_estimates_regular):
                if val is not None: first_known_idx = idx; break

            for c_target in range(expected_regular_cols):
                if col_x_estimates_regular[c_target] is None:
                    col_x_estimates_regular[c_target] = col_x_estimates_regular[first_known_idx] + \
                                                        (c_target - first_known_idx) * (avg_obu_w * 1.1) # 估算列间距
        else: # 一列都未学到，则在图像中大致均匀分布
            all_cx_coords = [a['cx'] for a in yolo_detections_for_calib]
            min_cx_overall = min(all_cx_coords) if all_cx_coords else img_w * 0.1
            max_cx_overall = max(all_cx_coords) if all_cx_coords else img_w * 0.9
            dist_region_w = max_cx_overall - min_cx_overall
            eff_w_per_col = dist_region_w / expected_regular_cols if expected_regular_cols > 0 else avg_obu_w
            for c_target in range(expected_regular_cols):
                col_x_estimates_regular[c_target] = min_cx_overall + (c_target + 0.5) * eff_w_per_col

    if any(x is None for x in col_x_estimates_regular): # 最终检查
        logger.error(f"{log_prefix} 学习后 col_x_estimates_regular 仍含None。学习失败。")
        return None
    logger.info(f"{log_prefix} 常规列X估算: {[int(x) for x in col_x_estimates_regular]}")

    # 6. 生成一个初始的 row_y_estimates_initial_guess (用于调试或在无Y轴锚定时回退)
    #    这个初始猜测基于最前景的可靠行作为逻辑行0（如果特殊行不在顶部）或逻辑行1（如果特殊行在顶部）
    row_y_estimates_initial_guess = [0.0] * expected_total_rows
    if reliable_physical_rows:
        first_reliable_row_y = np.mean([a['cy'] for a in reliable_physical_rows[0]])
        initial_anchor_log_row = 0
        if special_row_at_logical_top and len(reliable_physical_rows[0]) != special_row_cols:
            initial_anchor_log_row = 1 # 如果特殊行在顶部，且第一行是常规行，则它是L1

        for r_log in range(expected_total_rows):
            row_y_estimates_initial_guess[r_log] = first_reliable_row_y + \
                                                   (r_log - initial_anchor_log_row) * avg_physical_row_height

    stable_params = {
        "avg_physical_row_height": avg_physical_row_height,
        "col_x_estimates_regular": col_x_estimates_regular,
        "avg_obu_w": avg_obu_w,
        "avg_obu_h": avg_obu_h,
        "special_row_at_logical_top": special_row_at_logical_top,
        "row_y_estimates_initial_guess": row_y_estimates_initial_guess # 存储这个初始猜测
    }
    logger.info(f"{log_prefix} 初始稳定布局参数学习成功。")
    return stable_params

class LayoutStateManager:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config # 从 config.py 加载的配置
        self.logger = logger
        # 可以从config中提取常用的布局参数
        self.expected_total_rows = config.get("LAYOUT_EXPECTED_TOTAL_ROWS", 13)
        self.regular_cols_count = config.get("LAYOUT_REGULAR_COLS_COUNT", 4)
        self.special_row_cols_count = config.get("LAYOUT_SPECIAL_ROW_COLS_COUNT", 2)

    def learn_initial_stable_layout(self,
                                    yolo_detections_for_calib: List[Dict[str, Any]],
                                    image_wh: Tuple[int, int],
                                    session_id: str) -> Optional[Dict[str, Any]]:
        """调用内部函数学习初始稳定布局参数。"""
        return _learn_initial_stable_layout_params(
            yolo_detections_for_calib, image_wh,
            { # 传递一个临时的session_config给学习函数
                "expected_total_rows": self.expected_total_rows,
                "regular_cols_count": self.regular_cols_count,
                "special_row_cols_count": self.special_row_cols_count
            },
            self.logger, session_id
        )

    def determine_y_axis_anchor(self,
                                current_frame_verified_obus: List[Dict[str, Any]], # {'text', 'physical_anchor', ...}
                                obu_evidence_pool: Dict[str, Any],
                                stable_layout_params: Dict[str, Any],
                                session_id: str,
                                current_frame_num: int
                                ) -> Tuple[Optional[Dict[str, Any]], List[float], bool]:
        """
        实现“Y轴2点定位法”，确定当前帧的Y轴锚定信息和动态行Y估算。
        Returns:
            Tuple: (y_anchor_info, current_dynamic_row_y_estimates, is_frame_skipped)
                   y_anchor_info: 锚定信息字典，或None
                   current_dynamic_row_y_estimates: 动态调整后的行Y估算列表
                   is_frame_skipped: 布尔值，指示是否因漏帧而跳过
        """
        log_prefix = f"会话 {session_id} (determine_y_axis_anchor):"
        is_frame_skipped = False
        y_anchor_info = None

        # 默认使用稳定参数中的初始Y估算或空列表
        current_dynamic_row_y_estimates = list(stable_layout_params.get("row_y_estimates_initial_guess", []))

        historical_overlaps_in_current_frame = []
        for obu_cur in current_frame_verified_obus:
            if obu_cur["text"] in obu_evidence_pool:
                # 只考虑那些在历史证据中已有确定逻辑坐标的OBU作为锚定候选
                hist_entry = obu_evidence_pool[obu_cur["text"]]
                if hist_entry.get("logical_coord") is not None:
                    historical_overlaps_in_current_frame.append({
                        "text": obu_cur["text"],
                        "current_physical_anchor": obu_cur["physical_anchor"],
                        "historical_logical_coord": hist_entry["logical_coord"]
                    })

        self.logger.info(f"{log_prefix} 在当前帧找到 {len(historical_overlaps_in_current_frame)} 个有效的历史重叠OBU用于Y轴锚定。")

        if current_frame_num > 1 and not historical_overlaps_in_current_frame:
            self.logger.error(f"{log_prefix} 检测到漏帧！当前帧 (第{current_frame_num}帧) 与历史无有效重叠OBU。")
            is_frame_skipped = True
            return None, current_dynamic_row_y_estimates, is_frame_skipped # 保持初始Y估算

        min_y_historical_obu_for_anchor = None
        if historical_overlaps_in_current_frame:
            min_y_historical_obu_for_anchor = min(
                historical_overlaps_in_current_frame,
                key=lambda o: o["current_physical_anchor"]["cy"]
            )

        if min_y_historical_obu_for_anchor:
            y_ref_current = min_y_historical_obu_for_anchor["current_physical_anchor"]["cy"]
            l_ref = min_y_historical_obu_for_anchor["historical_logical_coord"][0] # row
            avg_row_h = stable_layout_params.get("avg_physical_row_height")

            if avg_row_h is None or avg_row_h <= 0:
                self.logger.warning(f"{log_prefix} 稳定参数中的平均行高无效 ({avg_row_h})，无法精确Y轴锚定。")
            else:
                current_dynamic_row_y_estimates = [
                    y_ref_current + (i - l_ref) * avg_row_h for i in range(self.expected_total_rows)
                ]
                y_anchor_info = {
                    "ref_obu_text": min_y_historical_obu_for_anchor['text'],
                    "ref_logical_row": l_ref,
                    "ref_physical_y_current": y_ref_current
                }
                self.logger.info(f"{log_prefix} Y轴锚定成功: 参照OBU '{y_anchor_info['ref_obu_text']}' "
                                 f"(历史逻辑行 {l_ref}) 在当前帧物理Y={y_ref_current:.0f}。")
                self.logger.info(f"  更新后动态行Y估算: {[int(y) for y in current_dynamic_row_y_estimates]}")
        elif current_frame_num > 1: # 后续帧，但没有找到Y轴锚定参照物
            self.logger.warning(f"{log_prefix} 后续帧未能找到Y轴锚定参照物。将使用基于稳定参数的初始Y估算。")
            # current_dynamic_row_y_estimates 已设为初始猜测

        return y_anchor_info, current_dynamic_row_y_estimates, is_frame_skipped

    def _map_single_anchor_to_logical_using_params(
        self, anchor_to_map: Dict[str, Any], # 单个YOLO锚点
        row_y_estimates_map: List[float],
        col_x_estimates_regular_map: List[float],
        stable_layout_params_map: Dict[str, Any], # 需要avg_obu_w, avg_obu_h, special_row_at_logical_top
        logger_map: Any,
        session_id_for_log_map: str,
        log_prefix_map: str = "(_map_single_anchor)"
    ) -> Optional[Tuple[int, int]]:
        """辅助函数：使用给定参数将单个锚点映射到逻辑坐标。"""
        # (这个函数是之前 _map_anchors_to_logical_using_params 的单锚点版本简化)
        if not row_y_estimates_map or not col_x_estimates_regular_map or \
           any(x is None for x in col_x_estimates_regular_map):
            # logger_map.debug(f"{log_prefix_map} 会话 {session_id_for_log_map}: 映射单个锚点时参数不足。")
            return None

        avg_row_h = stable_layout_params_map.get("avg_physical_row_height", 50)
        avg_obu_w = stable_layout_params_map.get("avg_obu_w", 100)
        y_match_threshold = avg_row_h * self.config.get("LAYOUT_Y_MATCH_THRESHOLD_FACTOR", 0.75)
        x_match_threshold = avg_obu_w * self.config.get("LAYOUT_X_MATCH_THRESHOLD_FACTOR", 0.75)
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

        if not cols_to_match_xs: return None

        cand_c_in_options = -1; min_x_d_sq = float('inf')
        for c_opt_idx, est_x in enumerate(cols_to_match_xs):
            dist_x_sq = (anchor_to_map['cx'] - est_x)**2
            if dist_x_sq < min_x_d_sq and dist_x_sq < x_match_threshold**2:
                min_x_d_sq = dist_x_sq; cand_c_in_options = c_opt_idx

        if cand_c_in_options != -1:
            final_c = cand_c_in_options
            if is_special_row and self.special_row_cols_count == 2 and self.regular_cols_count == 4:
                final_c = cand_c_in_options + 1 # 0->1, 1->2
            if 0 <= final_c < self.regular_cols_count:
                return (cand_r, final_c)
        return None

    def update_session_state_with_reference_logic(
        self,
        session_data: Dict[str, Any], # 包含 obu_evidence_pool, logical_matrix, etc.
        current_frame_verified_obus: List[Dict[str, Any]], # 当前帧已校验OBU (含physical_anchor)
        current_frame_all_yolo_detections: List[Dict[str, Any]], # 当前帧所有YOLO检测 (用于标记失败)
        current_frame_ocr_results: List[Dict[str, Any]], # 当前帧所有OCR结果 (用于标记失败)
        y_anchor_info: Optional[Dict[str, Any]],
        current_dynamic_row_y_estimates: List[float],
        stable_layout_params: Dict[str, Any],
        session_id: str,
        current_frame_num: int
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], str], List[Dict[str, Any]]]:
        """
        核心状态更新逻辑：参照物优先，100%信任历史，Y轴锚定，漏帧处理。
        直接修改 session_data 中的 obu_evidence_pool, logical_matrix, recognized_texts_map。
        返回更新后的 logical_matrix, recognized_texts_map 和警告。
        """
        log_prefix = f"会话 {session_id} (update_state_ref_logic F{current_frame_num}):"
        self.logger.info(f"{log_prefix} 开始核心状态更新...")

        obu_evidence_pool = session_data["obu_evidence_pool"]
        logical_matrix = session_data["logical_matrix"] # 直接修改
        recognized_texts_map = session_data["recognized_texts_map"] # 直接修改
        warnings = []

        # 1. 更新 obu_evidence_pool (历史OBU保持逻辑坐标，新OBU待定)
        for obu_cur_detail in current_frame_verified_obus:
            text = obu_cur_detail["text"]
            phys_anchor = obu_cur_detail["physical_anchor"]
            ocr_conf = obu_cur_detail.get("ocr_confidence", 0.0)

            if text in obu_evidence_pool: # 历史OBU
                obu_evidence_pool[text]["physical_anchors"] = [phys_anchor] # 更新物理锚点
                obu_evidence_pool[text]["ocr_confidence"] = max(obu_evidence_pool[text].get("ocr_confidence",0.0), ocr_conf)
                obu_evidence_pool[text]["last_seen_frame"] = current_frame_num
                # logical_coord 保持不变 (100%信任)
            else: # 新OBU
                obu_evidence_pool[text] = {
                    "physical_anchors": [phys_anchor], "ocr_confidence": ocr_conf,
                    "logical_coord": None, # 待分配
                    "first_seen_frame": current_frame_num, "last_seen_frame": current_frame_num
                }
        self.logger.info(f"{log_prefix} OBU证据池更新完毕。总数: {len(obu_evidence_pool)}")

        # 2. 为新OBU (logical_coord is None) 分配逻辑坐标 (参照物优先)
        # 2a. 对当前帧所有已验证OBU进行物理行分组
        if current_frame_verified_obus:
            sorted_current_obus_by_y = sorted(current_frame_verified_obus, key=lambda o: (o["physical_anchor"]["cy"], o["physical_anchor"]["cx"]))
            avg_h_group = stable_layout_params.get("avg_physical_row_height", 50) * LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR

            current_frame_physical_rows = []
            if sorted_current_obus_by_y:
                _row_grp = [sorted_current_obus_by_y[0]]
                for i in range(1, len(sorted_current_obus_by_y)):
                    if abs(sorted_current_obus_by_y[i]["physical_anchor"]["cy"] - _row_grp[-1]["physical_anchor"]["cy"]) < avg_h_group:
                        _row_grp.append(sorted_current_obus_by_y[i])
                    else:
                        current_frame_physical_rows.append(sorted(_row_grp, key=lambda o: o["physical_anchor"]["cx"]))
                        _row_grp = [sorted_current_obus_by_y[i]]
                if _row_grp: current_frame_physical_rows.append(sorted(_row_grp, key=lambda o: o["physical_anchor"]["cx"]))

            # 2b. 遍历物理行，尝试为新OBU定位
            for physical_row in current_frame_physical_rows:
                ref_obus_in_this_row = [o for o in physical_row if o["text"] in obu_evidence_pool and \
                                        obu_evidence_pool[o["text"]]["logical_coord"] is not None]

                determined_logical_r_for_row = None
                if ref_obus_in_this_row: # 行内有参照物
                    # 简单策略：用第一个参照物的逻辑行号 (假设同一行内参照物逻辑行号一致)
                    determined_logical_r_for_row = obu_evidence_pool[ref_obus_in_this_row[0]["text"]]["logical_coord"][0]

                if determined_logical_r_for_row is not None: # 使用参照物定位列
                    for obu_in_phys_row in physical_row:
                        if obu_in_phys_row["text"] in obu_evidence_pool and \
                           obu_evidence_pool[obu_in_phys_row["text"]]["logical_coord"] is None: # 是新OBU
                            # 使用稳定的列X参数和该行的逻辑行号来推断列
                            mapped_coords = self._map_single_anchor_to_logical_using_params(
                                obu_in_phys_row["physical_anchor"],
                                [current_dynamic_row_y_estimates[determined_logical_r_for_row]], # 只用确定的这一行的Y
                                stable_layout_params["col_x_estimates_regular"],
                                stable_layout_params, self.logger, session_id,
                                log_prefix_map=f"{log_prefix} (RefMapForRow{determined_logical_r_for_row})"
                            )
                            if mapped_coords and mapped_coords[0] == determined_logical_r_for_row: # 确保行号一致
                                obu_evidence_pool[obu_in_phys_row["text"]]["logical_coord"] = (mapped_coords[0], mapped_coords[1])
                                self.logger.info(f"{log_prefix} (参照物定位) OBU '{obu_in_phys_row['text']}' 定位到 {mapped_coords}")
                # else: # 行内无参照物，这些OBU将尝试在下一步全局定位中处理 (如果它们是新的)
                    # self.logger.debug(f"{log_prefix} 物理行 (首OBU Y:{physical_row[0]['physical_anchor']['cy']}) 无参照物。")
                    pass

        # 2c. 对剩余未定位的新OBU，尝试使用Y轴锚定后的全局参数进行定位
        for obu_text, evidence in obu_evidence_pool.items():
            if evidence["logical_coord"] is None: # 仍然未定位
                mapped_coords_global = self._map_single_anchor_to_logical_using_params(
                    evidence["physical_anchors"][-1], # 使用最新的物理锚点
                    current_dynamic_row_y_estimates, # 使用Y轴锚定后的动态行Y
                    stable_layout_params["col_x_estimates_regular"],
                    stable_layout_params, self.logger, session_id,
                    log_prefix_map=f"{log_prefix} (GlobalFallbackMap)"
                )
                if mapped_coords_global:
                    evidence["logical_coord"] = mapped_coords_global
                    self.logger.info(f"{log_prefix} (全局参数定位) OBU '{obu_text}' 定位到 {mapped_coords_global}")
                else:
                    self.logger.warning(f"{log_prefix} (全局参数定位) OBU '{obu_text}' 未能定位。")

        # 3. 构建最终输出矩阵
        # 清空矩阵 (保留-1)
        for r in range(self.expected_total_rows):
            for c in range(self.regular_cols_count): # 假设矩阵是按常规列数创建的
                if logical_matrix[r][c] != -1:
                    logical_matrix[r][c] = 0
                    if (r, c) in recognized_texts_map:
                        del recognized_texts_map[(r, c)]

        # 填充矩阵，解决冲突 (OCR分高的优先)
        # 按OCR置信度降序排列证据池中的OBU，确保高分先占位
        sorted_evidence = sorted(
            [(text, evi) for text, evi in obu_evidence_pool.items() if evi.get("logical_coord") is not None],
            key=lambda item: item[1].get("ocr_confidence", 0.0),
            reverse=True
        )

        for obu_text_fill, evidence_fill in sorted_evidence:
            r_fill, c_fill = evidence_fill["logical_coord"]
            if 0 <= r_fill < self.expected_total_rows and 0 <= c_fill < self.regular_cols_count:
                if logical_matrix[r_fill][c_fill] == 0: # 如果坑位是空的
                    logical_matrix[r_fill][c_fill] = 1
                    recognized_texts_map[(r_fill, c_fill)] = obu_text_fill
                elif logical_matrix[r_fill][c_fill] == 1: # 如果坑位已被占据
                    # （由于按OCR分排序，理论上这里不应该发生更高分被低分覆盖的情况，除非是同一OBU多次出现且逻辑坐标变了，但我们已信任历史坐标）
                    # 但如果不同OBU映射到同一位置，高分的会先占据。
                    self.logger.warning(f"{log_prefix} 矩阵填充冲突: 坑位 ({r_fill},{c_fill}) 已被 "
                                     f"'{recognized_texts_map.get((r_fill,c_fill))}' 占据，无法放入 '{obu_text_fill}'。")
            else:
                 self.logger.warning(f"{log_prefix} OBU '{obu_text_fill}' 的逻辑坐标 ({r_fill},{c_fill}) 超出矩阵范围。")

        num_filled_final = sum(1 for r in logical_matrix for status in r if status == 1)
        self.logger.info(f"{log_prefix} 最终矩阵构建完成，共填充 {num_filled_final} 个OBU。")

        # 4. 标记OCR失败或无效的格子 (使用当前帧所有YOLO检测和OCR结果)
        #    这里需要一个当前帧YOLO到逻辑坐标的初步映射（即使不完美）
        #    这个初步映射可以简单地通过 _map_single_anchor_to_logical_using_params 对所有当前帧YOLO框进行一次
        ocr_results_map_by_idx = {ocr.get("original_index"): ocr for ocr in current_frame_ocr_results if ocr}

        for yolo_det in current_frame_all_yolo_detections:
            original_idx = yolo_det["original_index"]
            # 为这个YOLO框获取一个初步的逻辑位置估计
            prelim_coords = self._map_single_anchor_to_logical_using_params(
                yolo_det, # 包含cx,cy,w,h
                current_dynamic_row_y_estimates,
                stable_layout_params["col_x_estimates_regular"],
                stable_layout_params, self.logger, session_id,
                log_prefix_map=f"{log_prefix} (MarkFailMap)"
            )
            if prelim_coords:
                r_log_fail, c_log_fail = prelim_coords
                if 0 <= r_log_fail < self.expected_total_rows and \
                   0 <= c_log_fail < self.regular_cols_count and \
                   logical_matrix[r_log_fail][c_log_fail] == 0: # 只标记之前是“未知”的

                    ocr_item = ocr_results_map_by_idx.get(original_idx)
                    is_ocr_valid_and_in_db = False
                    if ocr_item:
                        ocr_text_check = ocr_item.get("ocr_final_text", "")
                        if ocr_text_check in VALID_OBU_CODES:
                            is_ocr_valid_and_in_db = True

                    if not is_ocr_valid_and_in_db:
                        logical_matrix[r_log_fail][c_log_fail] = 2 # 标记为识别失败

        self.logger.info(f"{log_prefix} 核心状态更新完成。")
        return logical_matrix, recognized_texts_map, warnings