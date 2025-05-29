import cv2
import numpy as np
import onnxruntime
import os
import time # For timing
import traceback # For detailed error reporting

# --- V1.0 Configuration ---
ONNX_MODEL_PATH = r"./model/yolov8/yolov8_barcode_detection.onnx"
IMAGE_NAME = r"./3.jpg"
CONFIDENCE_THRESHOLD = 0.1  # User-defined: Low threshold, might detect more, potentially more false positives
IOU_THRESHOLD = 0.1         # User-defined: Low threshold for NMS, might keep more overlapping boxes

# COCO Dataset Classes (if using a COCO model)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- Timing Profile Dictionary ---
timing_profile_v1 = {}

# --- Helper Functions ---
def preprocess_image_v1(image_path, input_model_shape_hw):
    """
    Preprocesses an image for ONNX model inference.
    Loads, resizes, pads, normalizes, and transposes the image.
    Returns:
        input_tensor: Preprocessed image tensor for the model.
        original_img: The original image loaded by OpenCV.
        scale_ratio: The ratio used to scale the image.
        pad_x: Horizontal padding added.
        pad_y: Vertical padding added.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_h_orig, img_w_orig = original_img.shape[:2]
    model_input_h, model_input_w = input_model_shape_hw

    # Calculate new dimensions maintaining aspect ratio
    scale_ratio = min(model_input_w / img_w_orig, model_input_h / img_h_orig)
    new_w = int(img_w_orig * scale_ratio)
    new_h = int(img_h_orig * scale_ratio)

    resized_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a canvas and place the resized image on it
    canvas = np.full((model_input_h, model_input_w, 3), 128, dtype=np.uint8) # Gray padding
    pad_x = (model_input_w - new_w) // 2
    pad_y = (model_input_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    # Convert HWC to CHW, normalize, and add batch dimension
    input_tensor = canvas.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
    input_tensor /= 255.0  # Normalize to [0, 1]
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    return input_tensor, original_img, scale_ratio, pad_x, pad_y

def non_max_suppression_v1(boxes_np, scores_np, iou_threshold):
    """
    Performs Non-Maximum Suppression.
    Args:
        boxes_np (np.array): Array of bounding boxes (x1, y1, x2, y2).
        scores_np (np.array): Array of confidence scores.
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        list: Indices of boxes to keep.
    """
    if len(boxes_np) == 0:
        return []

    # Sort by score in descending order
    indices = np.argsort(scores_np)[::-1]

    keep_indices = []
    while len(indices) > 0:
        current_idx = indices[0]
        keep_indices.append(current_idx)

        if len(indices) == 1:
            break

        current_box = boxes_np[current_idx]
        remaining_indices = indices[1:]
        remaining_boxes = boxes_np[remaining_indices]

        # Calculate IoU
        x1_max = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1_max = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2_min = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2_min = np.minimum(current_box[3], remaining_boxes[:, 3])

        inter_area = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        areas_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                          (remaining_boxes[:, 3] - remaining_boxes[:, 1])

        union_area = area_current + areas_remaining - inter_area

        iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area, dtype=float), where=union_area != 0)

        # Keep boxes with IoU less than or equal to the threshold
        indices_to_retain = np.where(iou <= iou_threshold)[0]
        indices = remaining_indices[indices_to_retain]

    return keep_indices

def postprocess_output_v1(model_outputs, original_img_shape_hw, model_input_shape_hw,
                          scale_ratio, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    Postprocesses model output to get final bounding boxes, scores, and class IDs.
    """
    # Squeeze batch dimension, model_outputs[0] is typically the main detection output
    # Expected shape: (batch_size, num_channels, num_proposals) -> (num_channels, num_proposals)
    predictions_raw = np.squeeze(model_outputs[0])

    num_output_features = predictions_raw.shape[0] # e.g., 6 for (cx,cy,w,h,conf,class_id_or_other) or 84 for COCO
    # num_proposals = predictions_raw.shape[1] # e.g., 8400

    # Transpose so each row is a proposal: (num_proposals, num_output_features)
    transposed_predictions = predictions_raw.transpose()

    boxes_candidate = []
    scores_candidate = []
    class_ids_candidate = []

    for pred_data in transposed_predictions:
        # pred_data contains [cx, cy, w, h, ...class_info...]
        cx, cy, w, h = pred_data[:4]

        confidence = 0.0
        class_id = -1

        if num_output_features == 6: # Assuming your barcode model: [cx,cy,w,h, conf, class_id_explicit_or_fixed_to_0]
            confidence = pred_data[4]
            # If class_id is explicitly output by model at index 5:
            # class_id = int(pred_data[5])
            # If it's a single-class (barcode) model where class_id is always 0:
            class_id = 0
        elif num_output_features > 4: # General YOLO (e.g., COCO: 4 bbox_coords + 80 class_scores)
            class_scores = pred_data[4:]
            confidence = np.max(class_scores)
            class_id = np.argmax(class_scores)
        else:
            # print(f"Warning: Unsupported model output features: {num_output_features}")
            continue

        if confidence >= conf_thresh:
            # Convert cx,cy,w,h to x1,y1,x2,y2 (relative to model input size)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2])
            scores_candidate.append(confidence)
            class_ids_candidate.append(class_id)

    if not boxes_candidate:
        # print(f"No candidate boxes found above confidence threshold {conf_thresh:.2f}.")
        return [], [], []

    boxes_np = np.array(boxes_candidate)
    scores_np = np.array(scores_candidate)

    # Perform Non-Maximum Suppression
    t_nms_start = time.time()
    keep_indices = non_max_suppression_v1(boxes_np, scores_np, iou_thresh)
    timing_profile_v1['4a_nms_in_postprocess'] = time.time() - t_nms_start

    final_boxes = []
    final_scores = []
    final_class_ids = []

    orig_h, orig_w = original_img_shape_hw

    for index in keep_indices:
        box = boxes_np[index].copy() # Original box is relative to model input canvas (e.g., 640x640)

        # Scale back to original image coordinates
        # 1. Remove padding
        box_no_pad_x1 = box[0] - pad_x
        box_no_pad_y1 = box[1] - pad_y
        box_no_pad_x2 = box[2] - pad_x
        box_no_pad_y2 = box[3] - pad_y

        # 2. Rescale by inverse of scale_ratio
        orig_x1 = box_no_pad_x1 / scale_ratio
        orig_y1 = box_no_pad_y1 / scale_ratio
        orig_x2 = box_no_pad_x2 / scale_ratio
        orig_y2 = box_no_pad_y2 / scale_ratio

        # 3. Clip to original image boundaries
        final_x1 = np.clip(orig_x1, 0, orig_w)
        final_y1 = np.clip(orig_y1, 0, orig_h)
        final_x2 = np.clip(orig_x2, 0, orig_w)
        final_y2 = np.clip(orig_y2, 0, orig_h)

        final_boxes.append([int(final_x1), int(final_y1), int(final_x2), int(final_y2)])
        final_scores.append(scores_np[index])
        final_class_ids.append(class_ids_candidate[index])

    return final_boxes, final_scores, final_class_ids

def draw_detections_v1(image, boxes, scores, class_ids, class_names_list=None):
    """Draws detection results on the image."""
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[i]
        class_id = class_ids[i]

        label_name = f"ClassID:{class_id}"
        if class_names_list and 0 <= class_id < len(class_names_list):
            label_name = class_names_list[class_id]

        label_text = f"{label_name}: {score:.2f}"
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
        cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_out

# --- Main Program (V1.0) ---
if __name__ == "__main__":
    t_total_start = time.time()
    timing_profile_v1['0_total_script_execution'] = 0 # Initialize

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model file not found: {ONNX_MODEL_PATH}"); exit()
    if not os.path.exists(IMAGE_NAME):
        print(f"Error: Test image not found: {IMAGE_NAME}"); exit()

    try:
        # 1. Load ONNX model
        t_start = time.time()
        print(f"--- V1.0: Initializing and Loading Model ---")
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        timing_profile_v1['1_model_loading'] = time.time() - t_start
        print(f"Model loaded ({timing_profile_v1['1_model_loading']:.3f} sec)")

        # 2. Get model input/output details
        input_cfg = session.get_inputs()[0]
        output_cfg = session.get_outputs()[0]
        input_name = input_cfg.name
        model_input_shape_onnx = input_cfg.shape  # e.g., [1, 3, 640, 640]
        model_h, model_w = model_input_shape_onnx[2], model_input_shape_onnx[3]

        # This is the number of features per prediction box (e.g. 4 coords + 1 conf + 1 class_id = 6)
        model_num_output_features = output_cfg.shape[1]
        print(f"Model Input: {input_name} {model_input_shape_onnx}, Output Features: {model_num_output_features}")

        # 3. Determine class names
        class_names = None
        if model_num_output_features == 6: # Specific to your barcode model
            class_names = ["Barcode"]
            print("Barcode model (6 output features) detected.")
        elif model_num_output_features == (4 + len(COCO_CLASSES)):
            class_names = COCO_CLASSES
            print(f"COCO model ({len(COCO_CLASSES)} classes, {model_num_output_features} output features) detected.")
        elif model_num_output_features > 4:
            num_derived_classes = model_num_output_features - 4
            class_names = [f"Class_{i}" for i in range(num_derived_classes)]
            print(f"Custom model ({num_derived_classes} classes, {model_num_output_features} output features) detected.")
        else:
            class_names = []
            print(f"Warning: Could not determine class names for output features: {model_num_output_features}")

        # 4. Preprocess image
        t_start = time.time()
        print(f"--- V1.0: Processing Image: {IMAGE_NAME} ---")
        input_tensor, original_image, scale_r, p_x, p_y = preprocess_image_v1(
            IMAGE_NAME, (model_h, model_w)
        )
        timing_profile_v1['2_image_preprocessing'] = time.time() - t_start
        print(f"Image preprocessed ({timing_profile_v1['2_image_preprocessing']:.3f} sec). "
              f"Original: {original_image.shape[:2]}, Model Input: ({model_h},{model_w}), "
              f"Scale: {scale_r:.2f}, Pad: ({p_x},{p_y})")

        # 5. Run inference
        t_start = time.time()
        print(f"--- V1.0: Running Inference ---")
        model_outputs_raw = session.run(None, {input_name: input_tensor})
        timing_profile_v1['3_model_inference'] = time.time() - t_start
        print(f"Inference complete ({timing_profile_v1['3_model_inference']:.3f} sec).")
        # print(f"Raw output count: {len(model_outputs_raw)}, First output shape: {model_outputs_raw[0].shape}")


        # 6. Postprocess output
        t_start = time.time()
        print(f"--- V1.0: Postprocessing Output ---")
        final_boxes, final_scores, final_class_ids = postprocess_output_v1(
            model_outputs_raw,
            original_image.shape[:2], # Pass H, W
            (model_h, model_w),
            scale_r,
            p_x,
            p_y,
            CONFIDENCE_THRESHOLD,
            IOU_THRESHOLD
        )
        # Note: NMS time is captured inside postprocess_output_v1 and stored in '4a_nms_in_postprocess'
        timing_profile_v1['4_postprocessing_total'] = time.time() - t_start
        print(f"Postprocessing complete ({timing_profile_v1['4_postprocessing_total']:.3f} sec).")


        # 7. Visualize and save results
        t_start = time.time()
        if final_boxes:
            print(f"--- V1.0: Detected {len(final_boxes)} objects ---")
            for i, box_coords in enumerate(final_boxes):
                class_id_val = final_class_ids[i]
                class_name_str = f"ClassID:{class_id_val}"
                if class_names and 0 <= class_id_val < len(class_names):
                    class_name_str = class_names[class_id_val]
                print(f"  Object {i+1}: Class='{class_name_str}', Box={box_coords}, Score={final_scores[i]:.2f}")

            output_img_v1 = draw_detections_v1(original_image, final_boxes, final_scores, final_class_ids, class_names)

            base_name, ext = os.path.splitext(os.path.basename(IMAGE_NAME))
            output_image_filename = f"output_{base_name}_v1.0{ext}"

            cv2.imwrite(output_image_filename, output_img_v1)
            print(f"Output image saved to: {output_image_filename}")

            # Uncomment to display image
            # cv2.imshow("Detected Objects (V1.0)", output_img_v1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("--- V1.0: No objects detected with current settings. ---")
        timing_profile_v1['5_drawing_and_saving'] = time.time() - t_start
        print(f"Drawing and saving took {timing_profile_v1['5_drawing_and_saving']:.3f} sec.")


    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as general_error:
        print(f"An error occurred: {general_error}")
        traceback.print_exc()
    finally:
        timing_profile_v1['0_total_script_execution'] = time.time() - t_total_start
        print(f"\n--- V1.0: Timing Profile ---")
        # Sort profile by key for consistent output order
        sorted_timing_profile_v1 = {k: timing_profile_v1[k] for k in sorted(timing_profile_v1.keys())}
        for stage, duration in sorted_timing_profile_v1.items():
            print(f"  {stage}: {duration:.3f} sec")
        print(f"----------------------------")