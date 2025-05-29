import paddle.inference as paddle_infer
import cv2
import numpy as np
import os

# 加载模型
def load_predictor(model_dir):
    model_file = os.path.join(model_dir, "inference.json")
    params_file = os.path.join(model_dir, "inference.pdiparams")
    config = paddle_infer.Config(model_file, params_file)
    config.disable_gpu()  # 如使用 GPU，请注释此行
    config.switch_use_feed_fetch_ops(False)
    predictor = paddle_infer.create_predictor(config)
    return predictor

# 图像预处理函数（根据模型要求进行调整）
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # 根据模型的预处理要求进行处理，例如缩放、归一化等
    # 以下为示例，具体需根据模型的 inference.yml 配置进行调整
    image = cv2.resize(image, (960, 960))
    image = image.astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)
    return image

# 推理函数
def run_inference(predictor, input_data):
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(input_data)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    return output_data

# 主函数
def main():
    # 使用绝对路径
    det_model_dir = r"C:\Users\KKBIRD\Desktop\photo\yolo_model\model\PaddleOCR\PP-OCRv5_server_det_infer"
    rec_model_dir = r"C:\Users\KKBIRD\Desktop\photo\yolo_model\model\PaddleOCR\PP-OCRv5_server_rec_infer"

    # 加载模型
    det_predictor = load_predictor(det_model_dir)
    rec_predictor = load_predictor(rec_model_dir)

    # 读取并预处理图像
    image_path = r"C:\Users\KKBIRD\Desktop\photo\yolo_model\PIC\3.jpg"
    input_data = preprocess_image(image_path)

    # 运行检测模型
    det_output = run_inference(det_predictor, input_data)
    # 根据检测结果进行后处理，提取文本框等

    # 对每个文本框区域进行识别
    # 这里需要根据检测结果裁剪图像区域，并进行预处理后输入识别模型
    # 以下为示例，具体实现需根据实际情况编写
    # for box in det_boxes:
    #     cropped_image = crop_image(image, box)
    #     rec_input = preprocess_image(cropped_image)
    #     rec_output = run_inference(rec_predictor, rec_input)
    #     # 解析识别结果

if __name__ == "__main__":
    main()
