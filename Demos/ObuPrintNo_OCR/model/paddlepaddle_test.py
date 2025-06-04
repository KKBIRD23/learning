import paddle
import paddleocr
import os # 用于测试OCR时检查文件是否存在
import traceback

# 检查paddlepaddle版本和是否能用CPU
print(f"PaddlePaddle Version: {paddle.__version__}")
print(f"PaddlePaddle Compiled with CUDA: {paddle.is_compiled_with_cuda()}")
try:
    current_device = paddle.get_device()
    print(f"PaddlePaddle Device in use: {current_device}")
except Exception as e:
    print(f"Could not get paddle device info directly: {e}. Assuming CPU if not compiled with CUDA.")
    if not paddle.is_compiled_with_cuda():
        print("PaddlePaddle is likely using CPU.")

# 尝试初始化PaddleOCR (第一次运行时会自动下载模型)
print("Initializing PaddleOCR...")
ocr_engine = None # 先声明
try:
    ocr_engine = paddleocr.PaddleOCR(lang='en')
    print("PaddleOCR initialized successfully!")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    traceback.print_exc()
    exit()

# (可选) 进行一次简单的识别测试
if ocr_engine:
    test_image_content = "123 ABC"
    img_path_test = r"D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\model\PaddleOCR\3.jpg"

    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (200, 50), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            print("Arial.ttf not found, using default Pillow font.")
            font = ImageFont.load_default()

        if hasattr(d, 'textbbox'):
            text_bbox = d.textbbox((0,0), test_image_content, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            text_width, text_height = d.textsize(test_image_content, font=font)

        d.text(((200-text_width)/2, (50-text_height)/2), test_image_content, fill=(0,0,0), font=font)
        img.save(img_path_test)
        print(f"Created a test image: {img_path_test}")

        if os.path.exists(img_path_test):
            print(f"Performing OCR on {img_path_test}...")
            try:
                # PaddleOCR 3.0.0 ocr() 方法调用
                result = ocr_engine.ocr(img_path_test)

                # 解析返回结果
                # 对于单张图片，result 是一个列表，其第一个元素 result[0] 是一个包含详细信息的字典
                if result and isinstance(result, list) and len(result) > 0 and \
                   isinstance(result[0], dict):

                    print("OCR Result (raw dictionary for first image):", result[0])

                    image_result_dict = result[0]

                    # 从字典中提取识别出的文本列表
                    # .get() 方法可以在键不存在时返回一个默认值 (这里是空列表)
                    extracted_texts_list = image_result_dict.get('rec_texts', [])
                    # rec_scores_list = image_result_dict.get('rec_scores', []) # 如果需要置信度

                    if extracted_texts_list:
                        # 将所有识别出的文本片段连接起来
                        full_extracted_text = " ".join(extracted_texts_list)
                        print(f"Test OCR extracted text: '{full_extracted_text.strip()}'")
                    else:
                        print("Test OCR: 'rec_texts' key not found in result dictionary or no text recognized.")

                elif result and isinstance(result, list) and len(result) > 0 and result[0] is None:
                    print("Test OCR produced a result structure where the first image's result is None.")
                else:
                   print(f"Test OCR produced an unexpected result structure: {type(result)}")
                   if result: print(f"Content: {result}")

            except TypeError as te:
                print(f"TypeError during OCR call: {te}")
                print("This might indicate an issue with how ocr() is called or its expected arguments in this version.")
                traceback.print_exc()
            except Exception as ocr_test_e:
                print(f"Error during test OCR: {ocr_test_e}")
                traceback.print_exc()
        else:
            print(f"Test image {img_path_test} not found, skipping OCR test.")

    except ImportError:
        print("Pillow (PIL) library not found. Cannot create dynamic test image. Skipping OCR test.")
    except Exception as general_e:
        print(f"An error occurred in the test image creation or OCR section: {general_e}")
        traceback.print_exc()
else:
    print("PaddleOCR engine initialization failed. Skipping OCR test.")