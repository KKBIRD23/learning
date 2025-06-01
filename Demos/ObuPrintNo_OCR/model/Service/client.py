# D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\Service\client.py
import requests
import os

# --- 配置 ---
# 服务端地址和端口
SERVER_URL = "http://127.0.0.1:5000/predict"

# 要上传的图片路径
# !!! 请根据您的实际图片路径修改这里 !!!
# 例如，如果 3.jpg 在 D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\DATA\PIC\3.jpg
# 并且 client.py 在 D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\
# 那么相对路径可以是 r"..\DATA\PIC\3.jpg"
# 或者使用绝对路径 r"D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\DATA\PIC\3.jpg"
IMAGE_PATH_TO_UPLOAD = r"..\..\..\DATA\PIC\2.jpg" # 假设 3.jpg 在上一级目录的 DATA/PIC/ 下

# --- 主函数 ---
def send_image_for_prediction(image_path):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return

    try:
        # 以二进制读取模式打开图片文件
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')} # 'image/jpeg' 可以根据实际图片类型调整

            print(f"客户端：正在向 {SERVER_URL} 发送图片 '{os.path.basename(image_path)}'...")
            response = requests.post(SERVER_URL, files=files, timeout=30) # 设置30秒超时

            # 检查HTTP响应状态码
            if response.status_code == 200:
                print("客户端：成功接收到服务端响应:")
                print(response.json()) # 以JSON格式打印响应内容
            else:
                print(f"客户端：服务端返回错误状态码 {response.status_code}:")
                try:
                    print(response.json()) # 尝试打印JSON错误信息
                except requests.exceptions.JSONDecodeError:
                    print(response.text) # 如果不是JSON，打印原始文本

    except requests.exceptions.RequestException as e:
        print(f"客户端：请求过程中发生错误: {e}")
    except FileNotFoundError:
        print(f"客户端错误：无法打开图片文件 - {image_path}")
    except Exception as e:
        print(f"客户端：发生未知错误: {e}")

if __name__ == "__main__":
    send_image_for_prediction(IMAGE_PATH_TO_UPLOAD)