# test_paddleocr_init.py
import paddleocr
import cv2
import numpy as np
import os
import traceback

print(f"PaddleOCR version: {paddleocr.__version__}")

# --- 初始化方式 ---
paddle_ocr_engine = None
init_method_tried = "PaddleOCR with PP-OCRv5 params"
ocr_params = {
    'lang': 'en',
    'use_textline_orientation': False,
    'use_doc_orientation_classify': False, # From official docs
    'use_doc_unwarping': False,           # From official docs
}

try:
    print(f"\nAttempting PaddleOCR initialization with: {init_method_tried}")
    print(f"Initializing with params: {ocr_params}")
    paddle_ocr_engine = paddleocr.PaddleOCR(**ocr_params)
    print("PaddleOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR engine with {init_method_tried} and params {ocr_params}: {e}")
    traceback.print_exc()
    paddle_ocr_engine = None


if paddle_ocr_engine:
    print("\n--- PaddleOCR engine is available. ---")

    print("\n--- Main Test: Real Image ---")
    real_image_path = "3.jpg" # Or 2.jpg, 3.jpg - the large image
    if os.path.exists(real_image_path):
        real_image_data = cv2.imread(real_image_path)
        if real_image_data is not None:
            try:
                print(f"Running predict() on the REAL image: {real_image_path} ...")
                result_prediction = paddle_ocr_engine.predict(real_image_data)

                if result_prediction and len(result_prediction) > 0 and result_prediction[0] is not None:
                    ocr_result_data = result_prediction[0] # This is the OCRResult object, which should act like a dict

                    print(f"\nDEBUG: Type of ocr_result_data (result_prediction[0]): {type(ocr_result_data)}")

                    # --- Accessing data assuming ocr_result_data is the dictionary-like object ---
                    # It should contain keys like 'dt_polys', 'rec_texts', 'rec_scores' directly

                    print("\nAttempting to access data using dictionary-like .get() method:")

                    dt_polys = ocr_result_data.get('dt_polys')
                    rec_texts = ocr_result_data.get('rec_texts')
                    rec_scores = ocr_result_data.get('rec_scores')

                    # Verify if data was retrieved
                    data_retrieved = False
                    if dt_polys is not None:
                        print(f"  'dt_polys' RETRIEVED. Length: {len(dt_polys)}")
                        data_retrieved = True
                    else:
                        print("  'dt_polys' NOT RETRIEVED (returned None).")

                    if rec_texts is not None:
                        print(f"  'rec_texts' RETRIEVED. Length: {len(rec_texts)}")
                        data_retrieved = True
                    else:
                        print("  'rec_texts' NOT RETRIEVED (returned None).")

                    if rec_scores is not None:
                        print(f"  'rec_scores' RETRIEVED. Length: {len(rec_scores)}")
                        data_retrieved = True
                    else:
                        print("  'rec_scores' NOT RETRIEVED (returned None).")


                    if data_retrieved and rec_texts: # Check if we have texts to print
                        print(f"\n--- OCR Results from '{real_image_path}' (accessed via .get()) ---")
                        for i in range(len(rec_texts)):
                            text = rec_texts[i]
                            score = rec_scores[i] if rec_scores is not None and i < len(rec_scores) else -1.0
                            box_poly = dt_polys[i] if dt_polys is not None and i < len(dt_polys) else "N/A"

                            print(f"  Text: '{text}', Score: {score:.4f}, Box: {box_poly}")
                    elif data_retrieved and not rec_texts:
                         print(f"Data fields were accessible, but no text was recognized in '{real_image_path}'.")
                    else:
                        print("\nFailed to retrieve data using .get(). Printing full object via .print() for final review:")
                        # The ocr_result_data object itself has the .print() method as per docs
                        ocr_result_data.print()
                else:
                    print(f"Real Image ({real_image_path}) - OCR predict() returned an empty list, or its first element is None.")
            except Exception as e:
                print(f"Error during OCR on real image: {e}")
                traceback.print_exc()
        else:
            print(f"Error: Could not read the real image: {real_image_path}")
    else:
        print(f"Error: Real image file not found: {real_image_path}. Please create this file.")
else:
    print("\n--- PaddleOCR engine could not be initialized. OCR test skipped. ---")