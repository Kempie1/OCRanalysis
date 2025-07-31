from paddleocr import PaddleOCR
import numpy as np
from typing import Dict
import cv2
import time


ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        lang="en",
    )
def paddleOCRRunner(image):
    ndarray_image = np.array(image)
    if len(ndarray_image.shape) == 2:
        # Grayscale image, convert to BGR
        ndarray_image = cv2.cvtColor(ndarray_image, cv2.COLOR_GRAY2BGR)
    elif ndarray_image.shape[-1] == 4:
        # Convert RGBA to BGR
        ndarray_image = cv2.cvtColor(ndarray_image, cv2.COLOR_RGBA2BGR)
    start = time.time()
    result = ocr.predict(input=ndarray_image)
    elapsed = time.time() - start
    ocr_results = []
    for i, res in enumerate(result):
        result_json = res.json

        simple_text = extract_text_from_json(result_json, False)
        ocr_results.append(simple_text)

    # Combine all extracted texts into a single string
    print(ocr_results)
    paddleOcr_text_result = [res["extracted_text"] for res in ocr_results][0]
    return paddleOcr_text_result, elapsed


def extract_text_from_json(result_json: Dict, include_bbox: bool = False):
    """
    Code adapted from: sparrow
    Source: https://github.com/katanaml/sparrow/
    License: GPL-3.0
    Accessed: July 27, 2025
    """
    ocr_data = result_json.get('res', {})

    rec_texts = ocr_data.get('rec_texts', [])
    rec_scores = ocr_data.get('rec_scores', [])
    rec_boxes = ocr_data.get('rec_boxes', []) if include_bbox else []

    clean_texts = []
    text_regions = []

    for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
        if text and text.strip() and score > 0.3: 
            clean_text = text.strip()
            clean_texts.append(clean_text)

            if include_bbox and i < len(rec_boxes):
                box = rec_boxes[i]
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    text_regions.append({
                        "text": clean_text,
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        },
                        "confidence": round(float(score), 3)
                    })

    simple_output = {
        "extracted_text": " ".join(clean_texts),
        "text_count": len(clean_texts),
        "avg_confidence": round(sum(rec_scores) / len(rec_scores), 2) if rec_scores else 0
    }

    if include_bbox:
        simple_output["text_regions"] = text_regions

    return simple_output
