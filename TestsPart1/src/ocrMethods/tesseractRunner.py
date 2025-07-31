import pytesseract
from PIL import Image
import time

def tesseractRunner(image):
    # convert image to PIL Image if not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    start = time.time()
    try:
        orientation = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        if orientation['rotate'] != 0:
            image = image.rotate(orientation['rotate'], expand=True)
        # Perform OCR
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractError as e:
        print("Tesseract failed:", e)
        text = ""
    elapsed = time.time() - start
    return text, elapsed
