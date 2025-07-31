from dotenv import load_dotenv
import os
import io
from ollama import Client
import sys
from PIL import Image, ImageEnhance
import base64
import cv2
import numpy as np
import time

load_dotenv()
MODEL12= "google_gemma-3-12b-it-qat-q4_0-gguf_gemma-3-12b-it-q4_0.gguf"
MODELQWEN= "qwen2.5vl:7b-q8_0"

SYSTEM_PROMPT = """You are a highly accurate text extraction model. 
                 Your task is to extract all text from the provided document image.
                 Preserve the original order, spelling, punctuation, and formatting. 
                 Do not omit or add anything. 
                 Do not guess or infer missing information.
                 Pay attention to all words, letters and numbers in the image.
                 Do not add or change any words.
                 Don't say anything yourself, just return the text.
                 Read the image the best you can, and return the text as accurately as possible.
                 
                 """

USER_PROMPT = "Extract all text from the image."
def gemmaRunner( image ):
    # Convert Image to base64
    image = image.convert("RGB")
    image = resize_image(image, max_size=1024)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") 
    img_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    # Use Gemma 3 to get answers
    ollama_host = os.getenv("LLM_HOST")
    client = Client(host=ollama_host)
    start = time.time()
    response = client.chat(
        model=MODEL12,
        messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT
            },
                {"role": "user", 
                "content": USER_PROMPT,
                "images": [encoded_image],
                },
        ],
    )
    elapsed = time.time()-start
    return response.message.content, elapsed


def qwenRunner( image ):
    # Convert Image to base64
    image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") 
    img_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    # Use Gemma 3 to get answers
    ollama_host = os.getenv("LLM_HOST")
    client = Client(host=ollama_host)
    start = time.time()
    image = resize_image(image, max_size=1024)
    response = client.chat(
        model=MODELQWEN,
        messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT
            },
                {"role": "user", 
                "content": USER_PROMPT,
                "images": [encoded_image],
                },
        ],
    )
    elapsed = time.time()-start

    return response.message.content, elapsed


def resize_image(image, max_size=1024):
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image