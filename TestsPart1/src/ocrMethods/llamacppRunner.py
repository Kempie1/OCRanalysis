import requests
import base64
import os
import io
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import time
from openai import OpenAI

load_dotenv()
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
SERVER_URL = os.getenv("LLM_HOST")

client = OpenAI(
    base_url=SERVER_URL+"/v1", 
    api_key="your-api-key" 
)

def llamacppRunnerQwen(image: Image.Image):
    image = image.convert("RGB")
    buffered = io.BytesIO()
    
    start = time.time()
    image = resize_image(image, max_size=600)
    resize_time = time.time() - start

    image.save(buffered, format="JPEG") 
    img_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    start = time.time()
    response = client.chat.completions.create(
    model="anytext",
    messages=[
        {
            "role": "system", 
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                }
            ]
        }
    ]
    )
    elapsed = time.time()-start
    return response.choices[0].message.content, elapsed+resize_time


def resize_image(image:Image.Image, max_size=1024):
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size)
    return image

def llamacppRunnerQwen_with_retry(image: Image.Image, max_retries=20):
    for attempt in range(max_retries):
        try:
            return llamacppRunnerQwen(image)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise 