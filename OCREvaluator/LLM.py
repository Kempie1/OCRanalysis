import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ResponseModel(BaseModel):
    overall_score: float
    character_accuracy: float
    word_accuracy: float
    confidence_level: float

def judgeLLm(pair):
    """
    pairs: List of tuples (ocr_text, ground_truth)
    Returns: List of responses
    """
    ocr_text, ground_truth, index = pair
    if not ocr_text or not ground_truth:
        return {
            "overall_score": 0.0,
            "character_accuracy": 0.0,
            "word_accuracy": 0.0,
            "confidence_level": 100,
            "index": index
        }
    prompt = f"""You are an expert evaluator assessing the accuracy of OCR (Optical Character Recognition) output against ground truth text.

    Your task is to compare OCR-extracted text with the ground truth and provide a comprehensive accuracy assessment.
    Ground truth text is in markdown format, and OCR text is in plain text.
    Please ignore any markdown formatting in the ground truth text.

    **OCR OUTPUT:**
    {ocr_text}

    **GROUND TRUTH:**
    {ground_truth}

    Please analyze the OCR accuracy and provide your assessment in the following JSON format:

    {{
        "overall_score": <number between 0-100>,
        "character_accuracy": <percentage of correctly recognized characters>,
        "word_accuracy": <percentage of correctly recognized words>,
        "confidence_level": <your confidence in this assessment, 0-100>,
    }}

    Focus on:
    1. Character-level accuracy
    2. Word-level accuracy  
    3. Preservation of formatting and structure
    4. Common OCR error patterns
    5. Overall readability and usability

    Be precise and objective in your assessment. Consider the practical impact of errors on text usability.
    """
    response = client.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18", 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=ResponseModel,
        temperature=0
    )
    result =json.loads((response.choices[0].message.content))
    result["index"] = index
    return result
