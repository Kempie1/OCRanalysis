from datasets import load_from_disk
from PIL import Image
import os
import time
import easyocr
import numpy as np

OUTPUT_DIR = "output"

reader = easyocr.Reader(['en'])
def easyOCRrunner(image):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    start= time.time()
    result = reader.readtext(image_array, detail = 0)
    elapsed = time.time() - start
    result = " ".join(result)
    return result, elapsed

def perform_experiment():
    # Dataset includes image, id, metadata, true_markdown_output, json_schema, true_json_output
    # Image is a PIL Image object
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("CodeArte/ocr-benchmark")
    run_dir = os.path.join(OUTPUT_DIR, str(int(time.time())))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Running easyOCR...")
    test_output_dir = os.path.join(run_dir, "test_name")
    os.makedirs(test_output_dir, exist_ok=True)
    overall_elapsed = 0
    
    for index, dataset_item in enumerate(dataset["test"]):
        if index < 636:  
            continue
        print("current Index:", index)
        if isinstance(dataset_item["image"], Image.Image):
            image = dataset_item["image"]
        else: 
            print(f"Item {index} is not an image, skipping...")
            continue
        # Run the OCR method
        run_result, elapsed = easyOCRrunner(image)
        # Save the result to a file
        with open(os.path.join(test_output_dir, str(index)+"easyOCR"+".txt"), "w") as f:
            f.write(str(run_result).strip())
        
        with open(os.path.join(test_output_dir, f"{"easyOCR"}_time.txt"), "a") as f:
            f.write(f"Index {index} took {elapsed:.4f} seconds \n")
        overall_elapsed += elapsed
    num_tests = len(dataset["test"])
    with open(os.path.join(test_output_dir, f"{"easyOCR"}_time.txt"), "a") as f:
        f.write(f"{overall_elapsed:.4f} seconds, with average of  {overall_elapsed/num_tests:.4f} \n")
    return True
def main():
    perform_experiment()


if __name__ == "__main__":
    main()
