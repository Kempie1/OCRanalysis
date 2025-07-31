from datasets import load_from_disk
from PIL import Image
import ocrMethods.paddleocrRunner as paddleocrRunner
import ocrMethods.tesseractRunner as tesseractRunner
import ocrMethods.llmRunner as llmRunner
import ocrMethods.llamacppRunner as qwenRunner
import os
import time

OUTPUT_DIR = "output"

TEST_RUNNERS = {
    "paddleocr": paddleocrRunner.paddleOCRRunner,
    "tesseract": tesseractRunner.tesseractRunner,
    "llamacpp": qwenRunner.llamacppRunnerQwen_with_retry,
    "gemma3": llmRunner.gemmaRunner,
    "qwen": llmRunner.qwenRunner

}

def perform_experiment():
    # Dataset includes image, id, metadata, true_markdown_output, json_schema, true_json_output
    # Image is a PIL Image object
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("CodeArte/ocr-benchmark")
    run_dir = os.path.join(OUTPUT_DIR, str(int(time.time())))
    os.makedirs(run_dir, exist_ok=True)
    for test_name, test_runner in TEST_RUNNERS.items():  
        print(f"Running {test_name} OCR...")
        test_output_dir = os.path.join(run_dir, test_name)
        os.makedirs(test_output_dir, exist_ok=True)
        overall_elapsed = 0
        
        for index, dataset_item in enumerate(dataset["test"]):
            if index <365:
                continue
            print("current Index:", index)
            if isinstance(dataset_item["image"], Image.Image):
                image = dataset_item["image"]
            else: 
                print(f"Item {index} is not an image, skipping...")
                continue
            # Run the OCR method
            run_result, elapsed = test_runner(image)
            # Save the result to a file
            with open(os.path.join(test_output_dir, str(index)+test_name+".txt"), "w") as f:
                f.write(str(run_result).strip())
            
            with open(os.path.join(test_output_dir, f"{test_name}_time.txt"), "a") as f:
                f.write(f"Index {index} took {elapsed:.4f} seconds \n")
            overall_elapsed += elapsed
        num_tests = len(dataset["test"])
        with open(os.path.join(test_output_dir, f"{test_name}_time.txt"), "a") as f:
            f.write(f"{overall_elapsed:.4f} seconds, with average of  {overall_elapsed/num_tests:.4f} \n")
    return True
def main():
    perform_experiment()


if __name__ == "__main__":
    main()
