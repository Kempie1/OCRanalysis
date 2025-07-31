import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from file_processor import FileProcessor
from results_manager import ResultsManager
from datasets import load_from_disk
import LLM as LLM

OCR_DIR = Path("./input")
RESULTS_DIR = Path("./results")

def validate_directories(ocr_dir: Path, ground_truth_dir: Path) -> None:
    """Validate that input directories exist and contain files."""
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")
    
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")
    
    ocr_files = list(ocr_dir.glob("*.txt"))
    gt_files = list(ground_truth_dir.glob("*.txt"))
    
    if not ocr_files:
        raise ValueError(f"No TXT files found in OCR directory: {ocr_dir}")
    
    if not gt_files:
        raise ValueError(f"No TXT files found in ground truth directory: {ground_truth_dir}")
    
    logging.info(f"Found {len(ocr_files)} OCR files and {len(gt_files)} ground truth files")


TEST_SOURCE= "qwen"
FILE_SUFFIX= "llamacpp"
def main():
    """Main function to orchestrate the OCR grading process."""
    # Load environment variables
    load_dotenv()
    
    # Override config with command line arguments
    ocr_dir = OCR_DIR
    output_dir = RESULTS_DIR
    
    dataset = load_dataset("CodeArte/ocr-benchmark")


    try:
        tupple_list= []
        for index, dataset_item in enumerate(dataset["test"]):
            ground_truth = dataset_item["true_markdown_output"]
            # Get OCR text from the folder
            with open(ocr_dir / TEST_SOURCE / f"{str(index)+FILE_SUFFIX}.txt", "r") as f:
                ocr_text = f.read()
            tupple= (ocr_text, ground_truth, index)
            tupple_list.append(tupple)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        results_manager = ResultsManager(output_dir)
        
        # Get file pairs to process
        total_files = len(tupple_list)
        
        logging.info(f"Processing {total_files} file pairs")
        
        
        # Process files in batches
        all_results = []
        failed_files = []
        
        with tqdm(total_files, desc="Grading OCR files") as pbar:
            for i in range(0, len(tupple_list), 1):
                try:
                    result = LLM.judgeLLm(tupple_list[i])
                    all_results.append(result)
                    
                    # Save results incrementally
                    results_manager.save_result(result)
                    
                except Exception as e:
                    logging.error(f"Error processing batch {i//10 + 1}: {e}")
                    failed_files.extend([fp[0].name for fp in tupple_list[i]])
                
                pbar.update(1)
        
        # Generate final report
        logging.info("Generating final report...")
        results_manager.generate_final_report(all_results, failed_files)
        
        logging.info(f"Processing complete! Results saved to {output_dir}")
        logging.info(f"Successfully processed: {len(all_results)} files")
        logging.info(f"Failed: {len(failed_files)} files")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
