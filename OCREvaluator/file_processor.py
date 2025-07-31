"""
File processing module for handling OCR and ground truth file pairs.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import re


class FileProcessor:
    """Handles finding and pairing OCR files with their ground truth counterparts."""
    
    def __init__(self, ocr_dir: Path, ground_truth_dir: Path):
        self.ocr_dir = Path(ocr_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.logger = logging.getLogger(__name__)
    
    def _get_base_name(self, file_path: Path) -> str:
        """
        Extract base name for matching files.
        
        This method can be customized based on your file naming convention.
        Examples:
        - "document_001_ocr.txt" -> "document_001"
        - "page_001.txt" -> "page_001"
        """
        name = file_path.stem
        
        # Remove common OCR suffixes
        suffixes_to_remove = ['_ocr', '_OCR', '_extracted', '_text']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        return name
    
    def read_file_content(self, file_path: Path) -> str:
        """
        Read and return the content of a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                return ""
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def get_file_info(self, file_path: Path) -> dict:
        """
        Get metadata information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            stat = file_path.stat()
            content = self.read_file_content(file_path)
            
            return {
                'name': file_path.name,
                'size_bytes': stat.st_size,
                'char_count': len(content),
                'line_count': len(content.splitlines()),
                'word_count': len(content.split()) if content else 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get info for {file_path}: {e}")
            return {'name': file_path.name, 'error': str(e)}
    
    def validate_file_pair(self, ocr_file: Path, gt_file: Path) -> bool:
        """
        Validate that a file pair is suitable for processing.
        
        Args:
            ocr_file: Path to OCR file
            gt_file: Path to ground truth file
            
        Returns:
            True if pair is valid, False otherwise
        """
        try:
            # Check that both files exist and are readable
            ocr_content = self.read_file_content(ocr_file)
            gt_content = self.read_file_content(gt_file)
            
            if not ocr_content.strip():
                self.logger.warning(f"OCR file is empty: {ocr_file.name}")
                return False
            
            if not gt_content.strip():
                self.logger.warning(f"Ground truth file is empty: {gt_file.name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating file pair {ocr_file.name}, {gt_file.name}: {e}")
            return False
