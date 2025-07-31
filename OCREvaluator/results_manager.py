"""
Results management module for storing and reporting OCR grading results.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv


class ResultsManager:
    """Manages storage and reporting of OCR grading results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.results_file = self.output_dir / "grading_results.json"
        self.csv_file = self.output_dir / "grading_results.csv"
        self.summary_file = self.output_dir / "summary_report.json"
        self.detailed_report = self.output_dir / "detailed_report.html"
        self.progress_file = self.output_dir / "progress.json"
    
    def save_result(self, result: Dict[str, Any]) -> None:
        """
        Save a batch of results incrementally.
        
        Args:
            batch_results: List of grading result dictionaries
        """
        try:
            # Load existing results
            existing_results = []
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    existing_results = json.load(f)
            
            # Append new results
            existing_results.append(result)
            
            # Save updated results
            with open(self.results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            # Also save as CSV for easy analysis
            self._save_to_csv(existing_results)
            
            # Update progress
            self._update_progress(len(existing_results))
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {e}")
            raise
    
    def _save_to_csv(self, results: List[Dict[str, Any]]) -> None:
        """Save results to CSV format for easy analysis."""
        try:
            # Flatten the results for CSV
            flattened_results = []
            
            for result in results:
                if 'refusal' in result:
                    # Handle error cases
                    flattened_results.append({
                        'index': result.get('index'),
                        'overall_score': None,
                        'character_accuracy': None,
                        'word_accuracy': None,
                        'confidence_level': None,
                        'error': result.get('refusal', ''),
                    })
                else:
                    # Handle successful grading
                    flattened_results.append({
                        'index': result.get('index'),
                        'overall_score': result.get('overall_score'),
                        'character_accuracy': result.get('character_accuracy'),
                        'word_accuracy': result.get('word_accuracy'),
                        'confidence_level': result.get('confidence_level'),
                        'error': '',
                    })
            
            # Write to CSV
            if flattened_results:
                df = pd.DataFrame(flattened_results)
                df.to_csv(self.csv_file, index=False)
                
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
    
    def _update_progress(self, total_processed: int) -> None:
        """Update progress tracking."""
        progress = {
            'total_processed': total_processed,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def get_processed_files(self) -> set:
        """Get set of already processed filenames for resume functionality."""
        try:
            if not self.results_file.exists():
                return set()
            
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            
            processed = set()
            for result in results:
                if 'filename' in result:
                    # Remove .txt extension for matching
                    filename = result['filename']
                    if filename.endswith('.txt'):
                        filename = filename[:-4]
                    processed.add(filename)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error getting processed files: {e}")
            return set()
    
    def generate_final_report(self, all_results: List[Dict[str, Any]], failed_files: List[str]) -> None:
        """
        Generate comprehensive final report.
        
        Args:
            all_results: All grading results
            failed_files: List of files that failed processing
        """
        try:
            # Calculate summary statistics
            summary = self._calculate_summary_stats(all_results, failed_files)
            
            # Save summary report
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate HTML report
            self._generate_html_report(summary, all_results)
            
            self.logger.info(f"Final report generated: {self.summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            raise
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]], failed_files: List[str]) -> Dict:
        """Calculate summary statistics from results."""
        successful_results = [r for r in results if 'error' not in r or not r['error']]
        
        if not successful_results:
            return {
                'total_files': len(results) + len(failed_files),
                'successful': 0,
                'failed': len(results) + len(failed_files),
                'error': 'No successful results to analyze'
            }
        
        # Extract scores
        overall_scores = [r.get('overall_score', 0) for r in successful_results if r.get('overall_score') is not None]
        char_accuracies = [r.get('character_accuracy', 0) for r in successful_results if r.get('character_accuracy') is not None]
        word_accuracies = [r.get('word_accuracy', 0) for r in successful_results if r.get('word_accuracy') is not None]
        
        # Calculate statistics
        summary = {
            'total_files': len(results) + len(failed_files),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results) + len(failed_files),
            'processing_date': datetime.now().isoformat(),
            
            'overall_score_stats': {
                'mean': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                'min': min(overall_scores) if overall_scores else 0,
                'max': max(overall_scores) if overall_scores else 0,
                'count': len(overall_scores)
            },
            
            'character_accuracy_stats': {
                'mean': sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0,
                'min': min(char_accuracies) if char_accuracies else 0,
                'max': max(char_accuracies) if char_accuracies else 0,
                'count': len(char_accuracies)
            },
            
            'word_accuracy_stats': {
                'mean': sum(word_accuracies) / len(word_accuracies) if word_accuracies else 0,
                'min': min(word_accuracies) if word_accuracies else 0,
                'max': max(word_accuracies) if word_accuracies else 0,
                'count': len(word_accuracies)
            }
        }
        
        # Score distribution
        score_ranges = {'0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0}
        for score in overall_scores:
            if score <= 20:
                score_ranges['0-20'] += 1
            elif score <= 40:
                score_ranges['21-40'] += 1
            elif score <= 60:
                score_ranges['41-60'] += 1
            elif score <= 80:
                score_ranges['61-80'] += 1
            else:
                score_ranges['81-100'] += 1
        
        summary['score_distribution'] = score_ranges
        
        return summary
    
    def _generate_html_report(self, summary: Dict, results: List[Dict[str, Any]]) -> None:
        """Generate an HTML report for easy viewing."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR Grading Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ text-align: center; padding: 15px; background-color: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .distribution {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .error {{ color: red; }}
                .good {{ color: green; }}
                .average {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>OCR Accuracy Grading Report</h1>
                <p>Generated on: {summary.get('processing_date', 'Unknown')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>{summary.get('total_files', 0)}</h3>
                        <p>Total Files</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary.get('successful', 0)}</h3>
                        <p>Successfully Processed</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary.get('failed', 0)}</h3>
                        <p>Failed</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary.get('overall_score_stats', {}).get('mean', 0):.1f}</h3>
                        <p>Average Overall Score</p>
                    </div>
                </div>
            </div>
            
            <div class="distribution">
                <h2>Score Distribution</h2>
                <table>
                    <tr><th>Score Range</th><th>Count</th><th>Percentage</th></tr>
        """
        
        # Add score distribution
        total_successful = summary.get('successful', 1)
        for range_name, count in summary.get('score_distribution', {}).items():
            percentage = (count / total_successful) * 100 if total_successful > 0 else 0
            html_content += f"<tr><td>{range_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(self.detailed_report, 'w') as f:
            f.write(html_content)
    
    def export_for_analysis(self, format: str = "csv") -> Path:
        """
        Export results in specified format for further analysis.
        
        Args:
            format: Export format ("csv", "json", "excel")
            
        Returns:
            Path to exported file
        """
        if format == "csv":
            return self.csv_file
        elif format == "json":
            return self.results_file
        elif format == "excel" and pd is not None:
            excel_file = self.output_dir / "grading_results.xlsx"
            df = pd.read_csv(self.csv_file)
            df.to_excel(excel_file, index=False)
            return excel_file
        else:
            raise ValueError(f"Unsupported export format: {format}")
