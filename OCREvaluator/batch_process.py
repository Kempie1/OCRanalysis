#!/usr/bin/env python3
"""
Batch processing utility for OCR grading with common presets.
"""

import argparse
import sys
from pathlib import Path
import subprocess


def run_grading(ocr_dir, ground_truth_dir, preset="fast", custom_args=None):
    """Run the OCR grading with specified preset."""
    
    presets = {
        "fast": {
            "llm": "openai",
            "model": "gpt-4o-mini",
            "batch_size": 10,
            "description": "Fast and cost-effective using GPT-4o-mini"
        },
        "accurate": {
            "llm": "openai", 
            "model": "gpt-4o",
            "batch_size": 5,
            "description": "Most accurate using GPT-4o"
        },
        "claude": {
            "llm": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "batch_size": 5,
            "description": "Using Claude Sonnet for analysis"
        },
        "claude-fast": {
            "llm": "anthropic",
            "model": "claude-3-haiku-20240307",
            "batch_size": 10,
            "description": "Fast Claude Haiku processing"
        }
    }
    
    if preset not in presets:
        print(f"❌ Unknown preset: {preset}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return False
    
    config = presets[preset]
    print(f"🚀 Running with preset '{preset}': {config['description']}")
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--ocr-dir", str(ocr_dir),
        "--ground-truth-dir", str(ground_truth_dir),
        "--llm", config["llm"],
        "--model", config["model"],
        "--batch-size", str(config["batch_size"])
    ]
    
    # Add custom arguments
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Grading completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Grading failed with error code {e.returncode}")
        return False


def estimate_cost(num_files, preset="fast"):
    """Estimate approximate cost for processing."""
    
    cost_per_1000 = {
        "fast": 3.0,  # GPT-4o-mini
        "accurate": 20.0,  # GPT-4o
        "claude": 12.0,  # Claude Sonnet
        "claude-fast": 2.0  # Claude Haiku
    }
    
    cost = (num_files / 1000) * cost_per_1000.get(preset, 10.0)
    print(f"💰 Estimated cost for {num_files} files with '{preset}' preset: ${cost:.2f}")
    return cost


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch OCR grading with presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  fast        - GPT-4o-mini, batch_size=10 (cost-effective)
  accurate    - GPT-4o, batch_size=5 (most accurate)  
  claude      - Claude Sonnet, batch_size=5 (alternative LLM)
  claude-fast - Claude Haiku, batch_size=10 (fast Claude)

Examples:
  python batch_process.py /path/to/ocr /path/to/ground_truth
  python batch_process.py /path/to/ocr /path/to/ground_truth --preset accurate
  python batch_process.py /path/to/ocr /path/to/ground_truth --estimate-only
        """
    )
    
    parser.add_argument("ocr_dir", help="Directory containing OCR files")
    parser.add_argument("ground_truth_dir", help="Directory containing ground truth files") 
    parser.add_argument("--preset", choices=["fast", "accurate", "claude", "claude-fast"], 
                       default="fast", help="Processing preset (default: fast)")
    parser.add_argument("--estimate-only", action="store_true", 
                       help="Only estimate cost, don't run processing")
    parser.add_argument("--resume", action="store_true", help="Resume previous run")
    parser.add_argument("--output-dir", help="Custom output directory")
    
    args = parser.parse_args()
    
    # Validate directories
    ocr_dir = Path(args.ocr_dir)
    gt_dir = Path(args.ground_truth_dir)
    
    if not ocr_dir.exists():
        print(f"❌ OCR directory not found: {ocr_dir}")
        return 1
    
    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        return 1
    
    # Count files
    ocr_files = list(ocr_dir.glob("*.txt"))
    gt_files = list(gt_dir.glob("*.txt"))
    
    print(f"📁 Found {len(ocr_files)} OCR files and {len(gt_files)} ground truth files")
    
    if not ocr_files:
        print("❌ No OCR files found")
        return 1
    
    if not gt_files:
        print("❌ No ground truth files found")  
        return 1
    
    # Estimate cost
    estimate_cost(len(ocr_files), args.preset)
    
    if args.estimate_only:
        return 0
    
    # Confirm processing
    response = input(f"\n❓ Proceed with processing {len(ocr_files)} files using '{args.preset}' preset? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Processing cancelled")
        return 0
    
    # Build custom arguments
    custom_args = []
    if args.resume:
        custom_args.append("--resume")
    if args.output_dir:
        custom_args.extend(["--output-dir", args.output_dir])
    
    # Run processing
    success = run_grading(ocr_dir, gt_dir, args.preset, custom_args)
    
    if success:
        print("\n🎉 Processing completed!")
        output_dir = Path(args.output_dir) if args.output_dir else Path("./results")
        print(f"📊 Results available in: {output_dir}")
        print(f"📈 View summary: {output_dir}/summary_report.json")
        print(f"🌐 View report: {output_dir}/detailed_report.html")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
