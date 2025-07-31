#!/usr/bin/env python3
"""
Quick setup script for the OCR Grading System
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up OCR Grading System...\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "openai>=1.0.0",
        "anthropic>=0.18.0", 
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("\n🔧 Creating .env file...")
        try:
            with open(".env.example", "r") as example:
                content = example.read()
            with open(".env", "w") as env:
                env.write(content)
            print("✅ Created .env file - please add your API keys!")
        except Exception as e:
            print(f"⚠️  Warning: Could not create .env file: {e}")
    
    # Create example directories
    print("\n📁 Creating example directories...")
    example_dirs = ["./example_ocr", "./example_ground_truth", "./results"]
    for dir_path in example_dirs:
        Path(dir_path).mkdir(exist_ok=True)
    print("✅ Example directories created")
    
    # Create test files
    print("\n📝 Creating test files...")
    test_ocr = Path("./example_ocr/test_001.txt")
    test_gt = Path("./example_ground_truth/test_001.txt")
    
    if not test_ocr.exists():
        test_ocr.write_text("This is a sample OCR output with some erors and mispellings.")
    
    if not test_gt.exists():
        test_gt.write_text("This is a sample OCR output with some errors and misspellings.")
    
    print("✅ Test files created")
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Test with: python main.py --ocr-dir ./example_ocr --ground-truth-dir ./example_ground_truth")
    print("3. Check the results in ./results/")
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()
