# OCR analysis tool
These tools were used to perform analysis of OCR engines, specifically:

PaddleOCR3
Tesseract
EasyOCR
Gemma3:12b
Qwen2.5v:7b
## How to use
### TestsPart1
This part contains tests for:
PaddleOCR3
Tesseract
Gemma3:12b
Qwen2.5v:7b

For the Gemma3:12b, Qwen2.5v:7b you will need a way to host the model. For my testing I used llama.cpp, with the commands that are given in ```TestPart1/llamacpp_start_command.txt``` file.

Enter the folder, in the pyproject.toml set the torch version to the one you need, 

Make sure you have uv for python installed
then run
```uv sync```
and ```uv run src/main.py```

You might want to comment out, improts and TEST_RUNNERS dictionary entries for the tests you are not running as they will eat up memory.

### TestsPart2
This part contains tests for:
EasyOCR
Enter the folder, in the pyproject.toml set the torch version to the one you need, 

Make sure you have uv for python installed
then run
```uv sync```
and ```uv run src/main.py```

### OCREvaluator
This is the evaluator utility.

Make sure you have uv for python installed
then run
```uv sync```
and ```uv run src/main.py```

Put the result files from the Tests run, into a folder inside input folder such as: ```input/name```, 
then in main.py edit line 40 to correspond to your folder name, and line 41 to the suffix your test result files use, for example file named 0llamacpp.txt inside qwen folder, would have those lines be:

```
TEST_SOURCE= "qwen"
FILE_SUFFIX= "llamacpp"
```


## Citations
Dataset used: https://huggingface.co/datasets/getomni-ai/ocr-benchmark

TestsPart1/src/ocrMethods/paddleocrRunner.py  lines 37-84:

    Code adapted from: sparrow
    Source: https://github.com/katanaml/sparrow/
    License: GPL-3.0
    Accessed: July 27, 2025