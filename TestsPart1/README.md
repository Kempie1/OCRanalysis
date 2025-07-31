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