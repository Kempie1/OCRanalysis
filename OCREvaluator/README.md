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
