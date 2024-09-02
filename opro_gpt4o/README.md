## Script Guide

- Set CWD in `main.py` to the taget directory to store results. Set to "results/" for now.
- Set openai api key in a .env file with the name `OPENAI_API_KEY`.
- In `main.py`, uncomment any of the following to execute them.
    - The first argument is the list of prompts to be optimized. Feel free to change these to any list of prompts you want to optimize.
```
# Optimizing Prompts
# await opro(PROMPT_LIST_QA, "QA_refinement")
# await opro(PROMPT_LIST_TRANS, "translation")
# await opro(PROMPT_LIST_ERR, "error_correction")
# await opro(PROMPT_LIST_SUMM, "summarization")
```
- Once done, all results will be stored in the target directory (set to 'results/' for now). Navigate to the target directory and run `gen_report.py` to generate a html report. Open html report in browser to view results.
