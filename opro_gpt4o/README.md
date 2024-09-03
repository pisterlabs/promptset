## Usage Guide

- Set CWD in `main.py` to the target directory to store results. It is set to "results/" for now.
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
- Once done, all results will be stored in the target directory (value of CWD in `main.py`). Navigate to the target directory and run `gen_report.py` to generate a report in HTML. Open the report in the browser to view the results.

Notes: 
- Rate limits can be a problem when working with gpt4o if the account usage tier is not 2 or higher.
- To facilitate the experiment process, a sequential approach was employed, where OPro was run for one category at a time. Each category was assigned a unique target directory to ensure organized and distinct output for easier analysis and comparison.
