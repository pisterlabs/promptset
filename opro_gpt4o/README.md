## Prompt OPRO Usage Guide

- Set CWD in `main.py` to the taget directory to store results. It is set to "results/" for now.
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
- Once done, all results will be stored in the target directory (value of CWD in `main.py`). Navigate to the target directory and run `gen_report.py` to generate a html report. Open html report in browser to view results.

Note: Rate limits can be a problem when working with gpt4o if the account usage tier is not 2 or higher.


### Results
Our Results for OPRO with gpt4o are stored in the `results/` directory. `opro_stats.html` contains the results of the optimization process with graphical representations. `analysis.ipynb` shows a summary of the results.


### Alternative Usage
If you're just interested in a quick way to test OPRO for your own prompts, you can use the OPRO class in the `OPRO.py` file.

Here's an example of how you can use it:
```python
from OPRO import OPRO
import asyncio

if __name__ == "__main__":
    opro = OPRO("your_openai_api_key_here")  # recommended account tier is 2 or higher
    my_prompt = "Please provide a detailed character description for the following character type:\n{char_type}\n\nFeel free to include their personality, appearance, background, or any other relevant details."
    print(asyncio.run(opro.optimize(prompt=my_prompt, category="QA_refinement", id=0)))
```