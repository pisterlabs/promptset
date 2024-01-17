# How to parse repo data into prompt dataset

1. `python -m gen_prompts.find_prompts --run_id {x}` reads from the data/scraping-2 directory and calls `parsers.py` to find prompt patterns
2. `python -m gen_prompts.reader --run_id {x}`
3. `python -m gen_prompts.separate --run_id {x}`
4. `python -m gen_prompts.read_strings --run_id {x}`
