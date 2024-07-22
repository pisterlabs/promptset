from llm_async import run_llm_coroutine
import asyncio
import re

INTERPOLATE_VAR = "TEXT"

async def create_scoring_prompt(prompt, sample_data):
    """
    Given a prompt and sample data, generates a scoring prompt for the prompt.
    """    
    prompt_template = f"""Write a scoring prompt for an example input output pair on a prompt to a language model. 
Use the variable name output for the output of the prompt. The scoring prompt must contain the output variable and text variable.
Your answer should be inside <BEGIN_CRITERIA> and <END_CRITERIA> constructs.

## Example:
<BEGIN_PROMPT> 'what is a fruit of color: {{TEXT}}. Return the name of the fruit and nothing else:' <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {{"text": "yellow", "output": "banana"}} <END_EXAMPLE_INPUT>
<BEGIN_CRITERIA> Is a ${{output}} this color: ${{text}}? Answer yes or no only. <END_CRITERIA>

## Query:
<BEGIN_PROMPT> {prompt} <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {sample_data} <END_EXAMPLE_INPUT>"""

    res = await run_llm_coroutine([prompt_template], model="llama3-70b")
    res = res[0]
    
    # Extract the scoring prompt
    for i in range(3):
        try:
            match = re.search(r'<BEGIN_CRITERIA>(.*?)<END_CRITERIA>', res, re.DOTALL)
            assert match is not None, "No match found for <BEGIN_CRITERIA> and <END_CRITERIA> tags"
            extracted_text = match.group(1).strip()
            extracted_text.format(output="PLACEHOLDER", text="PLACEHOLDER")
            break
        except KeyError as e:
            print(f"KeyError: {e}. Extracted text does not have correct keywords")
        except AssertionError as e:
            print(f"Generating new scoring prompt. Attempt {i+1} failed.")
        
    return extracted_text

def check_and_reformat(prompt):
    """
    Checks if prompt is valid. If prompt is valid, returns a slightly modified prompt that can be evaluated and optimized.
    """
    pattern1 = r"{[^}]*}"
    pattern2 = r"PLACEHOLDER"
    matches1 = re.findall(pattern1, prompt)
    matches2 = re.findall(pattern2, prompt.upper())
    if not (len(matches1) == 1 or len(matches2) == 1):
        print(prompt)
    
    assert (
        len(matches1) == 1 or len(matches2) == 1
    ), "Invalid prompt format. Prompt must contain some str/var to be interpolated."

    # Reformat the prompt
    if len(matches1) == 1:
        return prompt.replace(matches1[0], INTERPOLATE_VAR)
    else:
        return prompt.replace(matches2[0], INTERPOLATE_VAR)

if __name__ == "__main__":
    sample_data = {"text": "print('hello world')", "output": "it prints 'hello world' to the screen"}
    prompt = "what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:"
    sample_data = {"text": "white and red", "output": "Japan"}
    prompt = check_and_reformat(prompt)
    print(prompt)
    
    res = asyncio.run(create_scoring_prompt(prompt, sample_data))
    print(res)