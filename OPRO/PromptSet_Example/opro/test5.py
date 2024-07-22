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

    res = await run_llm_coroutine([prompt_template], model="llama3-70b", temperature=1.0)
    res = res[0]
    
    # Extract the scoring prompt
    for i in range(3):
        try:
            # Extract Criteria
            match = re.search(r'<BEGIN_CRITERIA>(.*?)<END_CRITERIA>', res, re.DOTALL)
            assert match is not None, "No match found for <BEGIN_CRITERIA> and <END_CRITERIA> tags"
            extracted_text = match.group(1).strip()
            
            # Check if extracted text has the correct keywords
            matches = re.findall(r"{[^}]*}", extracted_text)
            assert matches is not None, "No matches found for variables in prompt"
            matches = list(map(lambda x: x.lower(), matches))
            print(matches)
            assert "{text}" in matches and "{output}" in matches, "Prompt does not contain the correct keywords"
            return extracted_text
        except AssertionError as e:
            print(e, f"Prompt: {extracted_text}")
            print(f"Generating new scoring prompt. Attempt {i+1} failed.")
        
    return None


def check_and_reformat(prompt):
    """
    Checks if prompt is valid. If prompt is valid, returns a slightly modified prompt that can be evaluated and optimized.
    """
    pattern1 = r"{[^}]*}"
    pattern2 = "PLACEHOLDER"
    matches1 = re.findall(pattern1, prompt)
    condition1 = len(matches1) == 1 
    condition2 = prompt.count(pattern2) == 1
    
    if not condition1 and not condition2:
        print(prompt)
    
    # Reformat the prompt
    if condition1:
        return prompt.replace(matches1[0], "{TEXT}")
    elif condition2:
        return prompt.replace(pattern2, "{TEXT}")
    
    raise ValueError("Invalid prompt format. Prompt must contain some str/var to be interpolated.")

if __name__ == "__main__":
    prompt = "Determine the probability of the patient to have {TEXT}"
    sample_data = {"text": "heart disease", "output": "The patient has a 23% probability of having heart disease based on their medical history and current symptoms."}
    prompt = check_and_reformat(prompt)
    print(prompt)
    
    res = asyncio.run(create_scoring_prompt(prompt, sample_data))
    print(res)