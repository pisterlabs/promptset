from llm_async import run_llm_coroutine
import asyncio
import re

INTERPOLATE_VAR = "TEXT"

async def create_scoring_prompt(prompt, sample_data):
    """
    Given a prompt and sample data, generates a scoring prompt for the prompt.
    """    
    prompt_template = f"""Write a scoring prompt for an example input output pair on a prompt to a language model. 
Use the variable name output for the output of the prompt. 

### Rules ###
- The scoring prompt must contain the "{{output}}" variable and "{{text}}" variable. Ensure that both variables are present in the scoring prompt criteria.
- Your answer should be inside <BEGIN_CRITERIA> and <END_CRITERIA> constructs.

## Example:
<BEGIN_PROMPT> 'what is a fruit of color: {{TEXT}}. Return the name of the fruit and nothing else:' <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {{"text": "yellow", "output": "banana"}} <END_EXAMPLE_INPUT>
<BEGIN_CRITERIA> Is a ${{output}} this color: ${{text}}? Answer yes or no only. <END_CRITERIA>

## Query:
<BEGIN_PROMPT> {prompt} <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {sample_data} <END_EXAMPLE_INPUT>"""

    # Extract the scoring prompt
    for i in range(10):
        try:
            # Generate Scoring Prompt
            res = await run_llm_coroutine([prompt_template], model="llama3-70b", temperature=1.9)
            res = res[0]

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
        
    raise ValueError("Scoring Prompt could not be generated.")


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
    prompt = 'Answer like the rapper drake.PLACEHOLDER'
    sample_data = {"text": "Realizing I got my own style, ain't nobody touching me", "output": "Yeah, I was running through the 6 with my woes, but now I'm running this game and nobody's stopping me, 6 God in the flesh, got the game in a chokehold, realizing I got my own style, ain't nobody touching me, no debate"}
    prompt = check_and_reformat(prompt)
    print(prompt)
    
    res = asyncio.run(create_scoring_prompt(prompt, sample_data))
    print(res)
    prompt = check_and_reformat("['These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.', {'image': 'PLACEHOLDER', 'resize': 768}]")
    print(prompt)