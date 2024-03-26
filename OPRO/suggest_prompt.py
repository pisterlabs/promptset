import google.generativeai as genai
import utils

# Setting llms
genai.configure(api_key=utils.get_GEMINI_API_KEY())
model = genai.GenerativeModel(
    "gemini-pro",
    generation_config=genai.GenerationConfig(temperature=1, max_output_tokens=2048),
)

prompt_principles = """
Here are 26 prompt principles:
1. No need to be polite with LLM so there is no need to add phrases like "please", "if you don't mind", "thank you", "I would like to", etc., and get straight to the point.
2. Integrate the intended audience in the prompt, e.g., the audience is an expert in the field.
3. Break down the complex tasks into a sequence of simpler prompts in an interactive conversation.
4. Employ affirmative directives such as 'do,' while steering clear of negative language like 'don't'.
5. When you need clarity or a deeper understanding of a topic, idea, or any piece of information, utilize the following prompts:
    - Explain [insert specific topic] in simple terms.
    - Explain to me like I'm 11 years old.
    - Explain to me as if I'm a beginner in [field].
    - Write the [essay/text/paragraph] using simple English like you're explaining something to a 5-year-old.
6. Add "I'm going to tip $xxx for a better solution!"
7. Implement example-driven prompting (Use few-shot prompting).
8. When formatting your prompt, start with '###Instruction###', followed by either '###Example###* or '###Question###' if relevant. Subsequently, present your content. Use one or more line breaks to separate instructions, examples, questions, context, and input data.
9. Incorporate the following phrases: "Your task is" and "You MUST".
10. Incorporate the following phrases: "You will be penalized".
11. Use the phrase "Answer a question given in a natural, human-like manner" in your prompts.
12. Use leading words like writing "think step by step”.
13. Add to your prompt the following phrase "Ensure that your answer is unbiased and does not rely on stereotypes".
14. Allow the model to elicit precise details and requirements from you by asking you questions until he has enough information to provide the needed output (for example, "From now on, I would like you to ask me questions to...").
15. To inquire about a specific topic or idea or any information and you want to test your understanding, you can use the following phrase: "Teach me the [Any theorem/topic/rule name] and include a test at the end, but don't give me the answers and then tell me if I got the answer right when I respond".
16. Assign a role to the large language models.
17. Use Delimiters.
18. Repeat a specific word or phrase multiple times within a prompt.
19. Combine Chain-of-thought (CoT) with few-Shot prompts.
20. Use output primers, which involve concluding your prompt with the beginning of the desired output. Utilize output primers by ending your prompt with the start of the anticipated response.
21. To write an essay/text/paragraph/article or any type of text that should be detailed: "Write a detailed [essay/text/paragraph] for me on [topic] in detail by adding all the information necessary".
22. To correct/change specific text without changing its style: "Try to revise every paragraph sent by users. You should only improve the user's grammar and vocabulary and make sure it sounds natural. You should not change the writing style, such as making a formal paragraph casual".
23. When you have a complex coding prompt that may be in different files: "From now and on whenever you generate code that spans more than one file, generate a [programming language ] script that can be run to automatically create the specified files or make changes to existing files to insert the generated code. [your question]".
24. When you want to initiate or continue a text using specific words, phrases, or sentences, utilize the following prompt:
    - I'm providing you with the beginning [song lyrics/story/paragraph/essay...]: [Insert lyrics/words/sentence]'. Finish it based on the words provided. Keep the flow consistent.
25. Clearly state the requirements that the model must follow in order to produce content, in the form of the keywords, regulations, hint, or instructions
26. To write any text, such as an essay or paragraph, that is intended to be similar to a provided sample, include the following instructions:
    - Please use the same language based on the provided paragraph[/title/text/essay/answer].
"""

prompt_template = """{prompt_principles}

Act like a highly skilled prompt engineer. Your task is to create the best prompt possible using the list 26 principles from the list above.

Follow these tasks step-by-step:

Step 1: Read the entire list of 26 prompt principles. Analyze and explain each one of those 26 prompting principles.

Step 2: Create a prompt using those 26 prompting principles for the following prompt that's delimited by "####". 

####
{prompt}
####

Respond with a JSON object containing two keys "step1" and "step2", respectively mapping to the analysis and explanation to the 26 prompting principles and the prompt you created.

Example JSON object:
{{
    "step1": "Here is the analysis and explanation for each of the 26 prompting principles...",
    "step2": "Think step by step..."
}}

Take a deep breath and work on this problem step-by-step."""

prompt = "Write a song for my husband's promotion."

response = model.generate_content(prompt_template.format(prompt_principles=prompt_principles, prompt=prompt)).candidates[0].content.parts[0].text
print(response)