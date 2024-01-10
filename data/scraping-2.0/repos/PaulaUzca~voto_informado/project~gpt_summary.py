import openai

def get_chatgpt_response(dump_json):
    # Initialize OpenAI API key
    api_key = ""
    # / new
    # / old
    # Initialize GPT model to be used, either "gpt-3.5-turbo" for GPT-3.5 or the equivalent for GPT-4
    model_engine = "gpt-3.5-turbo"
    #gpt-4 / expensive
    #gpt-3.5-turbo	/ cheaper
    # System message
    system_message = {
        "role": "system",
        "content": """
        You're assisting users in profiling a political candidate for the Colombian elections, focusing more on negative aspects.
        ## Objectives
        1. Summarize the candidate based on web-scraped data.
        2. Highlight their achievements and qualifications.
        3. Detail their negative news involvement, such as corruption.
        4. Ensure factual, unbiased information.
        5. If data is insufficient for a fair view, state it.

        ## Constraints
        - Respond in SPANISH.
        - Stay factual, no opinions.
        - Exclude citations; they'll be added later.
        - Remind users to verify the information from primary sources.
        - Mention this tool's educational intent and no political biases.

        ## Instructions
        1. Begin by stating the summary uses web-scraped data but don't mention the JSON file.
        2. Outline the candidate's qualifications and achievements.
        3. Mention negative news and specifics.
        4. If data lacks, mention the insufficiency.
        5. Urge users to cross-check from original sources.
        6. Conclude by emphasizing the tool's educational nature and neutrality.
        """
    }

    # Instruction message
    instruction_message = {
        "role": "user",
        "content": f"""Analyze the provided JSON data related to the candidate
        - Extract relevant details about the candidate for the Colombian elections, highlighting both achievements and negative news.
        - Summarize the candidate in a balanced manner.
        - Ensure data cleanliness, correct or ignore inconsistencies.
        - State explicitly if data is incomplete.
        - Use the JSON block as the primary source.

        ## Data JSON: 
        {dump_json}

        Afterwards, craft a concise Spanish paragraph profiling the candidate, emphasizing negative aspects if present."""
 }


    # Prepare the messages
    messages = [system_message, instruction_message]

    # Make API call
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
        api_key=api_key
    )

    # Extract and print generated text
    generated_text = response['choices'][0]['message']['content'].strip()
    return generated_text