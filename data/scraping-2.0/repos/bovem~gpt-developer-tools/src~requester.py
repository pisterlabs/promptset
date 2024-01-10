import openai

def send_gpt_request(script_path, line_start, line_end, prompt, open_ai_api_key, model_engine, max_tokens, n, temperature):
    with open(script_path, "r") as script:
        print("\nInput Script: {}".format(script_path))
        print()

        openai.api_key = open_ai_api_key

        prompt+="\n "
        total_lines = sum(1 for line in script)
        line_end = total_lines if line_end==None else line_end

        script.seek(0)

        for i, line in enumerate(script):
            if line_start<=(i+1) and line_end>=(i+1):
                prompt+=line
            elif line_end<(i+1):
                break

        # Generate a response
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=None,
            temperature=temperature,
        )

        response = completion.choices[0].text
        print("ChatGPT's Response:")
        print(response)

def send_gpt_request_create(prompt, write_flag, write_path, open_ai_api_key, model_engine, max_tokens, n, temperature):

    print("\nInput prompt: {}".format(prompt))
    print()

    openai.api_key = open_ai_api_key

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=None,
        temperature=temperature,
    )

    response = completion.choices[0].text

    if write_flag==True:
        script = open(write_path, "w")
        script.write(response)
        script.close()
        print("Program written to file: {}".format(write_path))
    else:
        print("ChatGPT's Response:")
        print(response)