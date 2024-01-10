import openai
import os
import re

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up the GPT-3 model
model_engine = "gpt-3.5-turbo"
prompt = "Hello, how can I assist you today?"

# Define a function to generate a response
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

def generate_respone_v2(prompt):
    import os
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pattern = "___code____"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are codeGPT. You return runnable code only. Produce"
                                          "an error message if appropriate.}"},
            {"role": "user", "content": f"Produce ONLY executable code for the following prompt. Do not include any explanation or natural language"
                                        # f"Give me a runnable python program for the following question. "
                                        # f""
                                        # f"Surround ONLY the "
                                        # f"executable code block with {pattern} before and after; i.e. make sure that what is surrounded "
                                        # f"by {pattern}  is executable: "
                                        + prompt}
            # {"role": "user", content: "" }
        ]
    )
    re_pattern = fr"{pattern}\s*(.*?)\s*{pattern}"
    match = re.search(re_pattern, str(completion.choices[0].message), re.DOTALL)

    # Code to extract the command
    if match:
        code = match.group(1)
        print('code extracted')
        print("")
        print("this is the extracted code",code.strip())

    # You: write me an endpoint for a flask POST request that accepts image upload, with a ratelimiter
    # Sanitization layer

    # Command to execute the command

    print(str(completion.choices[0].message.content))
    # return completion.choices[0].message


def generate_response_image(prompt):
    import time

    start_time = time.time()
    print('starting API call')
    response = openai.Image.create_edit(
        image=open("14.png", "rb"),
        mask=open("square_mask.png", "rb"),
        prompt=f"Make the area red",
        n=5,
        size="256x256"
    )
    with open(f"{prompt}.txt", "w") as file:
        for item in response['data']:
            file.write(str(item['url']) + "\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

print("image mode selected")
# Get input from the user and generate a response
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    prompt = "\nUser: " + user_input
    response = generate_response_image(prompt)
    print('finished command')

    # print("ChatGPT: " + response)

