import sys
import os
import openai

# Initialize the OpenAI API client with your API key
openai.api_key = os.environ.get("OPENAI_API_KEY", None)
if openai.api_key == None:
    print("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit()

def generate_new_content(file_content, prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 
                                            "You are a coding helper and modify code depenend on the prompts by the user. \
                                            The response should be only the modified code, which starts with: \
                                            [Modified Code]\n<here is the modified code>. \
                                            The user input is separated like this:\n\
                                            [Code]<here is the code>\n\
                                            [Prompt]<here is some instruction what should be changed in the code>"},
            {"role": "user", "content": f"[Code]{file_content}\n[Prompt]{prompt}"}
        ]
    )
    return completion['choices'][0]['message']['content'].replace("[Modified Code]\n", "").replace("[Modified Code]", "")

def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("Please provide a filename as an argument.")
        sys.exit()

    # Read the contents of the file
    with open(filename, 'r') as f:
        file_content = f.read()

    # Prompt the user to modify the file content
    prompt = input("Current content:\n{}\n\nEnter instruction what should be changed:\n".format(file_content))
    if prompt:
        new_content = generate_new_content(file_content, prompt)
        print(f"Here is the new content:\n{new_content}")

        res = input("Do you want to overtake this code (y/n)?")
        if res in ["y", "yes"]:
            with open(filename, 'w') as f:
                f.write(new_content)

if __name__ == '__main__':
    main()
