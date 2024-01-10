import openai
import anki_vector

openai.api_key = "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"

injection = "I'd like you to pretend to be a small robot called Vector. You are cheerful and friendly. Please respond to the prompt: "

def getData(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": injection + prompt}])
    answer = completion.choices[0].message.content
    print(answer)
    return answer

def main():
    args = anki_vector.util.parse_command_args()

    conversation = ''
    
    with anki_vector.Robot(args.serial) as robot:
        prompt = "What is transpiration?"
        conversation = conversation + "\n\n" +  prompt
        answer = getData(prompt)
        conversation = conversation + "\n\n" + answer
        robot.behavior.say_text(answer)



        # prompt = "What is your favourite number?"
        # conversation = conversation + "\n\n" +  prompt
        # answer = getData(prompt)
        # conversation = conversation + "\n\n" + answer
        # robot.behavior.say_text(answer)


if __name__ == "__main__":
    main()
