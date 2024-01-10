import os
import openai

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY

def send_question_to_api(selected_philosopher, thought_process, user_question):
    openai.model = "gpt-4"
    thought_process_steps = [{"role": "system", "content": str(step)} for step in thought_process]
    user_content = {"role": "user", "content": user_question}

    formatted_thought_process = "\n".join([value for step in thought_process for _, value in step.items()])

    messages = [{"role": "assistant", "content": f"You are a ThinkerGPT that follows the thought process outlined below. Provide definitive answer at at each step, do not be vague or inconclusive, even if informaiton is limited. \n\n{formatted_thought_process}\n\nYou are tasked with embodying this way of thinking to come up with an answer to a problem in an independent way without any further discussion with the user. You need to go through all of the process outlined below step by step without stopping. If you cannot complete a step, you can say 'Not able to complete' and move on to the next step.\n\nDo not use the full description of the step (stop before :), just show the title of the step, use : sympbol after each step and outline your thinking.\n\nMake sure you arrive at a firm conclusion at the end, do not keep it vague or NEVER say that 'it depends', it must always be a clear answer even if it is difficult to arrive at one. Your answer MUST be definitive."}] 
    messages.extend(thought_process_steps)
    messages.append(user_content)

    completion = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    
    response_content = completion.choices[0].message.content.strip()
    final_answer = response_content.split("\n\n")[-1].strip()

    return response_content, final_answer