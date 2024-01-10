from blocks.models import Answer, ChatGPTGradeAndFeedback
from django.db.models import Max

import openai
import time

from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')

# Set up Open AI
openai.api_key = openai_key

def chatgpt_question_only(question, answer, max_grade):
    prompt = [
                {"role": "system", "content": f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer."},
                {"role": "user", "content": f"{answer}"},
            ]
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["message"]["content"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def gpt3_question_only(question, answer, max_grade):
    prompt=f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer and feedback/reasoning for the grade so the student can learn from it.\nStudent: {answer}\nTeacher:"
    model="text-davinci-003"
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            stop= ["\nStudent:", "\nTeacher:"]
        )

        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["text"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

examples = {
    '23.0': [("1)  To understand what a certain sector of user would want.  If you approach them with a potential problem, would they agree it's a problem that needs a solution?\n2) Maybe you want to do observation; if your app requires you to spectate people naturally on how they do certain activities, observation is necessary\n3) Maybe you want to understand certain user preferences, then interviewing them is needed", 2.0, "You hit the main points of understanding users needs and validating user preferences but missed the idea of getting feedback/evaluating prototype. Your 1st and 3rd point are similar in concept."), ("In inspiration, diverge the ideas from the user research\nIn ideation,  converge and diverge the ideas\nIn implementation, converge the ideas to be specific.", 0.0, "You focused on the IDEO process instead of considering the use of user research as a whole.")],
    '27.0': [("We want to use Components to create instances in Figma because using components allows us to reuse and easily modify design elements within a design project. ", 1.0, "You are right with the reusability aspect but you didn't mention the second keypoint of consistency"), ("Because they can be reused to increase the efficiency for repeated work. Also, it can help to keep the same design style across the same platform.", 2.0, "Good job! You concisely described why we use components through the key concepts of reusability and consistency.")],
    '7.0': [("Asynchronous processing allows for updating only part of the page, especially if update is taking a long time. It's especially helpful for server requests when server is slow", 0.0, "You have the right idea of the benefits of asynchronous processing but not in the context of why it's useful to users on the frontend. We would expect to see more here about real-time content rendering, not blocking the user flow, or allowing user interaction at any time."), ("Asynchronous programming in Javascript allows us to partially update a webpage without having to refresh the whole page. Additionally, it allows us to execute instructions that can take a long time to execute such as waiting for HTTP responses, without blocking the rest of the code, which allows users to still interact with the user interface.", 2.0, "This is a great answer! You've hit the keypoints of not needing to reload the entire webpage, and allowing the user to interact without blockage.")]}

rubrics = {
    '23.0': ["Understanding user needs", "Validating user needs, probe into users' preferences around potential solutions in ideation", "Evaluating or getting feedback on ideas/prototypes"],
    '27.0': ["Clearly states concept of consistency", "Clearly states concept of reusability"],
    '7.0': ["Allow user interaction at any time", "Allow query data from server/API without disrupting user flow", "Render content on the webpage in real-time", "Javascript is single-threaded"]}

def chatgpt_examples(question, answer, max_grade, examples):
    prompt = [
                {"role": "system", "content": f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer."},
            ]
    for ex in examples:
        prompt.append({"role": "user", "content": f"{ex[0]}"})
        prompt.append({"role": "assistant", "content": f"Grade: {ex[1]} Feedback: {ex[2]}"})
    prompt.append({"role": "user", "content": f"{answer}"})  # add the final student answer to the back
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["message"]["content"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def gpt3_examples(question, answer, max_grade, examples):
    prompt=f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer and feedback/reasoning for the grade so the student can learn from it.\nStudent: {examples[0][0]}\nTeacher:Grade: {examples[0][1]} Feedback: {examples[0][2]}\nStudent: {examples[1][0]}\nTeacher:Grade: {examples[1][1]} Feedback: {examples[1][2]}\nStudent: {answer}\nTeacher:"
    model="text-davinci-003"
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            stop= ["\nStudent:", "\nTeacher:"]
        )

        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["text"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def chatgpt_rubrics(question, answer, max_grade, rubrics):
    prompt = [
                {"role": "system", "content": f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. The question has the following rubrics: {', '.join(rubrics)}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer."},
                {"role": "user", "content": f"{answer}"},
            ]
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["message"]["content"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def gpt3_rubrics(question, answer, max_grade, rubrics):
    prompt=f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. The question has the following rubrics: {', '.join(rubrics)}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer and feedback/reasoning for the grade so the student can learn from it.\nStudent: {answer}\nTeacher:"
    model="text-davinci-003"
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            stop= ["\nStudent:", "\nTeacher:"]
        )

        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["text"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def chatgpt_rubrics_and_examples(question, answer, max_grade, examples, rubrics):
    prompt = [
                {"role": "system", "content": f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. The question has the following rubrics: {', '.join(rubrics)}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer."},
            ]
    for ex in examples:
        prompt.append({"role": "user", "content": f"{ex[0]}"})
        prompt.append({"role": "assistant", "content": f"Grade: {ex[1]} Feedback: {ex[2]}"})
    prompt.append({"role": "user", "content": f"{answer}"})  # add the final student answer to the back
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["message"]["content"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def gpt3_rubrics_and_examples(question, answer, max_grade, examples, rubrics):
    prompt=f"You are a professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. The question has the following rubrics: {', '.join(rubrics)}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer and feedback/reasoning for the grade so the student can learn from it.\nStudent: {examples[0][0]}\nTeacher:Grade: {examples[0][1]} Feedback: {examples[0][2]}\nStudent: {examples[1][0]}\nTeacher:Grade: {examples[1][1]} Feedback: {examples[1][2]}\nStudent: {answer}\nTeacher:"
    model="text-davinci-003"
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            stop= ["\nStudent:", "\nTeacher:"]
        )

        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["text"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

curr_trial_number = ChatGPTGradeAndFeedback.objects.aggregate(Max('trial_run_number'))['trial_run_number__max'] + 1
print(f"Trial Run #{curr_trial_number}")

starting_index = 0
all_answers = list(Answer.objects.all())
methods = ["question_only", "examples", "rubrics", "rubrics_and_examples"]
max_grades = {'23.0':3.0, '27.0':2.0, '7.0':2.0}

for index, answer in enumerate(all_answers[starting_index:]):
    print(f"Creating chatgpt response: {index+1+starting_index}/{len(all_answers)}")
    curr_qid = answer.question.question_exam_id
    # # ---------------------
    # # GPT3 QUESTION_ONLY
    # chatgpt_response, prompt, model = gpt3_question_only(answer.question.question_text, answer.answer_text, max_grades[curr_qid])
    # time.sleep(5)
    # if (chatgpt_response == "ERROR"):
    #     print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
    #     time.sleep(60)
    #     chatgpt_response, prompt, model = gpt3_question_only(answer.question.question_text, answer.answer_text, max_grades[curr_qid])
    # if (chatgpt_response != "ERROR"):
    #     ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type="question_only",trial_run_number=curr_trial_number,openai_model=model)
    # # ---------------------
    # # GPT3 EXAMPLES
    # prompt_type="examples"
    # chatgpt_response, prompt, model = gpt3_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], examples=examples[curr_qid])
    # time.sleep(5)
    # if (chatgpt_response == "ERROR"):
    #     print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
    #     time.sleep(60)
    #     chatgpt_response, prompt, model = gpt3_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], examples=examples[curr_qid])
    # if (chatgpt_response != "ERROR"):
    #     ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # # ---------------------
    # # GPT3 RUBRICS
    # prompt_type="rubrics"
    # chatgpt_response, prompt, model = gpt3_rubrics(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid])
    # time.sleep(5)
    # if (chatgpt_response == "ERROR"):
    #     print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
    #     time.sleep(60)
    #     chatgpt_response, prompt, model = gpt3_rubrics(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid])
    # if (chatgpt_response != "ERROR"):
    #     ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # GPT3 RUBRICS AND EXAMPLES
    prompt_type="rubrics_and_examples"
    chatgpt_response, prompt, model = gpt3_rubrics_and_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid], examples=examples[curr_qid])
    time.sleep(5)
    if (chatgpt_response == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = gpt3_rubrics_and_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid], examples=examples[curr_qid])
    if (chatgpt_response != "ERROR"):
        ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # CHATGPT QUESTION_ONLY
    chatgpt_response, prompt, model = chatgpt_question_only(answer.question.question_text, answer.answer_text, max_grades[curr_qid])
    time.sleep(5)
    if (chatgpt_response == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = chatgpt_question_only(answer.question.question_text, answer.answer_text, max_grades[curr_qid])
    if (chatgpt_response != "ERROR"):
        ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type="question_only",trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # CHATGPT EXAMPLES
    prompt_type="examples"
    chatgpt_response, prompt, model = chatgpt_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], examples=examples[curr_qid])
    time.sleep(5)
    if (chatgpt_response == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = chatgpt_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], examples=examples[curr_qid])
    if (chatgpt_response != "ERROR"):
        ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # CHATGPT RUBRICS
    prompt_type="rubrics"
    chatgpt_response, prompt, model = chatgpt_rubrics(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid])
    time.sleep(5)
    if (chatgpt_response == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = chatgpt_rubrics(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid])
    if (chatgpt_response != "ERROR"):
        ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # CHATGPT RUBRICS AND EXAMPLES
    prompt_type="rubrics_and_examples"
    chatgpt_response, prompt, model = chatgpt_rubrics_and_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid], examples=examples[curr_qid])
    time.sleep(5)
    if (chatgpt_response == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = chatgpt_rubrics_and_examples(answer.question.question_text, answer.answer_text, max_grades[curr_qid], rubrics=rubrics[curr_qid], examples=examples[curr_qid])
    if (chatgpt_response != "ERROR"):
        ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type=prompt_type,trial_run_number=curr_trial_number,openai_model=model)
    # ---------------------
    # BREAK CONDITION
    if (index > 300): break

