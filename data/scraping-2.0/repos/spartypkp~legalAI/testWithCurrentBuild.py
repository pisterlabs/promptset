import json
import promptStorage as prompts
import openai

def main():
    pass

def read_all_test_queries():
    with open("testQueries.txt") as queries_file:
        text = queries_file.read()
    list_of_queries = queries_file.split(",")
    return list_of_queries

def test_all_questions(user_query, legal_text, template, answer_list):
    questions_list = template.split("\n")
    # This is stupid as fuck and I love it
    try:
        while True:
            questions_list.remove(" ")
    except:
        pass
    try:
        while True:
            questions_list.remove("")
    except:
        pass
    print(questions_list)
    used_model = "gpt-4-32k"
    prompt_score_questions = prompts.get_prompt_score_questions(legal_text, questions_list, answer_list)
    chat_completion = openai.ChatCompletion.create(model=used_model, messages=prompt_score_questions)
    result = chat_completion.choices[0].message.content
    print(result)
    exit(1)
    relevance_scores = [0]
    answer_scores = [0]

    with open("testQueries.txt","r") as read_file:
        text = read_file.read()
        test_dict = json.loads(text)
    read_file.close()
    if user_query in test_dict:
        copy = test_dict[user_query]
        copy[legal_text] = legal_text
        copy["questions"][0]["best_answer"] = answer_list[0]
        copy["questions"][0]["best_answer_metadata"]["relevance_score"] = relevance_scores[0]
        copy["questions"][0]["best_answer_metadata"]["answer_score"] = answer_scores[0]
 

if __name__ == "__main__":
    main()