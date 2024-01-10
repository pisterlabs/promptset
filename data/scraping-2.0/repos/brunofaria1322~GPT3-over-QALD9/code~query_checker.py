import dataset_parsing as dp
import json
import openai
import pickle
import time

from SPARQLWrapper import SPARQLWrapper, JSON


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def ask_gpt3(question, expected_answer, engine="text-davinci-002"):
    # set hyperparameters
    if expected_answer is None:
        answer_length = 1000
    else:
        answer_length = len(str(expected_answer)) * 2
    # GPT3
    openai.api_key = OPENAI_API_KEY
    if engine == "davinci_ft":
        print(question)
        response = openai.Completion.create(
            engine="davinci:ft-personal-2023-03-06-17-05-09",
            prompt=question,
            temperature=0,
            max_tokens=answer_length,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop="\n<EOQ>\n",
        )
    else:
        response = openai.Completion.create(
            engine=engine,
            prompt=question,
            temperature=0,
            max_tokens=answer_length,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            # don't put stop words (SPARQL queries have \n)
        )
    return response["choices"]


def get_gpt3_queries(
    questions,
    queries,
    kind="test",
    overwrite=False,
    num_shots=1,
    engine="text-davinci-002",
):
    engine_folder_name = (
        f"gpt3_{engine.split('-')[1]}{engine.split('-')[2]}"
        if engine != "davinci_ft"
        else "gpt3_davinci_ft"
    )
    if num_shots > 1:
        path = f"output/{engine_folder_name}-fs{num_shots}/{engine_folder_name}-fs{num_shots}"
        gpt3_questions = [
            ("Q1", 'The SPARQL query for the question "', '" is '),
            ("Q2", 'What is the SPARQL query for the question "', '"?'),
            ("Q3", 'SPARQL for "', '" is '),
            ("Q4", "Write the complete SPARQL query to answer the question : ", ""),
            ("Q5", 'Turn this into a SPARQL query: "', '"'),
            ("Q6", 'The DBpedia SPARQL query for the question "', '" is '),
            ("Q7", 'What is the DBpedia SPARQL query for the question "', '"?'),
            ("Q8", 'The DBpedia SPARQL for "', '" is '),
            (
                "Q9",
                "Write the complete DBpedia SPARQL query to answer the question : ",
                "",
            ),
            ("Q10", 'Turn this into a DBpedia SPARQL query: "', '"'),
        ]
        with open(f"data/fewshot_prompt.json", "r", encoding="utf-8") as f:
            json_file = json.load(f)
    else:
        path = f"output/{engine_folder_name}/{engine_folder_name}"
    try:
        with open(f"{path}-{kind}-query_dict.pkl", "rb") as f:
            query_dict = pickle.load(f)
    except FileNotFoundError:
        print("No query_dict found")
        if engine != "davinci_ft":
            query_dict = {"Q" + str(i): [] for i in range(1, 11)}
        else:
            query_dict = {"Q1": []}
    if overwrite:
        if engine != "davinci_ft":
            query_dict = {"Q" + str(i): [] for i in range(1, 11)}
        else:
            query_dict = {"Q1": []}
    length = len(query_dict["Q1"])
    # get the queries
    for i, question in enumerate(questions[length:]):
        if num_shots > 1:
            for id, begin_gpt3_question, end_gpt3_question in gpt3_questions:
                prompt = ""
                for example in json_file:
                    if engine != "davinci_ft":
                        prompt += (
                            begin_gpt3_question
                            + example["question"]
                            + end_gpt3_question
                            + "\n"
                            + example["query"]
                            + "\n\n"
                        )
                    else:
                        prompt += (
                            begin_gpt3_question
                            + example["question"]
                            + end_gpt3_question
                            + "\n ->"
                            + example["query"]
                            + "\n<EOQ>\n\n"
                        )
                prompt += begin_gpt3_question + question + end_gpt3_question + "\n"
                time.sleep(API_WAITING_TIME)
                print("=====================================")
                print(f"Question {length+i}")
                print(f"GPT3 question: {prompt}\n")
                query_dict[id].append(
                    ask_gpt3(prompt, queries[length + i], engine=engine)
                )
                print(query_dict[id][-1])
                print("=====================================")
        else:
            if engine != "davinci_ft":
                gpt3_questions = [
                    ("Q1", f'The SPARQL query for the question "{question}" is '),
                    (
                        "Q2",
                        f'What is the SPARQL query for the question "{question}"?\n',
                    ),
                    ("Q3", f'SPARQL for "{question}" is '),
                    (
                        "Q4",
                        f"Write the complete SPARQL query to answer the question : {question}",
                    ),
                    ("Q5", f'Turn this into a SPARQL query: "{question}"'),
                    (
                        "Q6",
                        f'The DBpedia SPARQL query for the question "{question}" is ',
                    ),
                    (
                        "Q7",
                        f'What is the DBpedia SPARQL query for the question "{question}"?\n',
                    ),
                    ("Q8", f'The DBpedia SPARQL for "{question}" is '),
                    (
                        "Q9",
                        f"Write the complete DBpedia SPARQL query to answer the question : {question}",
                    ),
                    ("Q10", f'Turn this into a DBpedia SPARQL query: "{question}"'),
                ]
            else:
                gpt3_questions = [
                    ("Q1", f"{question} ->"),
                ]
            print(gpt3_questions)
            for id, gpt3_question in gpt3_questions:
                time.sleep(API_WAITING_TIME)
                print("=====================================")
                print(f"Question {length+i}")
                print(f"GPT3 question: {gpt3_question}\n")
                query_dict[id].append(
                    ask_gpt3(gpt3_question, queries[length + i], engine=engine)
                )
                print(query_dict[id][-1])
                print("=====================================")
        with open(f"{path}-{kind}-query_dict.pkl", "wb") as f:
            pickle.dump(query_dict, f)
    return query_dict


def query_checker(query):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    print(query)
    # gets the first 3 geological ages
    # from a Geological Timescale database,
    # via a SPARQL endpoint
    sparql.setQuery(query)

    try:
        ret = sparql.queryAndConvert()
        print("\n=========================================\n")
        print(ret)
        print("\n=========================================\n")
        try:
            if "boolean" in ret.keys():
                # get the answer from the boolean
                return ("boolean", [ret["boolean"]])

            elif "results" in ret.keys():
                # get vars
                try:
                    vars = ret["head"]["vars"]
                except Exception as e:
                    assert False, f"No vars in query: {ret}\n{e}"

                # check the kind of answer
                bindings = ret["results"]["bindings"]

                if len(bindings) == 0:
                    print("No results found")
                    return ("empty", [])

                for key in vars:
                    if key in bindings[0]:
                        # get the answer(s) from the bindings
                        return (
                            bindings[0][key]["type"],
                            [binding[key]["value"] for binding in bindings],
                        )

            else:
                assert False, f"No key found"
        except Exception as e:
            assert False, f"No answer found -- {e}"
    except Exception as e:
        return ("error", [e])


def ckeck_qald(kind="test"):
    path_stats = f"output/stats/qald9/qald9-{kind}"
    qald = dp.read_qald(f"data/qald_9_{kind}.json")
    ids, questions, keywords, queries, answers = dp.parse_qald(qald)
    answers_dict = {}
    count_dict = {}
    for i, query in enumerate(queries):
        print(f"Query {i}: {query}")
        key, val = query_checker(query)
        if key == "error":
            print(val)
            assert False
        elif key == "empty":
            print("No results found")
        elif key == "boolean":
            print(val)
        elif "-" in val[0]:
            key = "date"
            print(val)
        elif is_float(val[0]):
            key = "number"
            print(val)
        if key in answers_dict.keys():
            answers_dict[key].append(val)
            count_dict[key] += 1
        else:
            answers_dict[key] = [val]
            count_dict[key] = 1
    with open(f"{path_stats}-answers_dict.pkl", "wb") as f:
        pickle.dump(answers_dict, f)
    with open(f"{path_stats}-count_dict.pkl", "wb") as f:
        pickle.dump(count_dict, f)


def check_gpt3(
    isFewShot=False,
    kind="test",
    get_pickle=True,
    overwrite=False,
    engine="text-davinci-002",
):
    engine_folder_name = (
        f"gpt3_{engine.split('-')[1]}{engine.split('-')[2]}"
        if engine != "davinci_ft"
        else "gpt3_davinci_ft"
    )
    if isFewShot:
        assert kind == "test", "Only test set is available for few-shot"
        shots = 5
        path = f"output/{engine_folder_name}-fs{shots}/{engine_folder_name}-fs{shots}-{kind}"
        path_stats = f"output/stats/{engine_folder_name}-fs{shots}/{engine_folder_name}-fs{shots}-{kind}"
    else:
        shots = 1
        path = f"output/{engine_folder_name}/{engine_folder_name}-{kind}"
        path_stats = f"output/stats/{engine_folder_name}/{engine_folder_name}-{kind}"

    qald = dp.read_qald(f"data/qald_9_{kind}.json")
    ids, questions, keywords, queries, answers = dp.parse_qald(qald)
    if not get_pickle:
        query_dict = get_gpt3_queries(
            questions, queries, num_shots=shots, overwrite=overwrite, engine=engine
        )
    else:
        with open(f"{path}-query_dict.pkl", "rb") as f:
            query_dict = pickle.load(f)

    answers_dict = {}
    count_dict = {}

    for query_id in query_dict.keys():
        query_text = [choice[0]["text"] for choice in query_dict[query_id]]
        count_dict = {}
        answers_dict = {}
        for i, query in enumerate(query_text):
            print(f"Query {i}: {query}")
            try:
                key, val = query_checker(query)
            except TypeError:
                key, val = "error", None
            if key == "error":
                print(val)
            elif key == "empty":
                print("No results found")
            elif key == "boolean":
                print(val)
            elif "-" in val[0]:
                key = "date"
                print(val)
            elif is_float(val[0]):
                key = "number"
                print(val)
            if key in answers_dict.keys():
                answers_dict[key][ids[i]] = val
                count_dict[key] += 1
            else:
                answers_dict[key] = {ids[i]: val}
                count_dict[key] = 1
        with open(f"{path_stats}-{query_id}-answers_dict.pkl", "wb") as f:
            pickle.dump(answers_dict, f)
        with open(f"{path_stats}-{query_id}-count_dict.pkl", "wb") as f:
            pickle.dump(count_dict, f)


def main():
    """Main function"""
    #
    """DAVINCI-002"""
    # ckeck_qald(kind="train")
    # check_gpt3(kind="train", get_pickle=True)                                                          # new version with 10 questions
    # ckeck_qald(kind="test")
    # check_gpt3(kind="test", get_pickle=False)                                                          # new version with 10 questions
    # check_gpt3(isFewShot=True, kind="test", get_pickle=True)                                           # new version with 10 questions
    """DAVINCI - 003"""
    # check_gpt3(kind="test", get_pickle=True, engine="text-davinci-003")                                # new version with 10 questions
    # check_gpt3(isFewShot=True, kind="test", get_pickle=True, engine="text-davinci-003")                # new version with 10 questions
    """FINE-TUNED"""
    # check_gpt3(kind="test", get_pickle=True, engine="davinci_ft")                                      # new version
    # check_gpt3(isFewShot=True, kind="test", get_pickle=False, engine="davinci_ft", overwrite=True)


if __name__ == "__main__":
    # min time between requests (3 seconds)
    API_WAITING_TIME = 3.1
    with open("data/key.txt", "r") as f:
        OPENAI_API_KEY = f.readline()
    main()
