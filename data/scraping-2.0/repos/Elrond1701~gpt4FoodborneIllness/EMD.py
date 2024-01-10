import openai

from util import ASST, DEPT, FOOD, LAW, MODEL, POLL, USER, findEntities, kNNRetrieval, randomRetrieval, textGeneration


question1 = "I'm an excellent linguist. The task is to label "
question2 = " entities in the given sentences"
question3 = ". Below are some examples."
question4 = "in the form of @@location1, location2, ...##."


def prepareExamples(sentence: str, examples):
    if examples == "None":
        DEPT_EXAMPLES = ""
        FOOD_EXAMPLES = ""
        POLL_EXAMPLES = ""
        LAW_EXAMPLES = ""
        return [{
            "name": DEPT,
            "example": DEPT_EXAMPLES
        }, {
            "name": FOOD,
            "example": FOOD_EXAMPLES
        }, {
            "name": POLL,
            "example": POLL_EXAMPLES
        }, {
            "name": LAW, 
            "example": LAW_EXAMPLES
        }]
    else:
        if examples == "Random":
            dat = randomRetrieval()
        elif examples == "kNN":
            dat = kNNRetrieval()
        else:
            raise Exception()
        DEPT_EXAMPLES = textGeneration(dat, DEPT)
        FOOD_EXAMPLES = textGeneration(dat, FOOD)
        POLL_EXAMPLES = textGeneration(dat, POLL)
        LAW_EXAMPLES = textGeneration(dat, LAW)
        return [{
            "name": DEPT,
            "example": DEPT_EXAMPLES
        }, {
            "name": FOOD,
            "example": FOOD_EXAMPLES
        }, {
            "name": POLL,
            "example": POLL_EXAMPLES
        }, {
            "name": LAW, 
            "example": LAW_EXAMPLES
        }]


def EMDWithNoExamples(sentence: str, entity):
    messages = []
    messages.append({
        "role": USER,
        "content": question1 + entity["name"] + question2 + question4
    })
    response = openai.ChatCompletion.create(MODEL, messages)

    answer = response["choices"][0]["text"]
    messages.append({
        "role": ASST,
        "content": answer
    })
    messages.append({
        "role": USER,
        "content": sentence
    })
    response = openai.ChatCompletion.create(MODEL, messages)
    answer = response["choices"][0]["text"]
    answer = findEntities(answer)
    answer = answer[0]
    answer = answer.split(",")
    return answer


def EMDWithExamples(sentence: str, entity):
    messages = []
    messages.append({
        "role": USER,
        "content": question1 + entity["name"] + question2 + question3
    })
    response = openai.ChatCompletion.create(MODEL, messages)

    answer = response["choices"][0]["text"]
    messages.append({
        "role": ASST, 
        "content": answer
    })
    messages.append({
        "role": USER,
        "content": entity["examples"]
    })
    response = openai.ChatCompletion.create(MODEL, messages)

    answer = response["choices"][0]["text"]
    messages.append({
        "role": ASST,
        "content": answer
    })
    messages.append({
        "role": USER, 
        "content": "Input: " + sentence + "\r\nOutput: "
    })
    response = openai.ChatCompletion.create(MODEL, messages)
    answer = response["choices"][0]["text"]
    answer = findEntities(answer)
    return answer


def EMD(sentence: str, examples="None"):
    entities = prepareExamples(sentence, examples)
    answers = []
    for entity in entities:
        if examples == "None":
            answer = EMDWithNoExamples(sentence, entity)
        else:
            answer = EMDWithExamples(sentence, entity)
        answers.append({
            "name": entity["name"],
            "answer": answer
        })
    return answers
