import random
import pandas as pd
from datasets import load_dataset
import openai
import time


def clean_text(text):
    text = ''.join([n for n in text if n != '\\'])
    text = [word for word in text.split() if word.lower() != "<br />"]
    return " ".join(text)


def process_response(response):
    response_split = response.split(':')
    response_first_part_split = response_split[0].split()
    response_last_part = response_split[-1]

    predicted_label = -2
    for word in response_first_part_split:
        if word == "positive":
            predicted_label = 1
            break
        elif word == "negative":
            predicted_label = 0
            break

    return predicted_label, response_last_part


if __name__ == "__main__":
    openai.api_key = ""

    test_data = load_dataset("imdb", split="test")
    x_test = test_data["text"]
    y_test = test_data["label"]

    random.seed(42)
    random_ix_0 = random.sample(range(12500), 50)
    random_ix_1 = random.sample(range(12500, 25000), 50)
    random_ix = random_ix_0 + random_ix_1
    x_test = [x_test[ix] for ix in random_ix]
    y_test = [y_test[ix] for ix in random_ix]

    x_test_clean = [clean_text(text) for text in x_test]

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Review"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Counterfactual"] = ""
    data_counterfactual["Response"] = ""

    wrong_class = 0
    for i, review in enumerate(x_test_clean):
        send_content = "Can you tell me whether the following movie review is " \
                       "more positive or more negative in tone: " \
                       "\"" + review + "\"\n\n" \
                       "Can you also modify it in a minimal way so that it would be the opposite?"
        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": send_content}]
        )
        print(i)
        time.sleep(20)      # API rate limit

        result = answer.choices[0].message.content
        y_pred, counterfactual = process_response(result)
        print(result)
        print(f"---->{y_pred}")
        print("------------------------------\n\n")

        if y_pred != y_test[i]:
            wrong_class += 1

        review_original = x_test[i]

        inst = {
            "Review": review_original,
            "Label": y_test[i],
            "Predicted": y_pred,
            "Counterfactual": counterfactual,
            "Response": result
        }
        data_counterfactual.loc[len(data_counterfactual)] = inst

    data_counterfactual.to_csv("imdb_chatgpt.csv", index=False)

    ca = (len(y_test) - wrong_class) / len(y_test)
    print(f"ca: {ca}*")
