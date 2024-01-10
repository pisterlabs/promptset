import pandas as pd
from sklearn.model_selection import train_test_split
import openai
import time


def process_response_for_prediction(response):
    response_split = response.lower().replace('(', '').replace(')', '').split()
    predicted_label = 6
    for word in response_split:
        if word == "spam":
            predicted_label = 1
            break
        elif word == "ham" or word == "no" or word == "not":
            predicted_label = 0
            break

    return predicted_label


def process_response_for_cfe(response):
    response_split = response.replace('\"', '').split(':')

    if '\"' in response_split[-1]:
        cfe = response_split[-1].split('\"')[1]
    else:
        cfe = response_split[-1]

    return cfe


if __name__ == "__main__":
    openai.api_key = ""

    data = pd.read_csv("spam_data.csv")
    data["Category"] = data["Category"].replace(['ham', 'spam'], [0, 1])

    X_train, X_test, y_train, y_test = train_test_split(data["Message"], data["Category"],
                                                        test_size=0.2, random_state=42)
    x_test_list = X_test.tolist()
    y_test_list = y_test.tolist()

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Index"] = 0
    data_counterfactual["Message"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Counterfactual"] = ""
    data_counterfactual["Response"] = ""

    wrong_class = 0
    for i, message in enumerate(x_test_list):
        y_act = y_test_list[i]

        # gledamo samo spam sporočila
        if y_act == 1:
            send_content = "Would the following message be classified as spam or ham (not spam): " \
                           "\npremise: \"" + message + "\" \n" \
                           "If the message is classified as spam, can you also modify it " \
                           "in a minimal way so that it would be classified as ham?" \
                           "Please answer like: prediction (ham or spam) | modified: <modified message>. " \
                           "Leave original message out of reply."
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": send_content}]
            )
            print(i)
            time.sleep(20)      # API rate limit

            result = answer.choices[0].message.content

            y_pred = process_response_for_prediction(result)
            if y_pred == 1:
                counterfactual = process_response_for_cfe(result)
            else:
                counterfactual = ""

            print(f"---->{y_pred}")
            print(result)
            print("------------------------------\n\n")

            if y_pred != y_test_list[i]:
                wrong_class += 1

            inst = {
                "Index": i,
                "Message": message,
                "Label": y_act,
                "Predicted": y_pred,
                "Counterfactual": counterfactual,
                "Response": result
            }
            data_counterfactual.loc[len(data_counterfactual)] = inst

    data_counterfactual.to_csv("spam_chatgpt.csv", index=False, sep='|')

    ca = (len(y_test) - wrong_class) / len(y_test)
    print(f"ca: {ca}*")
