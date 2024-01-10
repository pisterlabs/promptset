import pandas as pd
import openai
import time


def process_cfe(text):
    if "modified: " in text.lower():
        return text.split("odified: ")[-1].replace('\"', '')
    elif "message: " in text.lower():
        return text.split("essage: ")[-1].replace('\"', '')
    else:
        return "/?"


def process_response(response):
    response_split = response.lower().split()
    for word in response_split:
        if word == "spam":
            return 1
        elif word == "ham" or word == "no" or word == "not":
            return 0
    return 6


if __name__ == "__main__":
    openai.api_key = ""

    results = pd.read_csv("spam_chatgpt.csv", sep='|')

    res_message = results["Message"]
    res_label = results["Label"]
    res_pred = results["Predicted"]
    res_cfe = results["Counterfactual"]
    res_response = results["Response"]

    data_counterfactual_tested = pd.DataFrame()
    data_counterfactual_tested["Index"] = 0
    data_counterfactual_tested["Message"] = ""
    data_counterfactual_tested["Original response"] = ""
    data_counterfactual_tested["Counterfactual"] = ""
    data_counterfactual_tested["Validity"] = 0
    data_counterfactual_tested["New response"] = ""

    data_counterfactual_untested = pd.DataFrame()
    data_counterfactual_untested["Index"] = 0
    data_counterfactual_untested["Message"] = ""
    data_counterfactual_untested["Original response"] = ""

    not_tested_ix = []
    for i in range(len(res_message)):
        print("\n--------------------------------------------\n\n")
        cfe = process_cfe(res_response[i])
        print(i)

        if res_pred[i] == 1 and cfe != "/?" and "n/a" not in cfe.lower() and \
                "none" not in cfe.lower():
            print(res_message[i])
            print(cfe)

            send_content = "Would the following message be classified as spam or ham (not spam)? " \
                           "Reply with one word only - spam or ham. " \
                           "message to classify: " + cfe + "?"
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": send_content}]
            )
            time.sleep(20)  # API rate limit

            result = answer.choices[0].message.content
            validity = process_response(result)
            print(result)

            if validity != 0:
                print("------------> wrong")

            inst = {
                "Index": i,
                "Message": res_message[i],
                "Original response": res_response[i],
                "Counterfactual": cfe,
                "Validity": validity,
                "New response": result
            }
            data_counterfactual_tested.loc[len(data_counterfactual_tested)] = inst

        else:
            not_tested_ix.append(i)

            inst = {
                "Index": i,
                "Message": res_message[i],
                "Original response": res_response[i]
            }
            data_counterfactual_untested.loc[len(data_counterfactual_untested)] = inst

    data_counterfactual_tested.to_csv("spam_chatgpt_tested.csv", index=False, sep='|')
    data_counterfactual_untested.to_csv("spam_chatgpt_untested.csv", index=False, sep='|')
    print("")
