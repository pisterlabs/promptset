import openai

openai.api_key = "your_key"


def get_gpt_inference(data):
    list_of_preds = []
    for i in len(data):
        list_of_preds.append(
            openai.Completion.create(
                model="babbage:ft-mcgill-2022-12-15-12-39-35", prompt=i
            )["choices"][0]["text"].replace("\n\n###\n\n", "")
        )
    return list_of_preds
