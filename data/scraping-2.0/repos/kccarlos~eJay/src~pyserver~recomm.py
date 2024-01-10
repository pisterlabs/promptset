import openai
import sys

openai.api_key = ('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')

def recommend(browsingHistory):
    # print("inside python file...Loading model...")
    # print("browsingHistory: ", browsingHistory)

    instruction_recomm = "Generate recommendations different from my browsing history: "
    instruction_sum = "Choose one category that covers most browsing history below. \
        Category options are: 'furniture', 'cookware & tableware', 'children & nursery', 'storage & organization'."

    query = ""
    for i, item in enumerate(browsingHistory.split(',')):
        query += (f"{i+1}. \"browsing history: {item}\" \n\n")

    # print("query: ", query)

    answer_prompt_recomm = "Answer in one word, I'm looking for a: "

    # answer_prompt_sum = "Which category option does most browsing histories belong to: "
    # answer_prompt_sum = "Summarize in one word, which category does most browsing history belong to? Your answer must within the category options{'furniture', 'cookware & tableware', 'children & nursery', 'storage & organization'}. "

    response_recomm = openai.Completion.create(
        model="text-davinci-002",
        prompt=instruction_recomm+"\n\n"+query+"\n\n"+answer_prompt_recomm,
        temperature=0.5,
        max_tokens=50,
        top_p=0.2,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    for choice in response_recomm['choices']:
        print(choice['text'])


    # response_sum = openai.Completion.create(
    #     model="text-davinci-002",
    #     prompt=instruction_sum+"\n\n"+query+"\n\n"+answer_prompt_sum,
    #     temperature=0.8,
    #     max_tokens=50,
    #     top_p=0.7,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )


    # for choice in response_sum['choices']:
    #     print(choice['text'])



if __name__ == "__main__":
    recommend(sys.argv[1])
