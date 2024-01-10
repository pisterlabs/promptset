import openai

openai.api_key = '<your key here>'

def getQuestionName(question: str) -> str:

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {
                "role": "system",
                "content": "you will have to output a question name for the given question, for example for the prompt 'What is the correct syntax to create a column vector a (select correct options)' the question name should be 'ColumnVectorCreation' and for 'Given y = 1:0.5:3, how many elements will y have?' the question name should be 'NumberOfElemetsWithColonOperator'. the output should be printed as follows: '<question name>'"
            },
            {
                "role": "user",
                "content": question
            },
        ],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print((response['choices'][0]['message']['content']))
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print(getQuestionName("To plot a graph with two different y-axes, use the function"))
