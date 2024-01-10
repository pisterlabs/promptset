import openai

openai.api_key = 'Your_OpenAI_API_KEY'

posts = ["python", "go", "objective c", "dart", "golang"]

for query in posts:
    query = "what is " + query + " answer in 500 word blog post formatted in html"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=query,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    )
    file = open("blog.html", "a")
    answer = (response["choices"][0]["text"])
    input = (answer)
    file.write(input)
    file.close()
    print("\n\n" + query)
    print(answer)
