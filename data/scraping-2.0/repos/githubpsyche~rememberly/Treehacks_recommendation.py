from googlesearch import search
import openai

def googleSearchURL(list_words):
    result_URL = []
    for i in list_words:
        query = i.split()
        searchquery = query[0]
        for j in query[1:]:
            searchquery = searchquery + " AND " + j
        for result in search(searchquery,num_results=10):
            result_URL.append(result)
    return result_URL[1:]


def textsummary(input_string_list):
    output_list_summary = []
    for i in input_string_list:
        input_prompt = "Explain the content below content in simpler term:\n\n" + i
        openai.api_key = "sk-UQuGm7CmM87xbJWSKAsZ4QqYkeBTz63JOcR00ZJk"
        response = openai.Completion.create(
          engine="davinci-instruct-beta",
          prompt=input_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
        output_list_summary.append(response.choices[0]["text"].strip().replace("\n",""))
    return output_list_summary
#
# list1 = ["Coginitive Science Stanford","The Last Supper"]
# test_google_links = googleSearchURL(list1)
# test_OpenAI = textsummary(list1)
#
# print(test_google_links)
# print(test_OpenAI)
