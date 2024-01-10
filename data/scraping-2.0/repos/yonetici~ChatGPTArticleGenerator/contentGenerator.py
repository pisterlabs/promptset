import openai
import streamlit as st

openai.api_key = "YOUR-API-KEY"
st.title("SEO Article Generator with ChatGPT")
def generateArticle(keyword, writingStyle, wordCount):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
          {
                "role":"user",
                "content":"I am owner of tourism information web site. I want you to help me create quality content for my visitors."
            },
            {
                "role":"user",
                #"content":"Please act as an experienced SEO specialist and Content Creator and Write a SEO optimized article about " + keyword + " by using Google SEO standarts, Semantic SEO rules, LSI Paradigm, Long tail keywords."
                "content":"Please act as an experienced SEO specialist and Content Creator. You have to follow Google SEO standarts, Semantic SEO rules, LSI Paradigm concepts rules. Write a SEO optimized article about " + keyword + "."
            },
            {
                "role":"user",
                "content":"This article should be in style " + writingStyle
            },
                        {
                "role":"user",
                "content":"This article should have subheadings."
            },

            {
                "role":"user",
                "content":"The article must have more than " + str(wordCount) + " words."
            }
        ]
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    print(result)
    return result
    #return "Generated Article with out API"

keyword = st.text_input("Enter Keyword")
writingStyle = st.selectbox("Select Writing Style", ["Funny", "Academic", "Sarcastic", "Informative", "Witty"])
wordCount = st.slider("Select word count", min_value=300, max_value=2000, value=300)
submitButton = st.button("Generate Article")

if submitButton:
    message = st.empty()
    message.text("Generating Article")
    article = generateArticle(keyword, writingStyle, wordCount)
    message.text("")
    st.write(article)
    st.download_button(label="Download Article", data=article, file_name="article.txt", mime="text/txt")

