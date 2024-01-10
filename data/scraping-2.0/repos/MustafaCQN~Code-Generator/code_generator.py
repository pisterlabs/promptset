import openai
import streamlit as st


def openai_api(api_key, input):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input + '.',
            temperature=.5,
            max_tokens=1000,
            n=1,
            stop='.'
        )
    except openai.error.APIError as e:
        st.error(f"OpenAI API returned an API Error: {e.error.message}")
        return
    except openai.error.APIConnectionError as e:
        #Handle connection error here
        st.error(f"Failed to connect to OpenAI API: {e.error.message}")
        return
    except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        st.error(f"OpenAI API request exceeded rate limit: {e.error.message}")
        return
    except:
        st.error(f"Uncaught or API key error. Please check your API key")
        return
    
    main_container.code(response['choices'][0]['text'])
    main_container.text(f"Used token: {response['usage']['total_tokens']}")

def check_button(api_key, input):
    if api_key and input:
        openai_api(api_key, input.strip().replace('/n',' '))
    else:
        st.error("Please enter your API key and comments for code creation")

main_container = st.container()
main_container.title("Code Generator")
api_key = main_container.text_input("Enter your OpenAI API key", placeholder='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
input = main_container.text_area("Enter your comments for code creation", placeholder='create me two input blocks for username and password, code them using only html and bootstrap')
button = main_container.button("Generate Code", on_click=check_button, args=(api_key, input), type='primary')

    