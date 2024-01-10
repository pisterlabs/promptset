# chatgpt_integration.py
import openai

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
SUMMARY_LENGTH = 100  # Length of the summary in tokens

def initialize_openai_api():
    """
    Function to initialize the OpenAI API with the provided API key.
    """
    openai.api_key = OPENAI_API_KEY

def summarize_with_chatgpt(input_text):
    """
    Function to summarize and restructure the information using ChatGPT.
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=SUMMARY_LENGTH,
        temperature=0.7,
        stop=["\n"],
    )
    return response.choices[0].text.strip()

def save_summary_to_file(summary):
    """
    Function to save the summarized content to "summary_info_to_agents.txt" file.
    """
    with open("data/summary_info_to_agents.txt", "w", encoding="utf-8") as file:
        file.write(summary)

    print("Summary saved to 'summary_info_to_agents.txt'.")

if __name__ == "__main__":
    try:
        print("ChatGPT Integration: Initializing OpenAI API...")
        initialize_openai_api()

        print("ChatGPT Integration: Reading information for summarization...")
        with open("data/query_search_result.txt", "r", encoding="utf-8") as file:
            input_text = file.read()

        print("ChatGPT Integration: Summarizing information using ChatGPT...")
        summarized_info = summarize_with_chatgpt(input_text)

        print("ChatGPT Integration: Saving the summary...")
        save_summary_to_file(summarized_info)

        print("ChatGPT Integration: Summarization completed.")
    except Exception as e:
        print("ChatGPT Integration: An error occurred during summarization:", e)