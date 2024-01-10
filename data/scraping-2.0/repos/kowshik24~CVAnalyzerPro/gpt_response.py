import openai
import sqlite3

# Assuming you have the same database structure and connection
def get_openai_key(username):
    conn = sqlite3.connect('data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('SELECT openai_key FROM userstable WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_gpt_response(username, resume_text, requirements):
    # Fetch OpenAI key from database
    openai_api_key = get_openai_key(username)
    if not openai_api_key:
        return False

    # Initialize OpenAI key
    openai.api_key = openai_api_key

    try:
      # Make GPT request
      response = openai.Completion.create(
          engine="text-davinci-003",
          prompt="Act like you are an expert Resume Scorer. You will always give the similarities and disimilarities with the Requirements and Resume. Then give the score on range of 100 percentage. Now, Score this resume based on the following requirements:\n\n" + "Requirements: " + requirements + "\n\n" + "Resume: " + resume_text,
          max_tokens=512
      )
      return response.choices[0].text.strip()
    except Exception as e:
      print(e)
      return False
