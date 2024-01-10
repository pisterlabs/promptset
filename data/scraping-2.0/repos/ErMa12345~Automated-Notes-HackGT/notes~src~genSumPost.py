from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
import openai

app = FastAPI()

# Set your OpenAI API key here
api_key = "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"

@app.post("/generate_summary_and_questions")
async def generate_summary_and_questions(audio_text: str = Form(...)):
    try:
        # Make a request to ChatGPT-3.5 API for summarization and practice problems generation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a student that writes detailed bulleted lecture notes."""},
                {"role": "user", "content": audio_text},
                {"role": "assistant", "content": "Generate detailed bulleted lecture notes and 5 practice problems based on the lecture."}
            ],
            api_key=api_key
        )

        # Extract summary and practice problems from API response
        content = response.choices[0].message['content']
        practice_problems_start = content.find("Practice Problems:")
        summary = content[:practice_problems_start].strip()
        practice_problems = content[practice_problems_start + len("Practice Problems:"):].strip().split("\n")

        # Format practice problems as a list of questions
        questions = [problem.replace(str(idx) + ".", "").strip() for idx, problem in enumerate(practice_problems, start=1)]

        # Prepare the response
        response_data = {
            "summary": summary,
            "questions": questions
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
