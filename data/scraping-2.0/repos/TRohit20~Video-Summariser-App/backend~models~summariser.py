# Set your OpenAI API key
import openai

openai.api_key = "Use-your-own-API-Key"

# Function to generate a summary of the transcript
def generate_summary(transcript: str) -> str:
    instructPrompt = """
                    Given below is the transcript of a video. Please provide a concise yet comprehensive summary that captures the main points, key discussions, and any notable insights or takeaways.

                    How to perform this task:
                    First, break the transcript into logical sections based on topic or theme. Then, generate a concise summary for each section. Finally, combine these section summaries into an overarching summary of the entire video. The combined summary is what you should return back to me.
                    Things to focus on and include in your final summary:
                    - Ensure to extract the key insights, theories, steps, revelations, opinions, etc discussed in the video. Ensure that the summary provides a clear roadmap for listeners who want to implement the advice or insights(if any) shared.
                    - Identify any controversial or heavily debated points in the video. Summarize the various perspectives presented, ensuring a balanced representation of the video or points in the video.
                    - Along with a content summary, describe the overall mood or tone of the video. Were there moments of tension, humor, or any other notable ambiance details?
  
                    Here is the video transcript:
                    """
    
    request = instructPrompt + transcript
    resp = openai.ChatCompletion.create(
        model="gpt-4-32k",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that specialises in summarizing data."},
            {"role": "user", "content": request},
        ]
    )
    return resp['choices'][0]['message']['content']