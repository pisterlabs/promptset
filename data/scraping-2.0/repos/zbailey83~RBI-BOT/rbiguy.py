import dontshareconfig as d
from openai import OpenAI
import time
import re
from datetime import datetime
import requests
from io import BytesIO
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi

client = OpenAI(api_key=d.key)  # key = 'jhlkhjljkhlsf'

# what are assistants? https://platform.openai.com/docs/assistants/overview


def save_assistant_id(assistant_id, filename_base):
    max_filename_length = 255
    filename = f"{filename_base}_id.txt"
    if len(filename) > max_filename_length:
        allowed_length = max_filename_length - len("_id.txt")
        filename_base = filename_base[:allowed_length]
        filename = f"{filename_base}_id.txt"
    filepath = f"ids/{filename}"
    with open(filepath, "w") as file:
        file.write(assistant_id)


def generate_filename(strategy_description, extension):
    words = strategy_description.split()
    # If there are not enough words, fallback to using the first two words
    if len(words) >= 41:
        strategy_name = "_".join(words[39:42]).lower()  # 40th and 41st word
    else:
        strategy_name = "_".join(words[:2]).lower()
    timestamp = datetime.now().strftime("%m_%d_%y_%H%M")
    return f"{strategy_name}_{timestamp}.{extension}"


def save_output_to_file(output, idea, directory, extension):
    # Use the idea directly (trading_idea or strategy_description)
    filename = generate_filename(idea, extension)
    filepath = f"{directory}/{filename}"
    with open(filepath, "w") as file:
        file.write(output)
    print(output)
    print(f"Output saved to {filepath}")


def extract_assistant_output(messages):
    output = ""
    for message in messages:
        if message.role == "assistant" and hasattr(message.content[0], "text"):
            output += message.content[0].text.value + "\n"
    return output.strip()


def create_and_run_assistant(name, instructions, model, content, filename_base):
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=[{"type": "code_interpreter"}],
        model=model,
    )
    print(f"{name} created....")
    save_assistant_id(assistant.id, filename_base)
    thread = client.beta.threads.create()
    print(f"Thread for {name} created...{thread.id}")
    save_assistant_id(thread.id, filename_base)
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )
        if run_status.status in ["completed", "failed", "cancelled"]:
            print(f"Run completed with status: {run_status.status}")
            break
        else:
            print(f"{name} generating alpha...")
            time.sleep(5)
    print(f"Run for {name} finished, fetching messages...")
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return extract_assistant_output(messages.data)


def create_and_run_data_analysis(trading_idea):
    short_name = "strategy_creator_ai"
    content = (
        f"Create a trading strategy using {trading_idea}. The strategy should be..."
    )
    data_analysis_output = create_and_run_assistant(
        name="Strategy Creator AI",
        instructions="Create a trading strategy based on the given trading idea.",
        model="gpt-4-1106-preview",  # gpt-4-1106-preview
        content=content,
        filename_base=short_name,
    )
    if data_analysis_output:
        # Use data_analysis_output to generate the filename_base
        filename_base = generate_filename(data_analysis_output, "txt").split(".")[0]
        save_output_to_file(
            data_analysis_output,
            data_analysis_output,
            "/Users/tc/Dropbox/Trading/Strategies",
            "txt",
        )
        return data_analysis_output, filename_base
    else:
        print(f"No strategy output received for {trading_idea}.")
        return None, None


def create_and_run_backtest(strategy_output, trading_idea, filename_base):
    backtest_output = create_and_run_assistant(
        name="Backtest Coder AI",
        instructions="Code a backtest for the provided trading strategy using Python and backtrader.",
        model="gpt-4-1106-preview",
        content=f"Strategy Output: {strategy_output}. Please use backtrader for backtesting.",
        filename_base=filename_base,
    )
    if backtest_output:
        save_output_to_file(
            backtest_output,
            strategy_output,
            "/Users/tc/Dropbox/Trading/Backtests",
            "py",
        )
    else:
        print(f"No backtest output received for {trading_idea}.")


def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(["en"])
        return "".join([t["text"] for t in transcript.fetch()])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def get_pdf_text(url):
    try:
        response = requests.get(url)
        pdf = PyPDF2.PdfReader(BytesIO(response.content))
        text = ""
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text() + "\n"
        return text
    except PyPDF2.errors.PdfReadError:
        print(f"Error reading PDF from {url}")
        return None


def process_trading_ideas(ideas_list):
    for idea in ideas_list:
        print(f"Processing trading idea: {idea}")
        strategy_output, filename_base = create_and_run_data_analysis(idea)
        if strategy_output:
            create_and_run_backtest(strategy_output, idea, filename_base)


def read_trading_ideas_from_file(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def classify_and_process_idea(idea):
    youtube_pattern = (
        r"(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/(watch\?v=)?(.*)"
    )
    pdf_pattern = r"(https?:\/\/)?([\w\d]+\.)?([\w\d]+)\.(pdf)"

    youtube_match = re.match(youtube_pattern, idea)
    pdf_match = re.match(pdf_pattern, idea)

    if youtube_match:
        video_id = youtube_match.groups()[-1]
        transcript = get_youtube_transcript(video_id)
        if transcript:
            process_trading_ideas([transcript])
    elif pdf_match:
        pdf_text = get_pdf_text(idea)
        if pdf_text:
            process_trading_ideas([pdf_text])
    else:
        # It's considered a text idea if it doesn't match YouTube or PDF patterns
        process_trading_ideas([idea])


def main_idea_processor(file_path):
    with open(file_path, "r") as file:
        ideas = [line.strip() for line in file.readlines()]
    for idea in ideas:
        classify_and_process_idea(idea)


# The main entry point of the script
main_idea_processor("strat_ideas.txt")
