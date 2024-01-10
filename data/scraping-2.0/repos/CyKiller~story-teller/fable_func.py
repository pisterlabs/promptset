import os
import random
import requests
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENGINE_ID = "text-davinci-003"

headers_openai = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

def check_argument(arg, expected_type):
    if not isinstance(arg, expected_type):
        raise TypeError(f"Expected argument of type {expected_type}, but got {type(arg)}")

def ask_genre():
    check_argument(genre, str)
    return genre

def get_num_pages_per_chapter(genre, length):
    check_argument(genre, str)
    check_argument(length, int)
    if genre == "fantasy":
        # specifies the number of pages in each chapter 10-20
        num_pages = random.randint(13, 20)
    elif genre == "romance":
        num_pages = random.randint(7, 15)
    elif genre == "mystery":
        num_pages = random.randint(13, 25)
    elif genre == "sci-fi":
        num_pages = random.randint(13, 30)
    elif genre == "horror":
        num_pages = random.randint(7, 15)
    elif genre == "thriller":
        num_pages = random.randint(12, 25)
    elif genre == "historical":
        num_pages = random.randint(11, 30)
    elif genre == "action":
        num_pages = random.randint(13, 25)
    elif genre == "adventure":
        num_pages = random.randint(13, 25)
    elif genre == "comedy":
        num_pages = random.randint(6, 15)
    elif genre == "drama":
        num_pages = random.randint(5, 15)
    elif genre == "poetry":
        num_pages = random.randint(5, 15)
    elif genre == "satire":
        num_pages = random.randint(6, 15)

    # if user specifies a length, use that instead

    if length != None:
        num_pages = length

    return num_pages

def generate_plots(prompt, genre):
    check_argument(prompt, str)
    check_argument(genre, str)
    data = {
        "engine": OPENAI_ENGINE_ID,
        "prompt": prompt,
        "max_tokens": 100,
    }
    response = requests.post("https://api.openai.com/v1/engines/davinci-codex/completions", headers=headers_openai, json=data)
    if response.status_code != 200:
        raise Exception(f"Request to OpenAI API failed with status code {response.status_code}")
    return response.json()['choices'][0]['text']

def select_most_engaging(plots, genre):
    check_argument(plots, list)
    check_argument(genre, str)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"You are an expert in writing fantastic {genre} novel plots."},
            {"role": "user", "content": f"Here are a number of possible plots for a new novel: {plots}\n\n--\n\nNow, write the final plot that we will go with. It can be one of these, a mix of the best elements of multiple, or something completely new and better. The most important thing is the plot should be fantastic, unique, and engaging."}
        ]
    )

    print_step_costs(response, "gpt-3.5-turbo")

    return response['choices'][0]['message']['content']
def improve_plot(plot):
    check_argument(plot, str)
response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in improving and refining story plots."},
            {"role": "user", "content": f"Improve this plot: {plot}"}
        ]                                                                                                 
    )

    print_step_costs(response, "gpt-4")
    return response['choices'][0]['message']['content']


def get_title(plot):
    check_argument(plot, str)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are an expert writer."},
            {"role": "user", "content": f"Here is the plot: {plot}\n\nWhat is the title of this book? Just respond with the title, do nothing else."}
        ]
    )

    print_step_costs(response, "gpt-3.5-turbo-16k")
    return response['choices'][0]['message']['content']


def write_first_chapter(plot, genre, first_chapter_title, writing_style):
    check_argument(plot, str)
    check_argument(genre, str)
    check_argument(first_chapter_title, str)
    check_argument(writing_style, str)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a world-class {genre} writer."},
            {"role": "user", "content": f"Here is the high-level plot to follow: {plot}\n\nWrite the first chapter of this novel: `{first_chapter_title}`.\n\nMake it incredibly unique, engaging, and well-written.\n\nHere is a description of the writing style you should use: `{writing_style}`\n\nInclude only the chapter text. There is no need to rewrite the chapter name."}
        ]
    )

    print_step_costs(response, "gpt-3.5-turbo-16k")

    improved_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a world-class {genre} writer. Your job is to take your student's rough initial draft of the first chapter of their {genre} novel, and rewrite it to be significantly better, with much more detail."},
            {"role": "user",
                "content": f"Here is the high-level plot you asked your student to follow: {plot}\n\nHere is the first chapter they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the first chapter of this novel, in a way that is far superior to your student's chapter. It should still follow the exact same plot, but it should be far more detailed, much longer, and more engaging. Here is a description of the writing style you should use: `{writing_style}`"}
        ]
    )

    print_step_costs(response, "gpt-3.5-turbo-16k")
    return improved_response['choices'][0]['message']['content']

def write_chapter(previous_chapters, plot, genre, chapter_title):
    check_argument(previous_chapters, list)
    check_argument(plot, str)
    check_argument(genre, str)
    check_argument(chapter_title, str)
    try:
        i = random.randint(1, 2242)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": f"You are a world-class {genre} writer."},
                {"role": "user", "content": f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name."}
            ]
        )

        print_step_costs(response, "gpt-3.5-turbo-16k")

        return response['choices'][0]['message']['content']
    except:
        q   response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": f"You are a world-class {genre} writer."},
                {"role": "user", "content": f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name."}
            ]
        )

        print_step_costs(response, "gpt-3.5-turbo-16k")

        return response['choices'][0]['message']['content']
    

def generate_storyline(prompt, num_chapters, genre):
    check_argument(prompt, str)
    check_argument(num_chapters, int)
    check_argument(genre, str)
    print("Generating storyline with chapters and high-level details...")
    json_format = """[{"Chapter CHAPTER_NUMBER_HERE - CHAPTER_TITLE_GOES_HERE": "CHAPTER_OVERVIEW_AND_DETAILS_GOES_HERE"}, ...]"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"You are a world-class {genre} writer. Your job is to write a detailed storyline, complete with chapters, for a {genre} novel. Don't be flowery -- you want to get the message across in as few words as possible. But those words should contain lots of information."},
            {"role": "user", "content": f'Write a fantastic storyline with {num_chapters} chapters and high-level details based on this plot: {prompt}.\n\nDo it in this list of dictionaries format {json_format}'}
        ]
    )

    print_step_costs(response, "gpt-4")

    improved_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a world-class {genre} writer. Your job is to take your student's rough initial draft of the storyline of a {genre} novel, and rewrite it to be significantly better."},
            {"role": "user",
                "content": f"Here is the draft storyline they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the storyline, in a way that is far superior to your student's version. It should have the same number of chapters, but it should be much improved in as many ways as possible. Remember to do it in this list of dictionaries format {json_format}"}
        ]
    )

    print_step_costs(improved_response, "gpt-4")
    storyline = improved_response['choices'][0]['message']['content']

    # Ensure the storyline is a properly formatted list of dictionaries
    try:
        storyline = ast.literal_eval(storyline)
    except SyntaxError:
        print("The storyline is not a properly formatted list of dictionaries.")
        storyline = []

    return storyline


def write_vectorless_novel(prompt, num_chapters, writing_style, genre, length):
    check_argument(prompt, str)
    check_argument(num_chapters, int)
    check_argument(writing_style, str)
    check_argument(genre, str)
    check_argument(length, int)
    plots = generate_plots(prompt, genre)

    best_plot = select_most_engaging(plots, genre)

    improved_plot = improve_plot(best_plot)

    title = get_title(improved_plot)

    storyline = generate_storyline(improved_plot, num_chapters, genre)
    chapter_titles = storyline
    novel = f"Storyline:\n{storyline}\n\n"

    first_chapter = write_first_chapter(
        genre, storyline, chapter_titles[0], writing_style.strip())
    novel += f"Chapter 1:\n{first_chapter}\n"
    chapters = [first_chapter]

    for i in range(num_chapters - 1):
        # + 2 because the first chapter was already added
        print(f"Writing chapter {i+2}...")
        chapter = write_chapter(novel, storyline, genre, chapter_titles[i+1])
        novel += f"Chapter {i+2}:\n{chapter}\n"
        chapters.append(chapter)

    num_pages_per_chapter = get_num_pages_per_chapter(genre, length)

    return novel, title, chapters, chapter_titles