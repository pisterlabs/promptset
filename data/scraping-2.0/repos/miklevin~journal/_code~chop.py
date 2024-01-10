# Chops and publishes single file into multiple for Jekyll publishing.

import os
import re
import sys
import yaml
import nltk
import shlex
import pickle
import openai
import shutil
import argparse
import tiktoken
from time import sleep
from pathlib import Path
from slugify import slugify
from pyfiglet import Figlet
from datetime import datetime
from subprocess import Popen, PIPE
from nltk.stem import WordNetLemmatizer
from sqlitedict import SqliteDict as sqldict
from collections import Counter, defaultdict


# Set where the OpenAI Key is kept
OPENAI_KEY_PATH = "/home/ubuntu/.ssh/openai.txt"

nltk.download('wordnet', quiet=True)  # Update Lemmatizer

# Define what fields OpenAI will (optionally) generate.
ALL_FIELDS = ["date", "title", "headline", "description", "keyword", "categories"]
AI_FIELDS = ["headline", "description", "keywords"]
ENGINE = "text-davinci-003"
NUMBER_OF_CATEGORIES = 24  # Best as a multiple of NUMBER_OF_COLUMNS
NUMBER_OF_COLUMNS = 4
PAST_IS_LEFT = True
TEMPERATURE = 0.5
MAX_TOKENS = 100


# Load function early so we can use it, pronto!
def fig(text, description=None, font="standard"):
    #  _____ _       _      _
    # |  ___(_) __ _| | ___| |_
    # | |_  | |/ _` | |/ _ \ __|
    # |  _| | | (_| | |  __/ |_
    # |_|   |_|\__, |_|\___|\__|
    #          |___/
    """Let them see text!"""
    f = Figlet(font=font)
    print(f.renderText(text))
    if description:
        print(description)
    sleep(0.5)

fig("Chopping...")

#  ____                          _                             
# |  _ \ __ _ _ __ ___  ___     / \   _ __ __ _ ___            
# | |_) / _` | '__/ __|/ _ \   / _ \ | '__/ _` / __|           
# |  __/ (_| | |  \__ \  __/  / ___ \| | | (_| \__ \           
# |_|   \__,_|_|  |___/\___| /_/   \_\_|  \__, |___/           
# Define command line arguments.          |___/                
#                                                              
# Use in your .vimrc or init.vim like this:                    
# let @p = ":terminal python ~/repos/Journal/_code/chop.py -f " . expand('%:p')

aparser = argparse.ArgumentParser()
add_arg = aparser.add_argument

# Only 1 required argument: the full path to the YAMLesque file.
add_arg("-f", "--full_path", required=True)
add_arg("-a", "--author", default="site.author")
add_arg("-b", "--blog", default="blog")
add_arg("-o", "--output", default="_posts")

# Parse command line args as CONSTANTS
SEPARATOR = (80 * "-") + "\n"
CHOP = re.compile(r"-{78,82}\s*\n")
ARGS = aparser.parse_args()
BLOG = f"/{ARGS.blog}/" if not ARGS.blog.startswith("/") else ARGS.blog
OUTPUT = ARGS.output
AUTHOR = ARGS.author
YAMLESQUE = ARGS.full_path
parts = YAMLESQUE.split("/")

# Your source YAMLesque file can be in 1 of 2 places:
if parts[-2] == "_drafts":
    # The long-version remains in draft and is never published directly.
    REPO = parts[-3] + "/"
    PATH = "/".join(parts[:-3]) + "/"
else:
    # The long-version is in Jekyll publishing-space and auto-publishes.
    REPO = parts[-2] + "/"
    PATH = "/".join(parts[:-2]) + "/"

# Set up the rest of the constants
FILE = parts[-1]
INCLUDES = f"{PATH}{REPO}_includes/"
REPO_DATA = f"{PATH}{REPO}_data/"
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"
TEMP_OUTPUT = f"{REPO_DATA}{FILE}"
KEYWORDS_FILE = "{PATH}{REPO}_data/keywords.txt"
CATEGORY_PAGE = f"{PATH}{REPO}category.md"
CATEGORY_GRID = f"{INCLUDES}category_list.md"
CATEGORY_INCLUDE = f"{INCLUDES}category.md"
# LOAD SITE CONFIG

# Check if the repo has a _config.yml file
# If not, print a message and quit.
if not os.path.isfile(f"{PATH}{REPO}_config.yml"):
    fig(REPO, f"Missing {PATH}{REPO}_config.yml")
    print("Press :bd [Enter] to close this buffer.")
    raise SystemExit()


with open(f"{PATH}{REPO}_config.yml", "r") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise SystemExit()

# Set database constant names
SUMMARIESDB = REPO_DATA + "summaries.db"
for afield in AI_FIELDS:
    db_var = f"{afield.upper()}DB"
    db_file = f"{REPO_DATA}{afield}.db"
    command = f'{db_var} = "{db_file}"'
    exec(command)

print(f"CHOPPING: {PATH}{REPO}{FILE}")

# global variables (yes, this is fine in Python)
ydict = defaultdict(dict)  # A dict of all journal entry front matter
cdict = defaultdict(dict)  # A dict of category Capitalization & counts

# Create _posts, _data, etc. if they don't exist
for make_loc in [OUTPUT_PATH, REPO_DATA]:
    Path(make_loc).mkdir(parents=True, exist_ok=True)

# Assure consistent keyword variation usage
LEMMATIZER = WordNetLemmatizer()

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___    Above this is configuration
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|   And setting CONSTANTS.
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \   Below functions is a Playground.
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/   And finally, Flow Control.


def yaml_generator(full_path, reverse=False, drafts=False, clone=False):
    # __   __ _    __  __ _        ____                           _
    # \ \ / // \  |  \/  | |      / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
    #  \ V // _ \ | |\/| | |     | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
    #   | |/ ___ \| |  | | |___  | |_| |  __/ | | |  __/ | | (_| | || (_) | |
    #   |_/_/   \_\_|  |_|_____|  \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|
    """Yields a stream of 3-tuples (YAML, post, original) from YAMLesque file.
    If there's no YAML, the yielded tuple will be (None, original, original)"""
    with open(full_path, "r") as fh:
        posts = CHOP.split(fh.read())
        if reverse:
            posts.reverse()  # Reverse so article indexes don't change.
        for i, post in enumerate(posts):
            rv = None, post, post  # Always rebuild source at very least.
            py, yaml_str, body = None, "", ""
            if "---" in post:
                yaml_str, body = post.split("---", 1)
                try:
                    py = yaml.load(yaml_str, Loader=yaml.FullLoader)
                    rv = py, body, post
                except yaml.YAMLError:
                    # Deliberately passing silently to prevent attempts
                    # to create pages where there is no page to create.
                    ...
            if clone:
                # If we're cloning, we want to yield everything.
                yield rv
            elif py and "published" in py and py["published"] == False and drafts:
                # It's a draft and we're rendering drafts.
                yield rv
            elif py and "published" in py and py["published"] == False:
                # It's a draft and we're not rendering drafts.
                continue
            elif not drafts:
                # The general default.
                yield rv


def odb(DBNAME, slug, name, data):
    #   ___                      _    ___   _   _ _ _
    #  / _ \ _ __   ___ _ __    / \  |_ _| | | | (_) |_
    # | | | | '_ \ / _ \ '_ \  / _ \  | |  | |_| | | __|
    # | |_| | |_) |  __/ | | |/ ___ \ | |  |  _  | | |_
    #  \___/| .__/ \___|_| |_/_/   \_\___| |_| |_|_|\__|
    #       |_|
    """Retrieves and saves OpenAI request not already done.
    It checks if the data is there first, so safe to re-run."""
    api_hit = False

    if name in ['description', 'keywords']:
        chop_at = 3200
    else:
        chop_at = 3600
    encoding = "cl100k_base"
    with sqldict(DBNAME) as db:
        if slug in db:
            result = db[slug]
        else:
            # print(f"Hitting OpenAI for {DBNAME}")
            # Chop the article down to a summarize able length
            prompt_length = num_tokens_from_string(data, encoding)
            if prompt_length > chop_at:
                while prompt_length > chop_at:
                    data = data.rsplit(" ", 1)[0]
                    prompt_length = num_tokens_from_string(data, encoding)
                    # print(".", end="")
                print(f"Prompt is now {prompt_length} tokens.")

            aprompt = make_prompt(name, data)

            # Build a prompt and get a result from OpenAI.
            result = openai.Completion.create(
                engine=ENGINE,
                prompt=aprompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
                stop=None,
            )
            result = result.choices[0].text.strip()
            db[slug] = result
            db.commit()
            api_hit = True
    return result, api_hit


def make_prompt(dict_key, data):
    #  ____                            _
    # |  _ \ _ __ ___  _ __ ___  _ __ | |_ ___
    # | |_) | '__/ _ \| '_ ` _ \| '_ \| __/ __|
    # |  __/| | | (_) | | | | | | |_) | |_\__ \
    # |_|   |_|  \___/|_| |_| |_| .__/ \__|___/
    #                           |_|
    """Returns an OpenAI prompt for the given dict key-name and post data."""
    dict_o_prompts = {
        "headline": (
            f"Write a short headline for the following post:\n{data}\n\n"
            "You are the one who write this. Write from first person perspective. Never say 'The author'. '"
            "Do not reuse the title in the headline. Write something new. Use only one sentence. "
            "\nHeadline:\n\n"
        ),
        "description": (
            f"Write a concise and informative meta description for the following text:\n{data}\n\n"
            "...that will work well as summary-text in website navigation. "
            "Write from the first person perspective. "
            "Keep it short."
            "\nSummary:\n\n"
        ),
        "keywords": (
            f"Create a line of comma separated list of keywords to categorize the following text:\n\n{data}\n\n"
            "Do not use extremely broad words like Data, Technology, Blog, Post or Author. "
            "Use words that will be good for site categories, tags and search. "
            "Do not use quotes around keywords. "
            "\nKeywords:\n\n"
        ),
        "advice": (
            f"You are my work advisor and life-coach. "
            "Read what I have written and tell me what I should do next:\n{data}\n\n"
            "I am trying to achieve my ikigai, but don't mention ikigai in the response. "
            "Be specific in your advice and not 'decide goals', 'write plan' and 'celebrate successes'. "
            "Impress me with your insight."
            "\nAdvice:\n\n"
        ),
        "question": (
            f"You are someone just discovering my website. "
            "Read this post and tell me what question you have:\n{data}\n\n"
            "I will try to answer it on a follow-up post."
            "\nAdvice:\n\n"
        ),
    }
    return dict_o_prompts[dict_key]


#  _____ _                 _ _
# |  ___| | _____      __ (_) |_ ___ _ __ ___  ___
# | |_  | |/ _ \ \ /\ / / | | __/ _ \ '_ ` _ \/ __|
# |  _| | | (_) \ V  V /  | | ||  __/ | | | | \__ \
# |_|   |_|\___/ \_/\_/   |_|\__\___|_| |_| |_|___/


def delete_old():
    #  ____       _      _                    _
    # |  _ \  ___| | ___| |_ ___   _ __  _ __(_) ___  _ __
    # | | | |/ _ \ |/ _ \ __/ _ \ | '_ \| '__| |/ _ \| '__|
    # | |_| |  __/ |  __/ ||  __/ | |_) | |  | | (_) | |
    # |____/ \___|_|\___|\__\___| | .__/|_|  |_|\___/|_|
    #                             |_|
    for fh in os.listdir(OUTPUT_PATH):
        delete_me = f"{OUTPUT_PATH}/{fh}"
        os.remove(delete_me)
    for fh in os.listdir(f"{PATH}{REPO}"):
        if fh.startswith("cat_"):
            delete_me = f"{PATH}{REPO}/{fh}"
            os.remove(delete_me)
    for fh in os.listdir(f"{INCLUDES}"):
        if fh.startswith("cat_"):
            delete_me = f"{INCLUDES}/{fh}"
            os.remove(delete_me)
    of = Path(TEMP_OUTPUT)
    if of.exists():
        of.unlink()


def sync_check():
    #  ______   ___   _  ____    ____ _               _
    # / ___\ \ / / \ | |/ ___|  / ___| |__   ___  ___| | __
    # \___ \\ V /|  \| | |     | |   | '_ \ / _ \/ __| |/ /
    #  ___) || | | |\  | |___  | |___| | | |  __/ (__|   <
    # |____/ |_| |_| \_|\____|  \____|_| |_|\___|\___|_|\_\
    """Check for new posts needing AI-writing or YAMLESQUE source-file updating."""
    global ydict

    # Check if openai_key_path file exits:
    if Path(OPENAI_KEY_PATH).exists():
        with open(OPENAI_KEY_PATH, "r") as fh:
            openai.api_key = fh.readline().strip()  # Read in your OpenAI API key
        for i, (fm, apost, combined) in enumerate(yaml_generator(YAMLESQUE, clone=True)):
            if fm and len(fm) == 2 and "title" in fm and "date" in fm:
                # Only 2 fields of YAML front matter asks for release.
                title = fm["title"]
                slug = slugify(title)
                ydict[slug]["title"] = title
                # Setting these values ALSO commits it to the databases
                hits = []
                rdict = {}
                print("- Hitting OpenAI for Subhead, Description and Keywords.")
                for afield in AI_FIELDS:
                    dbname = f"{REPO_DATA}{afield}.db"
                    result, api_hit = odb(dbname, slug, afield, combined)
                    rdict[afield] = result
                    if api_hit:
                        hits.append(api_hit)
                print()
                if any(hits):
                    for afield in AI_FIELDS:
                        print(f"{afield}: {rdict[afield]}")
                        print()
                build_ydict()  # Rebuilds ydict from database
                update_yaml()  # Updates YAMLESQUE file data from database
                new_source()  # Replaces YAMLESQUE input with synchronized output
                if any(hits):
                    raise SystemExit("Review changes in source and re-release.\nClose this NeoVim buffer with :bd")


def build_ydict(yamlesque=YAMLESQUE):
    #  ____        _ _     _                         _       _ _      _
    # | __ ) _   _(_) | __| |  _   _  __ _ _ __ ___ | |   __| (_) ___| |_
    # |  _ \| | | | | |/ _` | | | | |/ _` | '_ ` _ \| |  / _` | |/ __| __|
    # | |_) | |_| | | | (_| | | |_| | (_| | | | | | | | | (_| | | (__| |_
    # |____/ \__,_|_|_|\__,_|  \__, |\__,_|_| |_| |_|_|  \__,_|_|\___|\__|
    #                          |___/
    """Rebuilds ydict from _data/*.dbs, which may have more daata than the YAMLESQUE source."""
    global ydict
    ydict = defaultdict(dict)
    for i, (fm, _, _) in enumerate(yaml_generator(YAMLESQUE)):
        if fm and isinstance(fm, dict):
            if "title" in fm:
                slug = slugify(fm["title"])
                fm["slug"] = slug
                ydict[slug] = fm
    return len(ydict)


def update_yaml():
    #  _   _           _       _        __   __ _    __  __ _
    # | | | |_ __   __| | __ _| |_ ___  \ \ / // \  |  \/  | |
    # | | | | '_ \ / _` |/ _` | __/ _ \  \ V // _ \ | |\/| | |
    # | |_| | |_) | (_| | (_| | ||  __/   | |/ ___ \| |  | | |___
    #  \___/| .__/ \__,_|\__,_|\__\___|   |_/_/   \_\_|  |_|_____|
    #       |_|
    """Updates the YAMLESQUE file data from the database"""
    print("Updating source file with OpenAI responses.")
    with open(TEMP_OUTPUT, "w", encoding="utf-8") as fh:
        for i, (fm, body, post) in enumerate(yaml_generator(YAMLESQUE, clone=True)):
            if i:
                fh.write(SEPARATOR)
            if fm and len(fm) == 2:
                try:
                    title = fm["title"]
                except:
                    print("No title found in YAML file.")
                    raise SystemExit()
                slug = slugify(title)
                ai_fields = []
                for afield in AI_FIELDS:
                    db_var = f"{afield.upper()}DB"
                    command = f'fm["{afield}"] = oget({db_var}, slug)'
                    exec(command)
                    ai_fields.append(afield)
                fields = ["date", "title"] + ai_fields
                ymlstr = ""
                for field in fields:
                    try:
                        ymlstr += f"{field}: {sq(fm[field])}\n"
                    except:
                        # Advice field won't be there for most.
                        ...
                fh.write(ymlstr)
                fh.write("---")
                fh.write(body)
            else:
                fh.write(post)
    build_ydict()
    print("updated!")


def new_source():
    #  _   _                 ____
    # | \ | | _____      __ / ___|  ___  _   _ _ __ ___ ___
    # |  \| |/ _ \ \ /\ / / \___ \ / _ \| | | | '__/ __/ _ \
    # | |\  |  __/\ V  V /   ___) | (_) | |_| | | | (_|  __/
    # |_| \_|\___| \_/\_/   |____/ \___/ \__,_|_|  \___\___|
    """If there's a new source, copy it to the input file. It's meta."""
    files_are_same = compare_files(YAMLESQUE, TEMP_OUTPUT)
    if files_are_same:
        print("Nothing's changed. Nothing to publish.")
    else:
        print("Something's changed. Copied output to input.")
        # Put a copy of the current YAMLESQUE file into data folder:
        shutil.copyfile(YAMLESQUE, f"{REPO_DATA}journal-prior.md")
        # Replaces old journal.md with the new journal.md (AI content filled-in)
        shutil.copyfile(TEMP_OUTPUT, YAMLESQUE)


def make_index():
    #  __  __       _          ___           _
    # |  \/  | __ _| | _____  |_ _|_ __   __| | _____  __
    # | |\/| |/ _` | |/ / _ \  | || '_ \ / _` |/ _ \ \/ /
    # | |  | | (_| |   <  __/  | || | | | (_| |  __/>  <
    # |_|  |_|\__,_|_|\_\___| |___|_| |_|\__,_|\___/_/\_\
    #
    """Builds the index pages"""
    num_posts = build_ydict()  # Rebuilds ydict from database
    with open(f"{INCLUDES}post_list.html", "w", encoding="utf-8") as fh:
        num_posts = len(ydict) + 1
        fh.write(f'<ol start="{num_posts}" reversed >\n')
        for i, (fm, apost, combined) in enumerate(yaml_generator(YAMLESQUE)):
            if fm and "title" in fm and "date" in fm and "description" in fm:
                title = fm["title"]
                slug = slugify(title)
                description = fm["description"]
                # Neutralize pointy brackets for description:
                description = description.replace("<", "&lt;")
                description = description.replace(">", "&gt;")
                adate = fm["date"]
                fh.write(f'<li><a href="{BLOG}{slug}/">{title}</a> ({adate})\n<br />{description}</li>\n')
        fh.write("</ol>\n")
    with open(f"{INCLUDES}post_short_list.html", "w", encoding="utf-8") as fh:
        num_posts = len(ydict) + 2
        fh.write(f"<ol>\n")
        to_ten = 0
        for i, (fm, apost, combined) in enumerate(yaml_generator(YAMLESQUE)):
            if fm and "title" in fm and "date" in fm and "description" in fm:
                title = fm["title"]
                slug = slugify(title)
                description = fm["description"]
                # Neutralize pointy brackets for description:
                description = description.replace("<", "&lt;")
                description = description.replace(">", "&gt;")
                adate = fm["date"]
                fh.write(f'<li><a href="{BLOG}{slug}/">{title}</a> ({adate})\n<br />{description}</li>\n')
                to_ten += 1
                if to_ten >= 10:
                    break
        fh.write("</ol>\n")


def category_scan():
    #   ____      _                                ____                  
    #  / ___|__ _| |_ ___  __ _  ___  _ __ _   _  / ___|  ___ __ _ _ __  
    # | |   / _` | __/ _ \/ _` |/ _ \| '__| | | | \___ \ / __/ _` | '_ \ 
    # | |__| (_| | ||  __/ (_| | (_) | |  | |_| |  ___) | (_| (_| | | | |
    #  \____\__,_|\__\___|\__, |\___/|_|   \__, | |____/ \___\__,_|_| |_|
    #                     |___/            |___/                         
    """Find Categories"""
    global cdict
    if "categories" in CONFIG and "filter" in CONFIG["categories"]:
        category_filter = CONFIG["categories"]["filter"]
    else:
        category_filter = None
    cat_dict = defaultdict(list)
    word_list = defaultdict(list)
    proper_words = defaultdict(lambda x=None: x)
    with open(YAMLESQUE) as fh:
        for post in CHOP.split(fh.read()):
            ystr, body = post.split("---", 1)
            if ystr:
                try:
                    yml = yaml.load(ystr, Loader=yaml.FullLoader)
                except yaml.scanner.ScannerError as e:
                    diagnose_yaml(ystr, e)
                if "title" in yml:
                    slug = slugify(yml["title"])
                if "keywords" in yml:
                    keywords = yml["keywords"].split(", ")
                    for keyword in keywords:
                        normalized_keyword = normalize_key(keyword)
                        word_list[normalized_keyword].append(keyword)
                        cat_dict[normalized_keyword].append(slug)
    # It's good to have an external data file of categories.
    cat_file = f"{REPO_DATA}categories.yml"
    with open(cat_file, "w") as fh:
        yaml.dump(list(word_list.keys()), fh)
    # Now we have a list of categories, but we need to know which
    # category is the most popular for each keyword.
    for key in word_list:
        alist = word_list[key]
        proper_words[key] = Counter(alist).most_common(1)[0][0]

    # Now we have a dictionary of categories and slugs.
    for key in cat_dict:
        cat_dict[key].reverse()  # Reverse the list so it's in chronological order.

    cat_counter = Counter()  # Create a counter object
    for cat, slugs in cat_dict.items():
        cat_counter[cat] = len(slugs)
    # common_cats = cat_counter.most_common()

    # Create the global category dictionary (cdict):
    for cat, count in cat_counter.most_common():
        # if category_filter and cat not in category_filter:
        #     continue
        # cdict[cat]["slug"] = slugify(cat)
        cdict[cat]["count"] = count
        cdict[cat]["title"] = proper_words[cat]

    # It's also good to have an external file of category frequencies.
    cat_file = f"{REPO_DATA}category_frequency.yml"
    with open(cat_file, "w") as fh:
        yaml.dump(cdict, fh)
    # print(f"Found {len(cdict):,} categories.")
    category_grid()  # Builds category_list.md include
    category_page()  # Builds category.md and include
    category_pages()  # Builds cat_*.md and cat_*.md includes


def category_grid():
    #   ____      _      ____      _     _
    #  / ___|__ _| |_   / ___|_ __(_) __| |
    # | |   / _` | __| | |  _| '__| |/ _` |
    # | |__| (_| | |_  | |_| | |  | | (_| |
    #  \____\__,_|\__|  \____|_|  |_|\__,_|
    #
    """Build the 100-cell grid of categories."""
    global cdict
    rows = NUMBER_OF_CATEGORIES // NUMBER_OF_COLUMNS
    counter = 0
    top_cats = get_top_cats()
    with open(CATEGORY_GRID, "w") as fh:
        # print(f"Len of top_cats: {len(top_cats)}")
        # print(f"Number of cells: {rows * NUMBER_OF_COLUMNS}")
        # raise SystemExit()
        if cdict and len(top_cats) >= (rows * NUMBER_OF_COLUMNS):
            for row in range(rows):
                fh.write("\n")
                for col in range(NUMBER_OF_COLUMNS):
                    cat = top_cats[counter]
                    title = cdict[cat]["title"]

                    slug = slugify(cat)
                    markdown_link = f"[{title}](/{slug}/)"
                    fh.write(f"{markdown_link} | ")
                    counter += 1
        else:
            # Make the include empty.
            print("Skipping category grid. Check _config.yml")
            fh.write("")


def category_page():
    #   ____      _     ____
    #  / ___|__ _| |_  |  _ \ __ _  __ _  ___
    # | |   / _` | __| | |_) / _` |/ _` |/ _ \
    # | |__| (_| | |_  |  __/ (_| | (_| |  __/
    #  \____\__,_|\__| |_|   \__,_|\__, |\___|
    #                              |___/
    """Build the category page (singular)"""
    global cdict
    if cdict:
        with open(CATEGORY_PAGE, "w") as fh:
            with open(CATEGORY_INCLUDE, "w") as fh2:
                fh.write("# Categories [(All)](/categories/)\n")  # This could be more frontmatter-y
                fh.write("{% include category.md %}\n")  # Reference to include
                fh2.write(f"<ol start='{NUMBER_OF_CATEGORIES}' reversed>\n")
                top_cats = get_top_cats()
                for i, cat in enumerate(top_cats):
                    permalink = slugify(cat)
                    try:
                        title = cdict[cat]["title"]
                    except:
                        print(title)
                        raise SystemExit()
                    fh2.write(f'<li><a href="/{permalink}/">{title}</a></li>\n')
                fh2.write("</ol>\n")


def category_pages():
    #   ____      _     ____
    #  / ___|__ _| |_  |  _ \ __ _  __ _  ___  ___
    # | |   / _` | __| | |_) / _` |/ _` |/ _ \/ __|
    # | |__| (_| | |_  |  __/ (_| | (_| |  __/\__ \
    #  \____\__,_|\__| |_|   \__,_|\__, |\___||___/
    #                              |___/
    """Outputs the individual category pages and includes"""
    global ydict
    build_ydict()
    top_cats = get_top_cats()
    # Map every slug to a category:
    slugcat = defaultdict(list)
    for i, (fm, apost, combined) in enumerate(yaml_generator(YAMLESQUE)):
        if fm:
            if "keywords" in fm and "title" in fm:
                slug = slugify(fm["title"])
                keywords = fm["keywords"]
                keyword_list = keywords.split(", ")
                for keyword in keyword_list:
                    keyword = normalize_key(keyword)
                    if keyword in top_cats:
                        slugcat[keyword].append(slug)

    # Make the list of tuples for build_arrow():
    href_title_list = [(f"/{slugify(x)}/", cdict[x]["title"]) for x in top_cats]

    # Create the category pages:
    for i, cat in enumerate(top_cats):
        slug = slugify(cat)
        filename = f"{PATH}{REPO}cat_{slug}.md"
        include_file = f"cat_{slug}.md"
        permalink = f"/{slug}/"
        with open(filename, "w", encoding="utf-8") as fh:
            cat_str = f"""---
title: {cdict[cat]["title"]}
permalink: {permalink}
layout: default
---

# {cdict[cat]["title"]}

"""
            fh.write(cat_str)
            fh.write(f"{{% include {include_file} %}}\n")
    # Create the category includes:
    for i, cat in enumerate(top_cats):
        slug = slugify(cat)
        filename = f"{INCLUDES}cat_{slug}.md"
        with open(filename, "w", encoding="utf-8") as fh:
            posts_in_cat = len(slugcat[cat])
            fh.write(f"<ol>\n")
            for slug in slugcat[cat]:
                title = ydict[slug]["title"]
                aslug = slugify(title)
                adate = ydict[slug]["date"]
                description = ydict[slug]["description"]
                apermalink = f"{BLOG}{aslug}/"
                alink = f'<li><a href="{apermalink}">{title}</a> ({adate})\n<br/>{description}</li>\n'
                fh.write(alink)
            fh.write("</ol>\n")
            # Arrow Maker needs index, len of list and list of tuples.
            arrow_link = arrow_maker(i, len(top_cats), href_title_list)
            fh.write(arrow_link)


def chop_the_yaml():
    # __   __ _    __  __ _    |  ____ _                  _  _  _
    # \ \ / // \  |  \/  | |   | / ___| |__   ___  _ __  | || || |
    #  \ V // _ \ | |\/| | |   || |   | '_ \ / _ \| '_ \ | || || |
    #   | |/ ___ \| |  | | |___|| |___| | | | (_) | |_) ||_||_||_|
    #   |_/_/   \_\_|  |_|_____| \____|_| |_|\___/| .__/ (_)(_)(_)
    """Chop a YAMLesque text-file into the individual text-files (posts) it implies."""
    global ydict, cdict
    num_pages = len(ydict)
    href_title_list = [(f'{BLOG}{ydict[x]["slug"]}/', ydict[x]["title"]) for x in ydict]
    counter = 0
    top_cats = get_top_cats()
    for i, (fm, body, combined) in enumerate(yaml_generator(YAMLESQUE)):
        if fm and isinstance(fm, dict) and len(fm) > 2:
            # Print a progress indicator:
            magic_number = (i + 1) - 10
            if magic_number and not magic_number % 10:
                print(f"{magic_number} ", end="", flush=True)

            # Build the categories:
            keyword_list = fm["keywords"].split(", ")
            category_map = get_cat_map()

            # Gather the page's keywords:
            categories = set()
            for keyword in keyword_list:
                keyword = keyword.lower()
                normalized_keyword = normalize_key(keyword)
                for candidate in [keyword, normalized_keyword]:
                    if candidate in category_map:
                        categories.add(category_map[candidate])
            fm["categories"] = ", ".join(categories)

            # Format the date:
            adate = fm["date"]
            date_object = datetime.strptime(adate, "%a %b %d, %Y")
            adate = date_object.strftime("%Y-%m-%d")
            fm["date"] = adate

            # Build the permalink:
            stem = slugify(fm["title"])
            fm["permalink"] = f"{BLOG}{stem}/"

            filename = f"{OUTPUT_PATH}/{adate}-{stem}.md"
            if i == 1:
                print(filename)
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write("---\n")
                for afield in fm:
                    fh.write(f"{afield}: {sq(fm[afield])}\n")
                fh.write("layout: post\n")
                fh.write("group: blog\n")
                fh.write("---\n")
                fh.write(body)
                # Arrow Maker needs index, len of list and list of tuples.
                arrow_link = arrow_maker(counter, num_pages, href_title_list)
                fh.write(arrow_link)
                fh.write("\n## Categories\n")
                fh.write("\n<ul>")
                for acat in categories:
                    acat = normalize_key(acat)
                    try:
                        fh.write(f"\n<li><h4><a href='/{slugify(acat)}/'>{cdict[acat]['title']}</a></h4></li>")
                    except:
                        print(f"Category {acat} not found!")
                        raise SystemExit()
                fh.write("</ul>")
                counter += 1
    if counter > 1:
        print(f"{counter} files chopped!")


def make_drafts():
    #  __  __       _          ____             __ _       
    # |  \/  | __ _| | _____  |  _ \ _ __ __ _ / _| |_ ___ 
    # | |\/| |/ _` | |/ / _ \ | | | | '__/ _` | |_| __/ __|
    # | |  | | (_| |   <  __/ | |_| | | | (_| |  _| |_\__ \
    # |_|  |_|\__,_|_|\_\___| |____/|_|  \__,_|_|  \__|___/
    #                                                      
    """Because we can't preview drafts with Github Pages, the system publishes
    with a secret permalink so you can view the rendered draft in a no-CSS
    style that is appropriate for copy/pasting into Docs or Word."""
    for i, (fm, body, combined) in enumerate(yaml_generator(YAMLESQUE, drafts=True)):
        # Format the date:
        adate = fm["date"]
        date_object = datetime.strptime(adate, "%a %b %d, %Y")
        adate = date_object.strftime("%Y-%m-%d")
        fm["date"] = adate

        # Build the permalink:
        stem = slugify(fm["title"])
        fm["permalink"] = f"{BLOG}{stem}/"

        filename = f"{OUTPUT_PATH}/{adate}-{stem}.md"
        print(filename)
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("---\n")
            for afield in fm:
                if afield != "published":  # Necessary to render previews
                    fh.write(f"{afield}: {sq(fm[afield])}\n")
            fh.write("layout: plain\n")
            fh.write("---\n")
            fh.write(f"# {fm['title']}\n\n")
            fh.write(body)


def git_push():
    """Pushes the changes to Github"""
    #   ____ _ _     ____            _
    #  / ___(_) |_  |  _ \ _   _ ___| |__
    # | |  _| | __| | |_) | | | / __| '_ \
    # | |_| | | |_  |  __/| |_| \__ \ | | |
    #  \____|_|\__| |_|    \__,_|___/_| |_|
    # Git commands
    here = f"{PATH}{REPO}"
    git(here, f"add cat_*")
    git(here, "add _data/*")
    git(here, "add _posts/*")
    git(here, "add _includes/*")
    git(here, "add assets/images/*")
    git(here, f'commit -am "Pushing {REPO} to Github."', False)
    git(here, "push", False)


#  _   _      _                  __        __        _       _                 _
# | | | | ___| |_ __   ___ _ __  \ \      / /_ _ ___| |_ ___| | __ _ _ __   __| |
# | |_| |/ _ \ | '_ \ / _ \ '__|  \ \ /\ / / _` / __| __/ _ \ |/ _` | '_ \ / _` |
# |  _  |  __/ | |_) |  __/ |      \ V  V / (_| \__ \ ||  __/ | (_| | | | | (_| |
# |_| |_|\___|_| .__/ \___|_|       \_/\_/ \__,_|___/\__\___|_|\__,_|_| |_|\__,_|
#              |_|


def diagnose_yaml(yaml_str, YMLError):
    fig("YAML ERROR", "READ THE YAML LINE-BY-LINE UNTIL KAPUT...")
    for j, astr in enumerate(yaml_str.split("\n")):
        print(f"LINE {j + 1}--> {astr}")
    print()
    print("And here's the error:")
    print(f"YMLError.context_mark: {YMLError.context_mark}")
    print()
    print("And the breakdown of the error:")
    print(f"exec.context_mark: {YMLError.context_mark}")
    print(f"exec.problem_mark: {YMLError.problem_mark}")
    raise SystemExit()


def oget(DBNAME, slug):
    """Return a value from a database."""
    # There's a fancier version that will set the data too, but when you
    # want to just safely (no chance of writing) fetch db data, use this.
    with sqldict(DBNAME) as db:
        if slug in db:
            result = db[slug]
        else:
            result = None
    return result


def git(cwd, line_command, quiet=True):
    #        _ _
    #   __ _(_) |_   This is it. This is git.
    #  / _` | | __|  It does the simple deed.
    # | (_| | | |_   From a shell what it does well
    #  \__, |_|\__|  Is move things where you need.
    #  |___/
    """Run a Linux git command."""
    cmd = ["/usr/bin/git"] + shlex.split(line_command)
    show_cmd = " ".join(cmd)
    if quiet:
        print(".", end="", flush=True)
    else:
        print()
        print(f"Running: {show_cmd}")
    process = Popen(
        args=cmd,
        cwd=cwd,
        stdout=PIPE,
        stderr=PIPE,
        shell=False,
        bufsize=1,
        universal_newlines=True,
    )
    flush(process.stdout)


def flush(std):
    """Flush a stream."""
    # The console is great for watching programs run, but sometimes
    # the "streaming" effect you want doesn't work. This makes it work.
    for line in std:
        line = line.strip()
        if line:
            print(line)
            sys.stdout.flush()


def compare_files(file1, file2):
    """Compare two files. Return true of they are the same."""
    # If your source and destination files haven't even changed,
    # why re-publish? To-do: cut off the git push using this.
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            byte1 = f1.read(1)
            byte2 = f2.read(1)
            if byte1 != byte2:
                return False
            if not byte1:
                return True


def get_top_cats():
    """Returns the top categories either from the config or the cdict."""
    global cdict
    if "categories" in CONFIG and len(CONFIG["categories"]) > 1:
        category_map = get_cat_map()

        if len(category_map) < NUMBER_OF_CATEGORIES:
            top_cats = category_map.keys()
        else:
            top_cats = list(category_map.keys())[0:NUMBER_OF_CATEGORIES]
    else:
        top_cats = [x[1] for x in enumerate(cdict) if x[0] < NUMBER_OF_CATEGORIES]
    return top_cats


def sq(text):
    """Safely return a quoted string for YAML front matter."""
    # Do you put quotes around that YAML data or not?
    # Why decide when a function can decide for you.
    if not text:
        return text
    text = text.strip()
    text = text.strip('"')
    text = re.sub(r"\"{2,}", '"', text)
    text = text.replace('"', "'")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Replace all cariage returns and line feeds with spaces:
    text = re.sub(r"[\r\n]+", " ", text)
    # Replace all multiple spaces with a single space:
    text = re.sub(r"\s+", " ", text)
    # If any of the following characters (including single quotes) are in the text, do if:
    if re.search(r"[;:`]", text):
        text = f'"{text}"'
    return text


def normalize_key(keyword):
    """Returns normalized key for keyword dictionaries."""
    # Keywords appear in a lot of capitalization variations.
    # This function ensures that all variations will return
    # the same form for use as dictionary keys.
    keyword = keyword.lower()
    keyword = LEMMATIZER.lemmatize(keyword)
    keyword = keyword.lower()
    return keyword


def arrow_maker(i, length, href_title_list):
    """Returns the prev/next arrows for a page. It must be given a list of
    tuples containing the hrefs and titles of the pages in the sequence. It
    must also be given the index of the current page in the sequence. It will
    return the HTML for the prev/next arrows. Path issues must be dealt with
    beforehand in the tuple sequence."""

    # Arrows always appear on their implied sides: left on left and right on right.
    larr = '<span class="arrow">&larr;&nbsp;</span>'
    rarr = '<span class="arrow">&nbsp;&rarr;</span>'
  
    # Handle arrows at beginning and end of sequence
    if not i:
        older_slug, older_title = href_title_list[i + 1]
        newer_slug, newer_title = "", ""
    elif i < length - 1:
        older_slug, older_title = href_title_list[i + 1]
        newer_slug, newer_title = href_title_list[i - 1]
    else:
        older_slug, older_title = "", ""
        newer_slug, newer_title = href_title_list[i - 1]

    if PAST_IS_LEFT:
        if not i:
            rarr = ""
        elif i == length - 1:
            larr = ""
        left_link = f'<a href="{older_slug}">{older_title}</a>'
        right_link = f'<a href="{newer_slug}">{newer_title}</a>'
    else:
        if not i:
            larr = ""
        elif i == length - 1:
            rarr = ""
        left_link = f'<a href="{newer_slug}">{newer_title}</a>'
        right_link = f'<a href="{older_slug}">{older_title}</a>'

    outer_div = '<div class="arrow-links">'
    left_div = '<div class="post-nav-prev">'
    right_div = '<div class="post-nav-next">'
    end_div = '</div>'
    
    arrow_link = f"{outer_div}{left_div}{larr}{left_link}{end_div} &nbsp; {right_div}{right_link}{rarr}{end_div}{end_div}"
    return arrow_link


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


#  _____ _            ____  _                                             _
# |_   _| |__   ___  |  _ \| | __ _ _   _  __ _ _ __ ___  _   _ _ __   __| |
#   | | | '_ \ / _ \ | |_) | |/ _` | | | |/ _` | '__/ _ \| | | | '_ \ / _` |
#   | | | | | |  __/ |  __/| | (_| | |_| | (_| | | | (_) | |_| | | | | (_| |
#   |_| |_| |_|\___| |_|   |_|\__,_|\__, |\__, |_|  \___/ \__,_|_| |_|\__,_|
# Put new stuff here                |___/ |___/


def get_cat_map():
    """Returns a flat dictionary mapping keywords to categories."""

    craw = CONFIG["categories"]  # Get hierarchical mapping
    cmap = {}  # Will be a flat map.

    # First pass maps every parent key to itself.
    for key in craw:
        cmap[key] = key

    # Second pass maps every sub-list item to parent category (the flattening).
    for key in craw:
        if craw[key]:
            for j, sub_key in enumerate(craw[key]):
                if key != "filter":
                    if isinstance(sub_key, dict):
                        # A trailing colon was left on a subcat mapping
                        # Grab first key-name of accidental dict as val.
                        sub_key = list(sub_key.keys())[0]
                    cmap[sub_key] = key
    return cmap


#  _____ _                                 _             _
# |  ___| | _____      __   ___ ___  _ __ | |_ _ __ ___ | |
# | |_  | |/ _ \ \ /\ / /  / __/ _ \| '_ \| __| '__/ _ \| |
# |  _| | | (_) \ V  V /  | (_| (_) | | | | |_| | | (_) | |
# |_|   |_|\___/ \_/\_/    \___\___/|_| |_|\__|_|  \___/|_|
# This controls the entire (usually linear) flow. Edit for debugging.

print("- Deleting prior auto-generated pages from site.")
delete_old()  # Deletes old posts
print("- Checking for new posts needing AI-writing.")
sync_check()  # Catches YAMLESQUE file up with database of OpenAI responses
print(f"- Making an index.")
make_index()  # Builds index page of all posts (for blog page)
print("- Scanning for categories.")
category_scan()  # Builds global categories and builds category pages
print("- Chopping source file into individual pages: ", end="")
chop_the_yaml()  # Writes out all Jekyll-style posts
# print("- Making drafts.")
# make_drafts()  # Writes out all Jekyll-style drafts (optional / easy un-publish)
print("- Publishing site with git.", end="")
git_push()  # Pushes changes to Github (publishes)
print("DONE!")
print("To close this buffer, type :bdel [Enter].")
