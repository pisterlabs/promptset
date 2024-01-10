import os, json, argparse, subprocess
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)
OPENAI_API_KEY = client.api_key

def generate_new_labels(labels, url, title, description):
    """Generate new labels if the existing labels are inadequate."""
    messages = [
        {"role": "system", "content": """You are a helpful assistant designed to output JSON lists of labels.
        Think carefully about the labels you select.
        The labels you create should make it easier to organize and search for information."""},
        {"role": "user", "content": f"""Think of some keywords for this link.\n
         url: {url}\n
         title: {title}\n
         description: {description}\n
         
         **labels:**
         {labels}\n
        Write A MAXIMUM OF TWO label,description pairs to describe this link:\n
        *IMPORTANT* Make sure the labels are unique and highly descriptive."""}
    ]
    # Step 1: call the model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        temperature=1,
        seed=0,
        messages=messages,
    )
    response_message = response.choices[0].message
    return response_message


def create_new_labels(repo, label_list):
    """Create new labels for a GitHub repo."""
    new_labels_created = []
    for label in label_list:
        label_name = label["name"]
        label_description = label["description"]
        command = ["gh", "label", "create", "-R", repo, label_name, "-d", label_description]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stderr:
            print("Error:", result.stderr)
        else:
            print(f"Created label: {label_name}")
            new_labels_created.append(label_name)
    
    return new_labels_created


def request_labels_list(repo):
    with open('/dev/tty', 'w') as f:
        f.write(f"get_issues_labels_list: {repo}\n\n")
        per_page = 100
        command = ["gh", "label", "list", "-R", repo, "-L", "100", "--json", "name,description,color"]
        
        # Execute the command using subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        labels = json.loads(result.stdout)
        if labels:
            f.write(f"got {len(labels)} labels\n\n")
        # Print the information or do further processing if needed
        # for label in labels:
        #     print(f"Label Name: {label['name']}, Color: {label['color']}")

        # If an error occurs, print the error message
        if result.stderr:
            print("Error:", result.stderr)
        parsed_labels = ""
        label_dict = {}
        
        for label in labels:
            parsed_labels += f"{label['name']}: {label['description']}\n"
            # label_dict[label['name']] = label['description']
        return parsed_labels


def new_labels_needed(labels, url, title, description):
    adequate_labels_query = f"""Given the following bookmark:
    url: {url}
    title: {title}
    description: {description}

Are new labels needed to adequately delineate this bookmark? (True) or can you label it accurately with the existing labels? (False)
Only answer True if you are certain that new labels are needed. If you are unsure, then answer False.
Only reply with True or False.

    **labels:**
    {labels}

**Important**: Say nothing except true or false."""
    messages = [
        {"role": "system", "content": """You are a helpful assistant designed to answer binary questions with True or False."""},
        {"role": "user", "content": adequate_labels_query}
    ]
    # Step 1: call the model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0,
        seed=0,
        messages=messages,
    )
    response_message = response.choices[0].message
    print(f"New Labels Are Needed: {response_message.content}")
    if response_message.content == "True":
        return True
    else:
        return False


def pick_labels(url, title, description, labels):
    """
    Choose the labels to assign to a bookmark.
    """
    
    pick_labels_query = f"""Pick A MINIMUM OF THREE (3) labels from the list to describe this link:\n
    *IMPORTANT* Only pick from the labels provided. Output a JSON list of labels.
    url: {url}\ntitle: {title}\ndescription: {description}\nlabels: {labels}
    """

    messages = [
        {"role": "system", "content": """You are a helpful assistant designed to output JSON lists of labels. 
        Think carefully about the labels you select. 
        The labels you select should make it easier to organize and search for information. 
         **IMPORTANT** Only pick from the labels provided."""},
        {"role": "user", "content": pick_labels_query}
    ]
    # Step 1: call the model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        temperature=1,
        seed=0,
        messages=messages
    )
    # return a list of labels
    response_message = response.choices[0].message.content
    print(f"Labels Picked: {response_message}")
    return response_message

parser = argparse.ArgumentParser(description='Generate labels for a given bookmark.')
parser.add_argument('--url', metavar='url', type=str, help='The url of the bookmark.')
parser.add_argument('--title', metavar='title', type=str, help='The title of the bookmark.')
parser.add_argument('--description', metavar='description', type=str, help='The selected text of the bookmark.')
parser.add_argument('--repo', metavar='repo', type=str, help='The repo to get labels from.', default="irthomasthomas/undecidability")
args = parser.parse_args()

# Algorithm:
# 1. Request a list of labels from the repo.
# 2. Check if the existing labels are adequate.
# 3. If not, generate new labels.
# 4. Create the new labels.
# 5. Pick the labels to assign to the bookmark.
# 6. Return the labels.
labels_dict = {}

if args.url:
    labels = request_labels_list(args.repo)
    print(f"labels count: {len(labels)}")
    if new_labels_needed(labels, args.url, args.title, args.description):
        generated_labels = generate_new_labels(labels, args.url, args.title, args.description)
        generated_labels_list = json.loads(generated_labels.content)
        print(f"LABELS REQUESTED:\n {generated_labels_list}")
    picked_labels = json.loads(pick_labels(args.url, args.title, args.description, labels))
    # picked_labels = {"labels:":["label1", "label2"]}
    if generated_labels:
        # manually add the 'New Label' label picked_labels
        picked_labels["labels"].append("New Label") # TypeError: string indices must be integers, not 'str'
    
        labels_dict["picked_labels"] = picked_labels
        # add the generated label's name,description pairs to the picked labels as a list of dicts
        labels_dict["generated_labels"] = generated_labels_list
    print(f"LABELS PICKED:\n {labels_dict}")