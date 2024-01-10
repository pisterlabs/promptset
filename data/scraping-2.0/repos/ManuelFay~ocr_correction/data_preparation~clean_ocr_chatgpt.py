import datasets
from illuin_llm_tools import OpenAIConnector, Prompt, PromptParameters

dataset = datasets.load_from_disk('./data/gallica_dirty')

print(dataset)

openai_connector = OpenAIConnector(cache_path="./data/.cache")

PROMPT_OG = """
Corrige l'OCR et enlève le bruit dans le texte suivant, provenant de manuscrits francais anciens.
Si certaines lignes contiennent des caractères spéciaux où sont incompréhensibles, ignore les afin de faciliter la lecture.

"""

prompts = [
    Prompt(
        id=i,
        text=PROMPT_OG + example['text'] + "\n\n",
        parameters=
        PromptParameters(
            model="gpt-3.5-turbo",
            max_tokens=2000,
            temperature=0.1,
        )
    )
    for i, example in enumerate(dataset)
]

# Make multi requests
responses = openai_connector.multi_requests(
    prompts,
    max_requests_per_second=50,
    progress_desc="Cleaning corpus",
    max_catching_retries=1,
    clean_cache_at_end=False,
)

# dataset from generator

def generator():
    for file, response, text in zip(dataset['file'], responses, dataset['text']):
        # add failsafe on response status
        if response.status == "OK":
            # verify that response.text is fairly similar to text using word overlap
            yield {'file': file, 'clean_text': response.text, 'text': text}

new_dataset = datasets.Dataset.from_generator(generator)

# Save the dataset to disk in data/gallica_clean
print(new_dataset[0])
new_dataset.save_to_disk('./data/gallica_clean')

# push to hub
new_dataset.push_to_hub('manu/gallica_ocr_cleaned')
