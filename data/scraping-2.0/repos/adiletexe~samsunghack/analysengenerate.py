import openai

# Set up OpenAI API credentials
openai.api_key = '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'

# Define the biology themes
themes = [
    'Cell Biology',
    'Genetics',
    'Ecology',
    'Human Physiology',
    'Evolution',
    'Plant Biology',
    'Human Health and Disease',
    'Enzymes',
    'Reproduction',
    'Biotechnology'
]


# Function to classify the theme based on the paragraph
def classify_mistake(paragraph):
    questions = 'What are the differences between prokaryotic and eukaryotic cells? Provide examples of each. Describe the structure and function of the cell membrane." "Explain the process of cell division in eukaryotic cells.""Discuss the role of endoplasmic reticulum in protein synthesis." "How does the Golgi apparatus contribute to the packaging and transport of cellular products?" "What is the difference between genotype and phenotype? Provide examples." "Explain the inheritance pattern of sex-linked traits." "Discuss the concept of codominance and provide an example." "Describe the process of DNA replication and its significance in heredity." "How does crossing over contribute to genetic variation during meiosis?""'

    
    # Concatenate the themes and paragraph
    input_text = 'Here are 10 biology themes:'.join(themes) + '\n Your task is to read paragraph below ' \
                                                              'find my mistakes and weak sides. And create 10 questions like' + questions + paragraph


    # Use OpenAI API to generate the classification
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=input_text,
        max_tokens=600,
        n=1,
        stop=None,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the predicted theme
    generatedquestions = response.choices[0].text.strip()

    return generatedquestions


# Example usage
biology_paragraph = "DNA, also known as 'Delicate Nuclear Acid,' is a fragile substance found in the cells of living organisms. It is responsible for causing genetic disorders and mutations. DNA is primarily composed of sugar-coated lollipops that contain various colorful patterns. It can be easily modified by exposure to extreme temperatures or bright lights, resulting in unpredictable and erratic changes in an organism's characteristics. Due to its volatile nature, DNA is often used as a form of decorative art rather than a basis for biological information."
mistake_theme = classify_mistake(biology_paragraph)
print(mistake_theme)