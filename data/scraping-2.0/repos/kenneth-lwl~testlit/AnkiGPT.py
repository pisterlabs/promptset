import streamlit as st
import PyPDF2
import openai
import os
import base64
import genanki

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    extracted_text = " ".join([page.extract_text() for page in pdf_reader.pages])
    return extracted_text

def split_text_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end += chunk_size - overlap
    return chunks

def create_anki_deck(flashcards_text):
    # Create a new Anki model
    model = genanki.Model(
    1607392319,
    "PDF2Anki",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
    ],
    templates=[
        {
        "name": "Card 1",
        "qfmt": "<h1 style='color:gray;font-size:12px;'>Created by tabmed.hk/pdf2anki</h1><br>❓: {{Question}}",
        "afmt": "<h1 style='color:gray;font-size:12px;'>Created by tabmed.hk/pdf2anki</h1><br>❓: {{Question}}<hr id=answer>👉: {{Answer}}",
        },
    ])

    # Create a new Anki deck
    deck = genanki.Deck(2059400110, "PDF2Anki")

    # Split the flashcards text into separate flashcards
    flashcards = flashcards_text.split('\n')

    # Add each flashcard to the deck
    for flashcard in flashcards:
        if flashcard and '; ' in flashcard:
            question, answer = flashcard.split('; ', 1)
            note = genanki.Note(model=model, fields=[question, answer])
            deck.add_note(note)

    # Generate the .apkg file
    apkg_filename = 'flashcards.apkg'
    genanki.Package(deck).write_to_file(apkg_filename)

    # Read the contents of the .apkg file
    with open(apkg_filename, 'rb') as f:
        apkg_contents = f.read()

    # Generate a download link for the .apkg file
    download_link = get_file_download_link(apkg_contents, 'flashcards.apkg', is_binary=True)
    return download_link

def generate_anki_flashcards(text, chunk_size, overlap, api_key, model_choice):
    openai.api_key = api_key
    text_chunks = split_text_into_chunks(text, chunk_size, overlap)
    flashcards = ''
    current_flashcard = ''

    # Check if the number of chunks exceeds 1000
    if len(text_chunks) > 500:
        st.error('The PDF is too large and creates more than 500 chunks. Please reduce the size of the PDF or increase the chunk size.')
        return

    # Create progress bar
    progress_bar = st.progress(0)
    # Create an empty slot
    placeholder = st.empty()
    # Create placeholders for the chunk and flashcard display

    chunk_display = st.empty()
    flashcard_display = st.empty()

    for i, chunk in enumerate(text_chunks):
        print(text_chunks)
        # Update the placeholder with current chunk
        placeholder.text(f'Processing chunk {i+1}/{len(text_chunks)}')
        # Update the chunk placeholder with current chunk
        chunk_display.text(f'Chunk {i+1}/{len(text_chunks)}: {chunk}')
        
      # Modify the message prompt according to the selected language
        if language == 'Spanish':
            question_prompt = "Por favor, genera tarjetas de estudio a partir del texto proporcionado, asegurándote de que cada pregunta y su respuesta comiencen en una nueva línea. Cada tarjeta de estudio debe seguir este formato: '¿Pregunta?; Respuesta.' Por ejemplo: '¿Cuál es el mecanismo de acción de los diuréticos de asa?; Inhibición de la reabsorción de Na+ y Cl-.' '¿Cómo afectan los diuréticos de asa a la excreción renal de agua y Na+?; Aumentan la excreción renal de agua y Na+.' Es esencial que cada par de preguntas y respuestas esté separado por una línea en blanco. Además, asegúrate de generar solo una pregunta por tarjeta de estudio. Aquí está el texto proporcionado: "
        elif language == 'French':
            question_prompt = "Veuillez générer des flashcards à partir du texte donné, en veillant à ce que chaque question et sa réponse commencent sur une nouvelle ligne. Chaque flashcard doit suivre ce format : 'Question ?; Réponse.' Par exemple : 'Quel est le mécanisme d'action des diurétiques de l'anse ?; Inhibition de la réabsorption de Na+ et Cl-.' 'Comment les diurétiques de l'anse affectent-ils l'excrétion rénale d'eau et de Na+ ?; Ils augmentent l'excrétion rénale d'eau et de Na+.' Il est essentiel que chaque paire de questions et réponses soit séparée par une ligne blanche. De plus, veuillez vous assurer de générer une seule question par flashcard. Voici le texte fourni : "
        elif language == 'German':
            question_prompt = "Bitte erstellen Sie Lernkarten aus dem gegebenen Text und stellen Sie sicher, dass jede Frage und ihre Antwort auf einer neuen Zeile beginnen. Jede Lernkarte sollte diesem Format folgen: 'Frage?; Antwort.' Zum Beispiel: 'Was ist der Wirkmechanismus von Schleifendiuretika?; Hemmung der Na+- und Cl--Resorption.' 'Wie beeinflussen Schleifendiuretika die renale Ausscheidung von Wasser und Na+?; Sie erhöhen die renale Ausscheidung von Wasser und Na+.' Es ist wesentlich, dass jedes Frage-Antwort-Paar durch eine Leerzeile getrennt ist. Stellen Sie außerdem sicher, dass Sie pro Lernkarte nur eine Frage generieren. Hier ist der bereitgestellte Text: "
        elif language == 'Traditional Chinese':
            question_prompt = "請從給定的文本中生成學習卡，確保每個問題及其答案都從新的一行開始。每張學習卡都應該遵循這種格式：'問題？;答案。'例如：'利尿劑的作用機制是什麼？; 抑制Na+和Cl-的重吸收。' '利尿劑如何影響腎臟對水和Na+的排泄？; 它們增加了腎臟對水和Na+的排泄。'此外，請確保每張學習卡只生成一個問題。這是提供的文本："
        elif language == 'Simplified Chinese':
            question_prompt = "请从给定的文本中生成学习卡，确保每个问题及其答案都从新的一行开始。每张学习卡都应该遵循这种格式：'问题？;答案。'例如：'利尿剂的作用机制是什么？; 抑制Na+和Cl-的重吸收。' '利尿剂如何影响肾脏对水和Na+的排泄？; 它们增加了肾脏对水和Na+的排泄。'此外，请确保每张学习卡只生成一个问题。这是提供的文本："
        else:
            question_prompt = "Please generate flashcards from the given text, ensuring each question and its answer start on a new line. Each flashcard should follow this format: 'Question?; Answer.' For example: 'What is the mechanism of action of loop diuretics?; Inhibition of Na+ and Cl- reabsorption.' 'How do loop diuretics affect renal excretion of water and Na+?; They increase renal excretion of water and Na+.' It's essential that each question and answer pair is separated by a blank line. The question and the answer must be separated by a semi-colon. Also, please make sure to generate only one question per flashcard. Here is the provided text: "

        message_prompt = [
            {"role": "system", "content": "You are a highly skilled assistant that specializes in creating educational Anki active recall flashacards."},
            {"role": "user", "content": f"{question_prompt} {chunk}"}
        ]
        
        api_response = openai.ChatCompletion.create(
            model=model_choice,
            messages=message_prompt,
            temperature=temperature,
            max_tokens=3500
        )
        current_flashcard = api_response['choices'][0]['message']['content']


        # Only add the flashcard if it contains a question and answer separated by a semi-colon
        if '; ' in current_flashcard:
            flashcards += '\n\n' + current_flashcard

        # Update the flashcard placeholder with the newly generated flashcard
        flashcard_display.text(f'Flashcard: {current_flashcard}')

        # Update the progress bar
        progress_bar.progress((i + 1) / len(text_chunks))

    placeholder.empty()
    return flashcards


def get_file_download_link(file, filename, is_binary=False):
    if is_binary:
        b64 = base64.b64encode(file).decode()
    else:
        b64 = base64.b64encode(file.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'


MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # convert to bytes
MAX_WORD_COUNT = 5000

st.title('📃 Tabmed - PDF2Anki')
st.caption('Version pre-release alpha v0.20 - last updated 18 July 2023 - changelog: added multi-lingual flashcard export for Spanish, French, German and Chinese')
st.caption('Converts PDF files such as lecture slides, notes and PPTs into a .txt file that can be imported into Anki and converted into flashcards automatically. A preformmated and clean document will yield a better output. Images will not be read with this version.')
st.caption('After the .txt file has been downloaded, go through it and check for any errors. Then import it to Anki by File -> Import -> Select flashcards.txt -> Import')
st.caption('Some questions might stack. Ensure that each question is separated by a new line before importing it to Anki.')
st.caption('Due to excessive demand, there is a word limit cap of 5000 words. Split your pdf file or contact info@tabmed.hk if you would like to bypass this limit.')
uploaded_file = st.file_uploader('Please upload your PDF file', type='pdf')

# Add a language selection option
language = st.selectbox('Select the language of the LECTURE/NOTE material', ['English', 'Spanish', 'French', 'German', 'Traditional Chinese', 'Simplified Chinese'], help='Select the language of the uploaded material.')

chunk_size = st.slider('Enter the chunk size. (Default: 500)', 
                       min_value=300,  # set a minimum value
                       max_value=700,  # set a maximum value
                       value=500, 
                       step=1, 
                       help='The chunk size determines the amount of text from the PDF that the program will process at once for generating flashcards. A smaller size may yield more specific flashcards, while a larger size could provide broader context.')

overlap = st.slider('Enter the chunk overlap. (Default: 50)', 
                    min_value=20, 
                    max_value=80,  # set a maximum value
                    value=50, 
                    step=1, 
                    help='The chunk overlap determines the amount of text from the end of one chunk that will be included at the start of the next chunk. This can help avoid sentences being cut off in the middle.')

temperature = st.slider('Set the AI model temperature. (Default: 0.2)', 
                        min_value=0.1,  # set a minimum value
                        max_value=1.0,  # set a maximum value
                        value=0.2, 
                        step=0.1, 
                        help='The temperature parameter controls the randomness of the AI model\'s output. A higher temperature will make the output more diverse but also more risky, while a lower temperature makes the output more focused and deterministic. We recommend a low temperature setting like 0.1 or 0.2.')


# api_key = st.text_input('Please enter your OpenAI API Key', help='At the moment, we only support your own API key but this will change in the future! Meanwhile, retrieve your OpenAI API Key from https://platform.openai.com/account/api-keys')
# model_choice = st.selectbox('Select the AI model to be used (please consider donating if you select GPT4 - it is expensive!)', ['gpt-3.5-turbo', 'gpt-4'], help='GPT4 is extremely expensive for us to maintain. Please consider donating us a coffee if you select GPT4.')
api_key = os.getenv('OPENAI_API_KEY')

# if api_key.strip() == '':
#     st.error('Please input your OpenAI API key before proceeding.')

if uploaded_file is None:
    st.error('Please upload a PDF file before proceeding.')
elif uploaded_file.size > MAX_FILE_SIZE_BYTES:
    st.error('Our demand is too high right now. We have currently limited file upload to 5 MB for now whilst we scale our severs. The uploaded file is too large. Please upload a file that is 5 MB or less.')
elif st.button('Generate Flashcards'):
    pdf_text = extract_text_from_pdf(uploaded_file)
    # Count the number of words in the text
    word_count = len(pdf_text.split())

    # Check word count
    if word_count > MAX_WORD_COUNT:
        st.error(f"Due to excessive demand, we have set a word limit cap for the PDF. The uploaded file exceeds the maximum allowed word count of {MAX_WORD_COUNT}. Contact us at info@tabmed.hk if you would like to bypass this limit.")
    else:
        flashcards = generate_anki_flashcards(pdf_text, chunk_size, overlap, api_key, "gpt-3.5-turbo")
        del pdf_text  # Clear the pdf_text variable from memory
        download_link = get_file_download_link(flashcards, 'flashcards.txt')
        apkg_download_link = create_anki_deck(flashcards)
        del flashcards  # Clear the flashcards variable from memory
        st.success('Flashcards successfully created! Click the link below to download. Please make sure to separate all question and answer pairs on a new pagragraph on the .txt file before importing it to Anki. Some question and answer pairs might stick to the same paragraph.')
        st.markdown(download_link, unsafe_allow_html=True)
        st.markdown(apkg_download_link, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
