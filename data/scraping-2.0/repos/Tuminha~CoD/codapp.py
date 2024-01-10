import openai
import PyPDF2
from dotenv import load_dotenv
import os
import streamlit as st
import io
import json  # Add this line



# Load .env file
load_dotenv()

# Get API key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')


def read_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def initiate_cod(article_text):
    # Prepare the prompt
    context = article_text[:30000]  # Adjust this number based on your needs
    prompt = f"""Article: {{context}}
            You will generate increasingly concise, entity-dense summaries of the above article.

            Repeat the following 2 steps 5 times.

            Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
            Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

            A missing entity is:
            - relevant to the main story,
            - specific yet concise (5 words or fewer),
            - novel (not in the previous summary),
            - faithful (present in the article),
            - anywhere (can be located anywhere in the article).

            Guidelines:

            - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
            - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
            - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
            - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
            - Missing entities can appear anywhere in the new summary.
            - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

            Remember, use the exact same number of words for each summary.

            Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".""".format(context=context)

    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000  # You can adjust this based on your needs
        )

        # Check if 'choices' key exists
        if 'choices' in response:
            # Check if 'message' and 'content' keys exist in response['choices'][0]
            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                return response['choices'][0]['message']['content'].strip()
            else:
                return "Keys 'message' and/or 'content' not found in response['choices'][0]."
        else:
            return "Key 'choices' not found in response."

    except KeyError as e:
        return f"KeyError: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Streamlit code
st.title('CoD Initiator')

# Add info text
st.markdown("""
<span style="color:#045dd5">**What is CoD (Chain of Density)?**</span><br>
CoD is a specialized approach for generating text summaries. It starts with an initial summary that is sparse in terms of specific entities and information. Over iterative steps, the summary is enriched with missing salient entities to make it more informative and detailed, all without increasing its length.

<span style="color:#045dd5">**When Can CoD Be Used?**</span><br>
CoD is versatile and can be applied to any text requiring summarization, such as news articles, research papers, or long-form content. It's especially useful when you want a summary that is both comprehensive and concise.

<span style="color:#045dd5">**What Can You Expect?**</span><br>
Summaries generated using the CoD approach are more abstractive, exhibit greater fusion of information, and are less biased towards the lead information in the original text.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if st.button('Initiate the CoD'):
    # Create a progress bar
    progress_bar = st.progress(0)

    # Update the progress bar to indicate that the process has started
    progress_bar.progress(10)

    article_text = read_pdf(uploaded_file.read())
    result = initiate_cod(article_text)

    # Update the progress bar to indicate that the process is halfway done
    progress_bar.progress(50)

    # Add the following code snippet where you display the summaries
    try:
        result_json = json.loads(result)
        # Initialize human preference percentages based on the summary step (i+1)
        human_pref = {1: 8.3, 2: 30.8, 3: 23.0, 4: 22.5, 5: 15.5}

        # Iterate over the result and display each item in a formatted way
        for i, item in enumerate(result_json):
            human_pref_percentage = human_pref[i+1]  # Retrieve the percentage for the current step
            color = "yellow"  # Initialize to yellow

            # Assign color based on the human preference percentage
            if human_pref_percentage == max(human_pref.values()):
                color = "green"
            elif human_pref_percentage == min(human_pref.values()):
                color = "red"

            st.markdown(f"<span style='color:{color}'>**Summary {i+1}**</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{color}'>This result will most likely be preferred by {human_pref_percentage}% of humans</span>", unsafe_allow_html=True)
            st.markdown(f"**Missing Entities:** {item['Missing_Entities']}")
            st.markdown(f"**Denser Summary:** {item['Denser_Summary']}")
            st.markdown("---")

        # Display the raw JSON at the end with an explanation
        st.markdown("### Raw JSON Output")
        st.markdown("The following JSON contains all the details of the CoD operation. It's useful for developers or anyone interested in the raw data.")
        st.code(result, language='json')

        # Update the progress bar to indicate that the process is done
        progress_bar.progress(100)

    except json.JSONDecodeError:
        st.error("Failed to decode the result as JSON.")


