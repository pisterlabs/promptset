import streamlit as st
import json
import pygsheets
from google.oauth2 import service_account
import re
from openai import OpenAI
import cohere
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
import re

# version 0.3 - Experiment for making sequential API calls for semantic search.

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()

os.environ["CO_API_KEY"]= st.secrets["cohere_api_key"]
co = cohere.Client()

scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

api_sheet = gc.open('api_outputs')
api_outputs = api_sheet.sheet1


# System prompt

def load_prompt(file_name):
    """Load prompt from a file."""
    with open(file_name, 'r') as file:
        return file.read()

# Function to ensure prompts are loaded into session state
def load_prompts():
    if 'keyword_model_system_prompt' not in st.session_state:
        st.session_state['keyword_model_system_prompt'] = load_prompt('prompts/keyword_model_system_prompt.txt')
    if 'response_model_system_prompt' not in st.session_state:
        st.session_state['response_model_system_prompt'] = load_prompt('prompts/response_model_system_prompt.txt')
    if 'app_into' not in st.session_state:
        st.session_state['app_intro'] = load_prompt('prompts/app_intro.txt')
    if 'keyword_search_explainer' not in st.session_state:
        st.session_state['keyword_search_explainer'] = load_prompt('prompts/keyword_search_explainer.txt')
    if 'semantic_search_explainer' not in st.session_state:
        st.session_state['semantic_search_explainer'] = load_prompt('prompts/semantic_search_explainer.txt')
    if 'relevance_ranking_explainer' not in st.session_state:
        st.session_state['relevance_ranking_explainer'] = load_prompt('prompts/relevance_ranking_explainer.txt')
    if 'nicolay_model_explainer' not in st.session_state:
        st.session_state['nicolay_model_explainer'] = load_prompt('prompts/nicolay_model_explainer.txt')

# Ensure prompts are loaded
load_prompts()

# Now you can use the prompts from session state
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
app_intro = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer = st.session_state['nicolay_model_explainer']

# Streamlit interface
st.title("Exploring RAG with Nicolay and Hay")

image_url = 'http://danielhutchinson.org/wp-content/uploads/2024/01/nicolay_hay.png'
#st.markdown(f'<img src="{image_url}" width="700">', unsafe_allow_html=True)
st.image(image_url, width=600)

st.subheader("**Navigating this App:**")
st.write("Expand the **How It Works?** box below for a walkthrough of the app. Continue to the search interface below to begin exploring Lincoln's speeches.")

with st.expander("**How It Works - Exploring RAG with Hay and Nicolay**"):
    st.write(app_intro)


# Query input
with st.form("Search Interface"):
    st.markdown("Enter your query below:")
    user_query = st.text_input("Query")

    st.write("**Search Options**:")
    st.write("Note that at least one search method must be selected to perform Response and Analysis.")
    perform_keyword_search = st.toggle("Weighted Keyword Search", value=True)
    perform_semantic_search = st.toggle("Semantic Search", value=True)
    # Always display the reranking toggle
    perform_reranking = st.toggle("Response and Analysis", value=False, key="reranking")

    # Display a warning message if reranking is selected without any search methods
    if perform_reranking and not (perform_keyword_search or perform_semantic_search):
        st.warning("Response & Analysis requires at least one of the search methods (keyword or semantic).")

    with st.expander("Additional Search Options (In Development)"):
        st.markdown("The Hay model will suggest keywords based on your query, but you can select your own criteria for more focused keyword search using the interface below.")

        st.markdown("Weighted Keywords")
        default_values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Default weights as floats
        user_weighted_keywords = {}
        for i in range(1, 6):
            col1, col2 = st.columns(2)
            with col1:
                keyword = st.text_input(f"Keyword {i}", key=f"keyword_{i}")
            with col2:
                weight = st.number_input(f"Weight for Keyword {i}", min_value=0.0, value=default_values[i-1], step=0.1, key=f"weight_{i}")

            if keyword:
                user_weighted_keywords[keyword] = weight

            # User input for year and text keywords
        st.header("Year and Text Filters")
        user_year_keywords = st.text_input("Year Keywords (comma-separated - example: 1861, 1862, 1863)")
        #user_text_keywords = st.text_input("Text Keywords")
        user_text_keywords = st.multiselect("Text Selection:", ['At Peoria, Illinois', 'A House Divided', 'Eulogy on Henry Clay', 'Farewell Address', 'Cooper Union Address', 'First Inaugural Address', 'Second Inaugural Address', 'July 4th Message to Congress', 'First Annual Message', 'Second Annual Message', 'Third Annual Message', 'Fourth Annual Message', 'Emancipation Proclamation', 'Public Letter to James Conkling', 'Gettysburg Address'])

    submitted = st.form_submit_button("Submit")
    if submitted:
        valid_search_condition = perform_keyword_search or perform_semantic_search

        if valid_search_condition:

            st.subheader("Starting RAG Process: (takes about 30-60 seconds in total)")


            # Load data
            #lincoln_speeches_file_path = 'C:\\Users\\danie\\Desktop\\Consulting Work\\Gibson - UTSA\\lincolnbot\\script development\\nicolay_assistant\\lincoln-speeches_final_formatted.json'
            lincoln_speeches_file_path = 'data/lincoln_speech_corpus.json'
            keyword_frequency_file_path = 'data/voyant_word_counts.json'
            lincoln_speeches_embedded = "lincoln_index_embedded.csv"

            # define functions

            def load_json(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data

            lincoln_data = load_json(lincoln_speeches_file_path)
            keyword_data = load_json(keyword_frequency_file_path)

            # Convert JSON data to a dictionary with 'text_id' as the key for easy access
            lincoln_dict = {item['text_id']: item for item in lincoln_data}

            # function for loading JSON 'text_id' for comparsion for semantic search results
            def get_source_and_summary(text_id):
                # Convert numerical text_id to string format used in JSON
                text_id_str = f"Text #: {text_id}"
                return lincoln_dict.get(text_id_str, {}).get('source'), lincoln_dict.get(text_id_str, {}).get('summary')


            def find_instances_expanded_search(dynamic_weights, original_weights, data, year_keywords=None, text_keywords=None, top_n=5):
                instances = []

                # original processing for text_keywords formatted as strings - however, inconsistencies in the finetuning dataset cause issues here. For now code below is used.
                #text_keywords_list = [keyword.strip().lower() for keyword in text_keywords.split(',')] if text_keywords else []

                if text_keywords:
                    if isinstance(text_keywords, list):
                        text_keywords_list = [keyword.strip().lower() for keyword in text_keywords]
                    else:
                        text_keywords_list = [keyword.strip().lower() for keyword in text_keywords.split(',')]
                else:
                    text_keywords_list = []


                for entry in data:
                    if 'full_text' in entry and 'source' in entry:
                        entry_text_lower = entry['full_text'].lower()
                        source_lower = entry['source'].lower()
                        summary_lower = entry.get('summary', '').lower()
                        keywords_lower = ' '.join(entry.get('keywords', [])).lower()

                        match_source_year = not year_keywords or any(str(year) in source_lower for year in year_keywords)
                        match_source_text = not text_keywords or any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', source_lower) for keyword in text_keywords_list)

                        if match_source_year and match_source_text:
                            total_dynamic_weighted_score = 0
                            keyword_counts = {}
                            keyword_positions = {}

                            combined_text = entry_text_lower + ' ' + summary_lower + ' ' + keywords_lower

                            for keyword in original_weights.keys():
                                keyword_lower = keyword.lower()
                                for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', combined_text):
                                    count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', combined_text))
                                    dynamic_weight = dynamic_weights.get(keyword, 0)

                                    if count > 0:
                                        keyword_counts[keyword] = count
                                        total_dynamic_weighted_score += count * dynamic_weight

                                        keyword_index = match.start()
                                        original_weight = original_weights[keyword]
                                        keyword_positions[keyword_index] = (keyword, original_weight)

                            if keyword_positions:
                                highest_original_weighted_position = max(keyword_positions.items(), key=lambda x: x[1][1])[0]
                                context_length = 300
                                start_quote = max(0, highest_original_weighted_position - context_length)
                                end_quote = min(len(entry_text_lower), highest_original_weighted_position + context_length)
                                snippet = entry['full_text'][start_quote:end_quote]

                                instances.append({
                                    "text_id": entry['text_id'],
                                    "source": entry['source'],
                                    "summary": entry.get('summary', ''),
                                    "quote": snippet.replace('\n', ' '),
                                    "weighted_score": total_dynamic_weighted_score,
                                    "keyword_counts": keyword_counts
                                })

                instances.sort(key=lambda x: x['weighted_score'], reverse=True)
                return instances[:top_n]

            # Updated main search function to use expanded search
            def search_with_dynamic_weights_expanded(user_keywords, json_data, year_keywords=None, text_keywords=None, top_n_results=5):
                total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
                relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}
                inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}
                max_weight = max(inverse_weights.values())
                normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}

                return find_instances_expanded_search(
                    dynamic_weights=normalized_weights,
                    original_weights=user_keywords,  # Using user-provided keywords as original weights for snippet centering
                    data=lincoln_data,
                    year_keywords=year_keywords,
                    text_keywords=text_keywords,
                    top_n=top_n_results
                )

            def get_embedding(text, model="text-embedding-ada-002"):
                text = text.replace("\n", " ")
                response = client.embeddings.create(input=[text], model=model)
                return np.array(response.data[0].embedding)

            def cosine_similarity(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                return dot_product / (norm_vec1 * norm_vec2)

            def search_text(df, user_query, n=5):
                user_query_embedding = get_embedding(user_query)
                df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_query_embedding))
                top_n = df.sort_values("similarities", ascending=False).head(n)
                return top_n, user_query_embedding

            def segment_text(text, segment_size=100):
                words = text.split()
                return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

            def compare_segments_with_query_parallel(segments, query_embedding):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(get_embedding, segment) for segment in segments]
                    segment_embeddings = [future.result() for future in futures]
                    return [(segments[i], cosine_similarity(segment_embeddings[i], query_embedding)) for i in range(len(segments))]


            def extract_full_text(record):
                marker = "Full Text:\n"
                # Check if the record is a string
                if isinstance(record, str):
                    # Finding the position where the 'Full Text' starts
                    marker_index = record.find(marker)
                    if marker_index != -1:
                        # Extracting text starting from the position after the marker
                        return record[marker_index + len(marker):].strip()
                    else:
                        return ""
                else:
                    # Handle cases where the record is NaN or None
                    return ""

            def remove_duplicates(search_results, semantic_matches):
                combined_results = pd.concat([search_results, semantic_matches])
                #st.write("Before Deduplication:", combined_results.shape)
                deduplicated_results = combined_results.drop_duplicates(subset='text_id')
                #st.write("After Deduplication:", deduplicated_results.shape)
                return deduplicated_results

            def format_reranked_results_for_model_input(reranked_results):
                formatted_results = []
                # Limiting to the top 3 results
                top_three_results = reranked_results[:3]
                for result in top_three_results:
                    formatted_entry = f"Match {result['Rank']}: " \
                                      f"Search Type - {result['Search Type']}, " \
                                      f"Text ID - {result['Text ID']}, " \
                                      f"Source - {result['Source']}, " \
                                      f"Summary - {result['Summary']}, " \
                                      f"Key Quote - {result['Key Quote']}, " \
                                      f"Relevance Score - {result['Relevance Score']:.2f}"
                    formatted_results.append(formatted_entry)
                return "\n\n".join(formatted_results)

            # Function to get the full text from the Lincoln data based on text_id for final display of matching results
            def get_full_text_by_id(text_id, data):
                return next((item['full_text'] for item in data if item['text_id'] == text_id), None)

            # Function to highlight truncated quotes for Nicolay model outputs
            def highlight_key_quote(text, key_quote):
                # Example based on your quotes
                # Split the key_quote into beginning and ending parts
                parts = key_quote.split("...")
                if len(parts) >= 2:
                    # Construct a regex pattern with the stable beginning and end, allowing for optional punctuation and spaces
                    pattern = re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
                else:
                    # If there's no '...', use the entire quote with optional punctuation and spaces at the end
                    pattern = re.escape(key_quote) + r"\s*[.;,]?"

                # Compile the regex pattern for efficiency
                regex = re.compile(pattern, re.IGNORECASE)

                # Find all matches
                matches = regex.findall(text)

                # Replace matches with highlighted version
                for match in matches:
                    text = text.replace(match, f"<mark>{match}</mark>")

                return text

            def record_api_outputs():
                now = dt.now()
                d1 = {'msg':[msg], 'date':[now]}
                df1 = pd.DataFrame(data=d1, index=None)
                sh1 = gc.open('api_outputs')
                wks1 = sh1[0]
                cells1 = wks1.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                end_row1 = len(cells1)
                api_outputs = api_sheet.sheet1
                wks1.set_dataframe(df1,(end_row1+1,1), copy_head=False, extend=True)


            if user_query:

                # Construct the messages for the model
                messages_for_model = [
                    {"role": "system", "content": keyword_prompt},
                    {"role": "user", "content": user_query}
                ]

             # Send the messages to the fine-tuned model
                response = client.chat.completions.create(
                    model="ft:gpt-3.5-turbo-1106:personal::8XtdXKGK",  # Replace with your fine-tuned model
                    messages=messages_for_model,
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                msg = response.choices[0].message.content
                record_api_outputs()

                # Parse the response to extract generated keywords
                api_response_data = json.loads(msg)
                initial_answer = api_response_data['initial_answer']
                model_weighted_keywords = api_response_data['weighted_keywords']
                model_year_keywords = api_response_data['year_keywords']
                model_text_keywords = api_response_data['text_keywords']

                # Check if user provided any custom weighted keywords
                if user_weighted_keywords:
                    # Use user-provided keywords
                    weighted_keywords = user_weighted_keywords
                    year_keywords = user_year_keywords.split(',') if user_year_keywords else []
                    text_keywords = user_text_keywords if user_text_keywords else []
                else:
                    # Use model-generated keywords
                    weighted_keywords = model_weighted_keywords
                    year_keywords = model_year_keywords
                    text_keywords = model_text_keywords


                with st.expander("**Hay's Response**", expanded=True):
                    st.markdown(initial_answer)
                    st.write("**How Does This Work?**")
                    st.write("The Initial Response based on the user quer is given by Hay, a finetuned large language model. This response helps Hay steer in the search process by guiding the selection of weighted keywords and informing the semantic search over the Lincoln speech corpus. Compare the Hay's Response Answer with Nicolay's Response and Analysis and the end of the RAG process to see how AI techniques can be used for historical sources.")

                # Use st.columns to create two columns
                col1, col2 = st.columns(2)

                # Display keyword search results in the first column
                with col1:

                    # Perform the dynamically weighted search
                    if perform_keyword_search:
                        search_results = search_with_dynamic_weights_expanded(
                            user_keywords=weighted_keywords,
                            json_data=keyword_data,
                            year_keywords=year_keywords,
                            text_keywords=text_keywords,
                            top_n_results=5  # You can adjust the number of results
                            )

                        st.markdown("### Keyword Search Results")

                        with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                            st.write(keyword_search_explainer)

                        for idx, result in enumerate(search_results, start=1):
                            expander_label = f"**Keyword Match {idx}**: *{result['source']}* `{result['text_id']}`"
                            with st.expander(expander_label):
                                st.markdown(f"{result['source']}")
                                st.markdown(f"{result['text_id']}")
                                st.markdown(f"{result['summary']}")
                                st.markdown(f"**Key Quote:**\n{result['quote']}")
                                st.markdown(f"**Weighted Score:** {result['weighted_score']}")
                                st.markdown("**Keyword Counts:**")
                                st.json(result['keyword_counts'])

                        with st.expander("**Keyword Search Metadata**"):
                            st.write("**Keyword Search Metadata**")
                            st.write("**User Query:**")
                            st.write(user_query)
                            st.write("**Model Response:**")
                            st.write(initial_answer)
                            st.write("**Weighted Keywords:**")
                            st.json(weighted_keywords)  # Display the weighted keywords
                            st.write("**Year Keywords:**")
                            st.json(year_keywords)
                            st.write("**Text Keywords:**")
                            st.json(text_keywords)
                            #st.json(text_keywords)  # Display the weighted keywords
                            st.write("**Raw Search Results**")
                            st.dataframe(search_results)
                            st.write("**Full Model Output**")
                            st.write(msg)

                # Display semantic search results in the second column
                with col2:
                    if perform_semantic_search:
                        embedding_size = 1536
                        st.markdown("### Semantic Search Results")

                        with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                            st.write(semantic_search_explainer)

                        # Before starting the semantic search
                        progress_text = "Semantic search in progress."
                        my_bar = st.progress(0, text=progress_text)

                        # Initialize the match counter
                        match_counter = 1

                        df = pd.read_csv(lincoln_speeches_embedded)
                        df['full_text'] = df['combined'].apply(extract_full_text)
                        df['embedding'] = df['full_text'].apply(lambda x: get_embedding(x) if x else np.zeros(embedding_size))
                        #st.write("Sample text_id from DataFrame:", df['Unnamed: 0'].iloc[0])

                        # After calculating embeddings for the dataset
                        my_bar.progress(20, text=progress_text)  # Update to 20% after embeddings

                        df['source'], df['summary'] = zip(*df['Unnamed: 0'].apply(get_source_and_summary))
                        #st.write("Sample source and summary from DataFrame:", df[['source', 'summary']].iloc[0])


                        # Perform initial semantic search, using HyDE approach
                        #semantic_matches, user_query_embedding = search_text(df, user_query, n=5)
                        semantic_matches, user_query_embedding = search_text(df, user_query + initial_answer, n=5)
                        # After performing the initial semantic search
                        my_bar.progress(50, text=progress_text)  # Update to 50% after initial search

                        # Loop for top semantic matches
                        for idx, row in semantic_matches.iterrows():
                            # Update progress bar based on the index
                            progress_update = 50 + ((idx + 1) / len(semantic_matches)) * 40
                            progress_update = min(progress_update, 100)  # Ensure it doesn't exceed 100
                            my_bar.progress(progress_update / 100, text=progress_text)  # Divide by 100 if using float scale
                            if match_counter > 5:
                                break

                            # Updated label to include 'text_id', 'source'
                            semantic_expander_label = f"**Semantic Match {match_counter}**: *{row['source']}* `Text #: {row['Unnamed: 0']}`"
                            with st.expander(semantic_expander_label, expanded=False):
                                # Display 'source', 'text_id', 'summary'
                                st.markdown(f"**Source:** {row['source']}")
                                st.markdown(f"**Text ID:** {row['Unnamed: 0']}")
                                st.markdown(f"**Summary:**\n{row['summary']}")

                                # Process for finding key quotes remains the same
                                segments = segment_text(row['full_text'])  # Use 'full_text' for segmenting
                                #segment_scores = compare_segments_with_query(segments, user_query_embedding)
                                segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
                                top_segment = max(segment_scores, key=lambda x: x[1])

                                st.markdown(f"**Key Quote:** {top_segment[0]}")
                                st.markdown(f"**Similarity Score:** {top_segment[1]:.2f}")

                            # Increment the match counter
                            match_counter += 1

                        my_bar.progress(100, text="Semantic search completed.")
                        time.sleep(1)
                        my_bar.empty()  # Remove the progress bar

                        with st.expander("**Semantic Search Metadata**"):
                            st.write("**Semantic Search Metadata**")
                            st.dataframe(semantic_matches)

                # Reranking results with Cohere's Reranker API Endpoint

                if perform_reranking:

                    if isinstance(search_results, list):
                        search_results = pd.DataFrame(search_results)

                    # Convert 'text_id' in search_results to numeric format
                    search_results['text_id'] = search_results['text_id'].str.extract('(\d+)').astype(int)

                    # Rename the identifier column in semantic_matches to align with search_results
                    semantic_matches.rename(columns={'Unnamed: 0': 'text_id'}, inplace=True)
                    semantic_matches['text_id'] = semantic_matches['text_id'].astype(int)

                    deduplicated_results = remove_duplicates(search_results, semantic_matches)

                    all_combined_data = []

                    # Format deduplicated results for reranking
                    for index, result in deduplicated_results.iterrows():
                        # Check if the result is from keyword search or semantic search
                        if result.text_id in search_results.text_id.values and perform_keyword_search:
                            # Format as keyword search result
                            combined_data = f"Keyword|Text ID: {result.text_id}|{result.summary}|{result.quote}"
                            all_combined_data.append(combined_data)
                        elif result.text_id in semantic_matches.text_id.values and perform_semantic_search:
                            # Format as semantic search result
                            segments = segment_text(result.full_text)
                            segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
                            top_segment = max(segment_scores, key=lambda x: x[1])

                            combined_data = f"Semantic|Text ID: {result.text_id}|{result.summary}|{top_segment[0]}"
                            all_combined_data.append(combined_data)

                    # Use all_combined_data for reranking
                    if all_combined_data:
                        st.markdown("### Ranked Search Results")
                        try:
                            reranked_response = co.rerank(
                                model='rerank-english-v2.0',
                                query=user_query,
                                documents=all_combined_data,
                                top_n=10
                            )

                            with st.expander("**How Does This Work?: Relevance Ranking with Cohere's Rerank**"):
                                st.write(relevance_ranking_explainer)

                            # DataFrame for storing all reranked results
                            full_reranked_results = []

                            for idx, result in enumerate(reranked_response):
                                combined_data = result.document['text']
                                data_parts = combined_data.split("|")

                                if len(data_parts) >= 4:
                                    search_type, text_id_part, summary, quote = data_parts
                                    text_id = str(text_id_part.split(":")[-1].strip())
                                    summary = summary.strip()
                                    quote = quote.strip()

                                    # Retrieve source information
                                    text_id_str = f"Text #: {text_id}"
                                    source = lincoln_dict.get(text_id_str, {}).get('source', 'Source information not available')

                                    # Store each result in the DataFrame
                                    full_reranked_results.append({
                                        'Rank': idx + 1,
                                        'Search Type': search_type,
                                        'Text ID': text_id,
                                        'Source': source,
                                        'Summary': summary,
                                        'Key Quote': quote,
                                        'Relevance Score': result.relevance_score
                                    })

                                    # Display only the top 3 results
                                    if idx < 3:
                                        expander_label = f"**Reranked Match {idx + 1} ({search_type} Search)**: `Text ID: {text_id}`"
                                        with st.expander(expander_label):
                                            st.markdown(f"Text ID: {text_id}")
                                            st.markdown(f"{source}")
                                            st.markdown(f"{summary}")
                                            st.markdown(f"Key Quote:\n{quote}")
                                            st.markdown(f"**Relevance Score:** {result.relevance_score:.2f}")
                        except Exception as e:
                            st.error("Error in reranking: " + str(e))

                    # Format reranked results for model input
                    formatted_input_for_model = format_reranked_results_for_model_input(full_reranked_results)

                    # Display full reranked results in an expander
                    with st.expander("**Result Reranking Metadata**"):
                        reranked_df = pd.DataFrame(full_reranked_results)
                        st.dataframe(reranked_df)
                        st.write("**Formatted Results:**")
                        st.write(formatted_input_for_model)


                    # API Call to the second GPT-3.5 model
                    if formatted_input_for_model:

                        # Construct the message for the model
                        messages_for_second_model = [
                            {"role": "system", "content": response_prompt},
                            {"role": "user", "content": f"User Query: {user_query}\n\n"
                                                        f"Initial Answer: {initial_answer}\n\n"
                                                        f"{formatted_input_for_model}"}
                        ]


                        # Send the messages to the finetuned model
                        second_model_response = client.chat.completions.create(
                            model="ft:gpt-3.5-turbo-1106:personal::8clf6yi4",  # Specific finetuned model
                            messages=messages_for_second_model,
                            temperature=0,
                            max_tokens=2000,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                        # Process and display the model's response
                        response_content = second_model_response.choices[0].message.content

                        if response_content:  # Assuming 'response_content' is the output from the second model

                            model_output = json.loads(response_content)

                            # Displaying the Final Answer
                            st.header("Nicolay's Response & Analysis:")

                            #with st.expander("Output Debugging:"):
                            #    st.write(response_content)

                            with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                                st.write(nicolay_model_explainer)

                            with st.expander("**Nicolay's Response**", expanded=True):
                                final_answer = model_output.get("FinalAnswer", {})
                                st.markdown(f"**Response:**\n{final_answer.get('Text', 'No response available')}")
                                if final_answer.get("References"):
                                    st.markdown("**References:**")
                                    for reference in final_answer["References"]:
                                        st.markdown(f"{reference}")


                            highlight_style = """
                                <style>
                                mark {
                                    background-color: #90ee90;
                                    color: black;
                                }
                                </style>
                                """

                            doc_match_counter = 0

                            if "Match Analysis" in model_output:
                                st.markdown(highlight_style, unsafe_allow_html=True)
                                for match_key, match_info in model_output["Match Analysis"].items():
                                    text_id = match_info.get("Text ID")
                                    formatted_text_id = f"Text #: {text_id}"
                                    key_quote = match_info.get("Key Quote", "")

                                    speech = next((item for item in lincoln_data if item['text_id'] == formatted_text_id), None)

                                    # Increment the counter for each match
                                    doc_match_counter += 1

                                    #if speech:
                                        # Use the doc_match_counter in the expander label
                                    #    expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                                    #    with st.expander(expander_label, expanded=False):
                                    #        st.markdown(f"**Source:** {speech['source']}")
                                    #        st.markdown(f"**Text ID:** {speech['text_id']}")
                                    #        st.markdown(f"**Summary:**\n{speech['summary']}")

                                            # Handling escaped line breaks and highlighting the key quote
                                    #        formatted_full_text = speech['full_text'].replace("\\n", "<br>").replace(key_quote, f"<mark>{key_quote}</mark>")

                                    #        st.markdown(f"**Key Quote:**\n{key_quote}")
                                    #        st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                    #        st.markdown(formatted_full_text, unsafe_allow_html=True)
                                    #else:
                                    #    with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                    #        st.markdown("Full text not found.")

                                    if speech:
                                        # Use the doc_match_counter in the expander label
                                        expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                                        with st.expander(expander_label, expanded=False):
                                            st.markdown(f"**Source:** {speech['source']}")
                                            st.markdown(f"**Text ID:** {speech['text_id']}")
                                            st.markdown(f"**Summary:**\n{speech['summary']}")

                                            # Attempt direct highlighting
                                            if key_quote in speech['full_text']:
                                                formatted_full_text = speech['full_text'].replace("\\n", "<br>").replace(key_quote, f"<mark>{key_quote}</mark>")
                                            else:
                                                # If direct highlighting fails, use regex-based approach
                                                formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                                formatted_full_text = formatted_full_text.replace("\\n", "<br>")

                                            st.markdown(f"**Key Quote:**\n{key_quote}")
                                            st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                            st.markdown(formatted_full_text, unsafe_allow_html=True)
                                    else:
                                        with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                            st.markdown("Full text not found.")


                            # Displaying the Analysis Metadata
                            with st.expander("**Analysis Metadata**"):
                                # Displaying User Query Analysis
                                if "User Query Analysis" in model_output:
                                    st.markdown("**User Query Analysis:**")
                                    for key, value in model_output["User Query Analysis"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                # Displaying Initial Answer Review
                                if "Initial Answer Review" in model_output:
                                    st.markdown("**Initial Answer Review:**")
                                    for key, value in model_output["Initial Answer Review"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                # Displaying Match Analysis
                                #if "Match Analysis" in model_output:
                                #    st.markdown("**Match Analysis:**")
                                #    for match_key, match_info in model_output["Match Analysis"].items():
                                #        st.markdown(f"- **{match_key}:**")
                                #        for key, value in match_info.items():
                                #            st.markdown(f"  - {key}: {value}")

                                # Displaying Match Analysis
                                # Displaying Match Analysis
                                # Displaying Match Analysis
                                # Displaying Match Analysis
                                # Displaying Match Analysis
                                if "Match Analysis" in model_output:
                                    st.markdown("**Match Analysis:**", unsafe_allow_html=True)
                                    for match_key, match_info in model_output["Match Analysis"].items():
                                        st.markdown(f"- **{match_key}:**", unsafe_allow_html=True)
                                        for key, value in match_info.items():
                                            if isinstance(value, dict):
                                                nested_items_html = "<br>".join([f"&emsp;&emsp;<b>{sub_key}:</b> {sub_value}" for sub_key, sub_value in value.items()])
                                                st.markdown(f"&emsp;<b>{key}:</b><br>{nested_items_html}<br>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"&emsp;<b>{key}:</b> {value}<br>", unsafe_allow_html=True)


                                # Displaying Meta Analysis
                                if "Meta Analysis" in model_output:
                                    st.markdown("**Meta Analysis:**")
                                    for key, value in model_output["Meta Analysis"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                # Displaying Model Feedback
                                if "Model Feedback" in model_output:
                                    st.markdown("**Model Feedback:**")
                                    for key, value in model_output["Model Feedback"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                st.write("**Full Model Output**:")
                                st.write(response_content)

        else:
            st.error("Search halted: Invalid search condition. Please ensure at least one search method is selected.")
