import os
import csv
import time
import json
import re 
import uuid
import yaml
import boto3
import botocore
import requests
import pandas as pd
import cv2
import openai
import streamlit as st
import replicate
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import random



def upload_image_to_s3(bucket_name, file_path, s3_key, overwrite=False):
    # Initialize boto3 client for S3
    s3 = boto3.client('s3')

    try:
        # Check if the file already exists
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        if not overwrite:
            print(f"File {s3_key} already exists in bucket {bucket_name}.")
            # Get the public URL of the file
            image_url = f'https://{bucket_name}.s3.amazonaws.com/{s3_key}'
            return image_url
    except botocore.exceptions.ClientError:
        pass  # If the file does not exist, we'll upload it below

    # Upload the file (either it doesn't exist or overwrite is True)
    s3.upload_file(file_path, bucket_name, s3_key, ExtraArgs={'ACL':'public-read'})
    print(f"Uploaded file {s3_key} to bucket {bucket_name}.")

    # Get the public URL of the file
    image_url = f'https://{bucket_name}.s3.amazonaws.com/{s3_key}'

    return image_url

def prompt_gpt(input, model="gpt-4", temperature=1, max_tokens=100, top_p=1, frequency_penalty=0, presence_penalty=0, system_message="You are a nice robot :)"):
    # Hardcoded system message    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"{input}"
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    print(response)

    return response.choices[0].message.content

def replace_value_and_save(old_value, new_value, csv_file='output.csv'):
    # Read the DataFrame from the CSV file
    df = pd.read_csv(csv_file)
    
    # Replace the old value with the new value in the entire DataFrame
    df.replace(old_value, new_value, inplace=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

def update_field_and_save(uuid, field_name, new_value, csv_file='output.csv'):
    # Read the DataFrame from the CSV file
    df = pd.read_csv(csv_file)
    
    # Update the specific field for the given UUID
    df.loc[df['uuid'] == uuid, field_name] = new_value
    
    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

def fetch_and_filter_data(statuses=None, rating=None, filter_by_user_fix_suggestions=False, filter_for_watermark_not_removed=False):
    # Read the updated csv file
    output_df = pd.read_csv('output.csv')

    # If statuses are provided, filter the DataFrame
    if statuses:
        output_df = output_df[output_df['status'].isin(statuses)]
        print(f"Filtered by statuses {statuses}, number of rows: {len(output_df)}")

    # If rating is provided, filter the DataFrame
    if rating is not None:
        # Process the 'user_rating' column
        processed_ratings = output_df['user_rating'].apply(lambda x: json.loads(x)[0]['rating'])

        # Filter the DataFrame based on the rating
        output_df = output_df[processed_ratings == rating]
        print(f"Filtered by rating {rating}, number of rows: {len(output_df)}")

    # If filter_by_user_fix_suggestions is True, filter rows where user_fix_suggestions is not empty
    if filter_by_user_fix_suggestions:
        output_df = output_df[output_df['user_fix_suggestions'].notna()]
        print(f"Filtered by user_fix_suggestions, number of rows: {len(output_df)}")

    # If filter_for_watermark_not_removed is True, filter rows where watermark_removed is empty or False
    if filter_for_watermark_not_removed:
        output_df = output_df[(output_df['watermark_removed'].isna()) | (output_df['watermark_removed'] == False)]
        print(f"Filtered for watermark not removed, number of rows: {len(output_df)}")

    return output_df

def display_rows_with_editing(status='reviewed', allow_action=True):

    def process_caption(row, cols, disabled=False):
        if row['caption'].startswith('[') and row['caption'].endswith(']'):
            caption_dict = {k: v for d in json.loads(row['caption']) for k, v in d.items()}
            for key, value in caption_dict.items():
                unique_key = f"{row['uuid']}_{key}"
                new_value = cols[2].text_input(key, value, key=unique_key, disabled=disabled)
                # Update the value in the DataFrame if it has changed
                if new_value != value and not disabled:
                    caption_dict[key] = new_value
                    row['caption'] = json.dumps([caption_dict])
                    update_field_and_save(row['uuid'], 'caption', row['caption'])
        else:
            cols[2].error(f"Invalid JSON format in caption: {row['caption']}")

    def handle_buttons(status, row, cols):
        button_labels = {
            'reviewed': ('Approve', 'approved', 'Reject', 'rejected'),
            'approved': ('Move to reviewed', 'reviewed', 'Reject', 'rejected')
        }
        if status in button_labels:
            button1_label, status1, button2_label, status2 = button_labels[status]
            button1 = cols[2].button(button1_label, key=f'{status1}-{row["uuid"]}', type="primary")
            button2 = cols[2].button(button2_label, key=f'{status2}-{row["uuid"]}')
            if button1:
                update_field_and_save(row['uuid'], 'status', status1)
                st.experimental_rerun()
            elif button2:
                update_field_and_save(row['uuid'], 'status', status2)
                st.experimental_rerun()
    
    def pagination_buttons(total_pages, location):
        # Add previous and next buttons
        cols = st.columns([1,1,1,9])
        prev_button = cols[1].button('Previous', key=f'prev-{location}', disabled=(st.session_state.page_num == 1))
        next_button = cols[2].button('Next', key=f'next-{location}', disabled=(st.session_state.page_num == total_pages))

        # Display the page information in the third column
        cols[0].info(f"Page {st.session_state.page_num} of {total_pages}")

        # Update the page number if a button is clicked
        if prev_button:
            st.session_state.page_num = max(1, st.session_state.page_num - 1)
            st.experimental_rerun()
        elif next_button:
            st.session_state.page_num = min(total_pages, st.session_state.page_num + 1)
            st.experimental_rerun()

    def calculate_pagination(data, rows_per_page=50):
        if 'page_num' not in st.session_state:
            st.session_state.page_num = 1

        total_pages = len(data) // rows_per_page
        if len(data) % rows_per_page > 0:
            total_pages += 1

        if st.session_state.page_num > total_pages or st.session_state.page_num < 1:
            st.session_state.page_num = 1
        
        return total_pages, rows_per_page
    
    data = fetch_and_filter_data([status],rating=2)

    total_pages, rows_per_page = calculate_pagination(data)

    st.success(f"There are **{len(data)} rows** in the {status} status.")

    pagination_buttons(total_pages, 'top')
    
    st.markdown('***')

    rows = data.tail(rows_per_page * st.session_state.page_num).head(rows_per_page)

    for _, row in rows.iterrows():
        cols = st.columns([1,1,1])
        # Display the images in the columns
        cols[0].image(row['image_0_location'])
        cols[1].image(row['image_1_location'])
        # Display the instructions in the third column
        process_caption(row, cols, disabled=not allow_action)

        if allow_action:
            handle_buttons(status, row, cols)

        st.markdown('***')
    
    pagination_buttons(total_pages, 'bottom')

def main():

    load_dotenv(override=True)

    st.set_page_config(layout="wide") 

    sections = ['Input Videos', 'Extract Frames', 'Describe Frames', 'Caption Pairs', 'Send To Review Queue', 'Final Review', 'Approved Pairs']

    section = st.radio('Select Section', sections, horizontal=True)


    if section == 'Input Videos':

        def filter_chunk_on_keywords(chunk, keyword, negative_keywords, additional_keywords="", skip_existing_videos=False):
            keywords = [kw.strip() for kw in keyword.split(',')]
            keyword_pattern = '|'.join(keywords)
            filtered_chunk = chunk[chunk['name'].str.contains(keyword_pattern, case=False)]
                    
            if isinstance(negative_keywords, str):
                negative_keywords = [kw.strip() for kw in negative_keywords.split(',') if kw.strip()]
                    
            if negative_keywords:
                neg_keyword_pattern = '|'.join(negative_keywords)
                try:
                    filtered_chunk = filtered_chunk[~filtered_chunk['name'].str.contains(neg_keyword_pattern, case=False)]
                except re.error as e:                                
                    pass  
            
            if additional_keywords:
                contains_terms = [term.strip() for term in additional_keywords.split(',') if term.strip()]
                contains_pattern = '|'.join(contains_terms)
                filtered_chunk = filtered_chunk[filtered_chunk['name'].str.contains(contains_pattern, case=False)]
            
            if skip_existing_videos:
                if os.path.isfile('output.csv'):
                    output_df = pd.read_csv('output.csv')
                    filtered_chunk = filtered_chunk[~filtered_chunk['videoid'].isin(output_df['video_id'])]

            return filtered_chunk

        def download_dataset(url, filename):

            response = requests.get(url)
            
            if response.status_code == 200:        
                with open(filename, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Failed to download file: {response.status_code}")

            st.experimental_rerun()

        def calculate_total_matches(positive_keywords, negative_keywords, additional_keywords,skip_existing_videos):
            
            total_matches = 0

            total_placeholder = st.empty()

            chunksize = 10000
            total_rows = sum(1 for row in open('results_10M_train.csv', 'r'))
            
            rows_to_process = int(total_rows * 0.1)

            for i, chunk in enumerate(pd.read_csv('results_10M_train.csv', chunksize=chunksize)):
            
                if (i * chunksize) >= rows_to_process:
                    break
                
                filtered_chunk = filter_chunk_on_keywords(chunk, positive_keywords, negative_keywords, additional_keywords,skip_existing_videos)

                # Add the number of matches in this chunk to the total
                total_matches += len(filtered_chunk)

            return total_matches * 10

        def search_videos(positive_keywords, negative_keywords, start=0,additional_keywords="",skip_existing_videos=False):
            
            results = pd.DataFrame()
        
            chunksize = 1000 
            for chunk in pd.read_csv('results_10M_train.csv', chunksize=chunksize):
                
                filtered_chunk = filter_chunk_on_keywords(chunk,positive_keywords,negative_keywords,additional_keywords,skip_existing_videos)
                            
                results = pd.concat([results, filtered_chunk])
                            
                if len(results) >= start + 50:
                    break
            
            return results.iloc[start:start+50]

        def create_next_button(location):
            col1, col2, col3 = st.columns([1,5,1])

            with col3:
                if st.button('Next', key=f'next-button-{location}', type="primary"):
                    st.session_state.start += 50  # Increase the start index by 10
                    st.session_state.results = search_videos(positive_keywords, negative_keywords, st.session_state.start,additional_keywords)
                    st.experimental_rerun()

        def download_matching_rows(positive_keywords, negative_keywords, filename, num_rows, additional_keywords="", skip_existing_videos=False):
            results = pd.DataFrame()
            output_df = pd.read_csv('output.csv') if os.path.isfile('output.csv') else pd.DataFrame()

            chunksize = 1000
            for chunk in pd.read_csv('results_10M_train.csv', chunksize=chunksize):

                filtered_chunk = filter_chunk_on_keywords(chunk, positive_keywords, negative_keywords, additional_keywords,skip_existing_videos)

                results = pd.concat([results, filtered_chunk])

                if len(results) >= num_rows:
                    break

            results = results.iloc[:num_rows]

            if not filename.endswith('.csv'):
                filename += '.csv'

            results.to_csv(filename, index=False)

        def update_keywords_file(positive_keywords, negative_keywords, additional_keywords, keywords_data):
            # Check if any keyword has been updated
            if (positive_keywords != keywords_data.get('positive_keywords', '') or 
                negative_keywords != keywords_data.get('negative_keywords', '') or 
                additional_keywords != keywords_data.get('additional_keywords', '')):
                
                # Update the YAML file
                with open('configs/keywords.yaml', 'w') as file:
                    yaml.dump({
                        'positive_keywords': positive_keywords,
                        'negative_keywords': negative_keywords,
                        'additional_keywords': additional_keywords
                    }, file)

                st.experimental_rerun()

        if 'start' not in st.session_state:
            st.session_state.start = 0
            st.session_state.results = pd.DataFrame()

        if not os.path.isfile('results_10M_train.csv' or  'results_2M_train.csv'):

            st.write('Dataset not found in the current directory.')    

            btn1, btn2, btn3 = st.columns([1,1,2])

            with btn1:
                if st.button('Download Webvid 10m Dataset'):
                    download_dataset('http://www.robots.ox.ac.uk/~maxbain/webvid/results_10M_train.csv', 'results_10M_train.csv')
                                    
            with btn2:
                if st.button('Download Webvid 2m Dataset'):
                    download_dataset('http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv', 'results_2M_train.csv')
                                    
            st.markdown('***')

        else:

            with st.sidebar:

                st.title('Video Searcher')

                with open('configs/keywords.yaml', 'r') as file:
                    keywords_data = yaml.safe_load(file)

                positive_keywords = st.text_input('Enter keywords:', value=keywords_data.get('positive_keywords', ''))
                negative_keywords = st.text_area('Enter negative keywords:', value=keywords_data.get('negative_keywords', ''))
                additional_keywords = st.text_area('Also filter for any of the following keywords:', value=keywords_data.get('additional_keywords', ''))

                skip_existing_videos = st.checkbox('Skip videos that are already present:', value=True)

                update_keywords_file(positive_keywords, negative_keywords, additional_keywords, keywords_data)

                col1, col2, col3 = st.columns([2,2,2])

                with col1:
                    if st.button('Search'):
                        st.session_state.start = 0  # Reset the start index when a new search is performed
                        st.session_state.results = search_videos(positive_keywords, negative_keywords, st.session_state.start,additional_keywords,skip_existing_videos)

                if not st.session_state.results.empty:
                    
                
                    with col2:
                        
                        if st.session_state.start != 0:
                            if st.button('Update search'):
                                st.session_state.results = search_videos(positive_keywords, negative_keywords, st.session_state.start,additional_keywords,skip_existing_videos)
                        else:
                            st.write('')

                    st.markdown('***')

                    if st.button('Estimate total matches'):
                        total_matches = calculate_total_matches(positive_keywords, negative_keywords,additional_keywords,skip_existing_videos)
                        st.success(f"Estimated total matches: {total_matches}")

                    st.markdown('***')

                    filename = st.text_input('Enter filename for download:',value='results')

                    download_all = st.checkbox('Download all matching rows',value=True)

                    if not download_all:
                        num_rows = st.number_input('Enter the maximum number of rows to download:', min_value=1, value=500)
                    else:
                        num_rows = 1000000000
                                    
                    if st.button('Download matching rows'):            
                        download_matching_rows(positive_keywords, negative_keywords, filename + '.csv', num_rows, additional_keywords,skip_existing_videos)
                        st.success(f"Downloaded {filename}.csv")
            

            if not st.session_state.results.empty:
                create_next_button('top')

                st.write(f"Displaying {len(st.session_state.results)} videos.")
                for _, row in st.session_state.results.iterrows():
                    st.write(row['name'])
                    st.video(row['contentUrl'])

                create_next_button('bottom')
            
            else:
                st.error("Do a search to display videos.")

            st.write("")

    elif section == 'Extract Frames':

        def download_video(url):
            video_data = requests.get(url).content
            with open('temp.mp4', 'wb') as handler:
                handler.write(video_data)

        def extract_frame(cap, frame_number):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            return frame

        def save_frame(frame, filename):
            cv2.imwrite(filename, frame)

        def extract_frames_from_videos(selected_file, frames_to_extract, skip_existing_videos, how_would_you_like_to_extract, batch_name, second_interval=0):
            df = pd.read_csv(selected_file)

            if not os.path.exists('output'):
                os.makedirs('output')

            file_is_empty = os.stat('output.csv').st_size == 0

            output_df = pd.read_csv('output.csv') if not file_is_empty else pd.DataFrame(columns=['video_id', 'uuid', 'image_0_location', 'image_0_description', 'image_1_location', 'image_1_description', 'caption'])

            with open('output.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file_is_empty:
                    writer.writerow(['video_id', 'uuid', 'image_0_location', 'image_0_description', 'image_1_location', 'image_1_description', 'caption'])

                for index, row in df.iterrows():

                    if skip_existing_videos is True:
                        if row['videoid'] in output_df['video_id'].values:
                            continue

                    video_url = row['contentUrl']

                    download_video(video_url)

                    cap = cv2.VideoCapture('temp.mp4')

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    # Choose frame extraction method based on user choice
                    if how_would_you_like_to_extract == 'At even intervals':
                        frame_numbers = [total_frames * i // frames_to_extract for i in range(frames_to_extract)]
                        frame_numbers[-1] = total_frames - 1

                    elif how_would_you_like_to_extract == 'Every X seconds':
                        frame_numbers = [i * second_interval * fps for i in range(frames_to_extract)]
                        frame_numbers = [frame for frame in frame_numbers if frame < total_frames]
                        if len(frame_numbers) < frames_to_extract:
                            frame_numbers.append(total_frames - 1)

                    frame_filenames = []
                    for i, frame_number in enumerate(frame_numbers):
                        frame = extract_frame(cap, frame_number)
                        frame_filename = f'output/{row["videoid"]}_{frame_number}.png'
                        save_frame(frame, frame_filename)
                        frame_filenames.append(frame_filename)

                    cap.release()

                    for i in range(len(frame_filenames) - 1):
                        writer.writerow([
                            row['videoid'],
                            uuid.uuid4(),
                            frame_filenames[i],
                            '',
                            frame_filenames[i + 1],
                            '',
                            '',
                            frame_numbers[i],
                            frame_numbers[i + 1],
                            'extracted',
                            row['name'],
                            '',
                            batch_name

                        ])

                os.remove('temp.mp4')
        

        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != 'output.csv' and f != 'results_10M_train.csv' and f != 'results_2M_train.csv']
        if csv_files:

            batch_name = st.text_input('Enter batch name:', value='batch_1')

            selected_file = st.selectbox('Which input file?', csv_files)

            how_would_you_like_to_extract = st.radio('How would you like to extract frames?', ['At even intervals', 'Every X seconds'])

            if how_would_you_like_to_extract == 'At even intervals':
                frames_to_extract = st.slider('How many frames would you like to extract from each video?', 1, 10, 3)
                
            elif how_would_you_like_to_extract == 'Every X seconds':
                second_interval = st.slider('How many seconds would you like between each frame extraction?', 1, 10, 3)
                frames_to_extract = st.slider('What\'s the maximum number of frames you\'d like to extract from each video?', 1, 10, 3)
                
            st.info(f"There are **{len(pd.read_csv(selected_file))} videos** in the {selected_file} file so this will result in **{(len(pd.read_csv(selected_file)) * (frames_to_extract -1))} frame pairs**.")          
            
            skip_existing_videos = st.checkbox('Skip videos that are already present:', value=True)

            if st.button('Extract Frame Pairs'):
                second_interval = second_interval if 'second_interval' in locals() else 0

                extract_frames_from_videos(selected_file, frames_to_extract, skip_existing_videos,how_would_you_like_to_extract,batch_name,second_interval)

        else:
            st.error("There are no input .csvs in your current directory - to create one, go to the Input Videos section.")


    elif section == 'Describe Frames':
            
        def describe_image_blip2(image_path, question, context):
            # Run the model
            output = replicate.run(
                "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
                input={
                    "image": open(image_path, "rb"),
                    "question": question,
                    "context": context
                }
            )
        
            return output
        
        def describe_image_with_questions(image_path, questions, context):
            results = []
            for question in questions:
                result = describe_image_blip2(image_path, question, context)
                results.append({
                    'question': question,
                    'answer': result
                })
            return results
        
        def process_row(row, questions):
            context = "This is an image of a person."
            # Get the image locations
            image_0_location = row['image_0_location']
            image_1_location = row['image_1_location']

            # Get the descriptions for both images
            description_0 = describe_image_with_questions(image_0_location, questions, context)
            description_1 = describe_image_with_questions(image_1_location, questions, context)

            # Convert the descriptions to JSON strings
            description_0 = json.dumps(description_0)
            description_1 = json.dumps(description_1)

            # Update the descriptions in the row
            row['image_0_description'] = description_0
            row['image_1_description'] = description_1

            return row
        
        def describe_images_and_update(df, rows_to_process, which_questions,filtering_question):
            
            for index, row in rows_to_process.iterrows():      
                row_contain_person = describe_image_blip2(row['image_0_location'], filtering_question, "")    
                if row_contain_person == "no":
                    row['status'] = 'rejected'
                    df.loc[index] = row
                else:
                    row = process_row(row, which_questions)
                    row['status'] = 'described'

                # get the latest version of the CSV file
                df = pd.read_csv('output.csv')
            
                # Update the row in the DataFrame
                df.loc[index] = row

                # Write the entire DataFrame to the CSV file
                df.to_csv('output.csv', index=False)

            st.success(f"Updated {len(rows_to_process)} rows in output.csv.")
            time.sleep(1)
            st.experimental_rerun()
        
        def describe_image_blip2(image_path, question, context):
                    
            if image_path.startswith('http'):
                image_input = image_path
            else:
                image_input = open(image_path, "rb")

            # Run the model
            output = replicate.run(
                "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
                input={
                    "image": image_input,
                    "question": question,
                    "context": context
                }
            )
        
            return output

        def update_questions_file(updated_questions, updated_filtering_question, questions, filtering_question):
            # Check if any question has been updated
            if updated_questions != questions or updated_filtering_question != filtering_question:
                # Update the YAML file
                with open('configs/questions.yaml', 'w') as file:
                    yaml.dump({
                        'questions': updated_questions,
                        'filtering_question': updated_filtering_question
                    }, file)

        

        with open('configs/questions.yaml', 'r') as file:
            data = yaml.safe_load(file)

        tab1, tab2 = st.tabs(['Queue Descriptions', 'Question Testing'])
        
        with tab1:
            questions = data.get('questions', [])
            filtering_question = data.get('filtering_question', '')

            updated_filtering_question = st.text_area("Filtering Question", value=filtering_question)

            updated_questions = []

            for question in questions:
                updated_question = st.text_input(f"Question {questions.index(question)+1}", value=question)
                updated_questions.append(updated_question)

            update_questions_file(updated_questions, updated_filtering_question, questions, filtering_question)
                    
            describe_all = st.checkbox('Describe All', value=True)

            if describe_all:
                num_to_describe = len(fetch_and_filter_data(['extracted']))        
            else:
                num_to_describe = st.number_input('How many would you like to describe?', min_value=1, max_value=1000, value=20)
            
            st.info(f"Describing **{num_to_describe} images** with **{len(questions)} questions** will cost **${(len(questions) * 2) * num_to_describe * (0.00115):.2f}**.")
                
            if st.button('Describe Images'):
                
                describe_images_and_update(pd.read_csv('output.csv'), fetch_and_filter_data(['extracted']).head(num_to_describe), questions, filtering_question)
                
            st.markdown("***")

            st.success('#### Previously described pairs:')
                
            for _, row in fetch_and_filter_data(['described']).tail(20).iterrows():
                # Create a new set of columns for each row
                cols = st.columns(2)
                # Display the images in the columns
                cols[0].image(row['image_0_location'])
                cols[1].image(row['image_1_location'])

                # Convert the descriptions to pandas DataFrames and display them as tables
                description_0_df = pd.DataFrame(json.loads(row['image_0_description']))
                description_1_df = pd.DataFrame(json.loads(row['image_1_description']))

                cols[0].dataframe(description_0_df)
                cols[1].dataframe(description_1_df)

                st.markdown("***")

        with tab2:
            st.write("")

            query_col_1, query_col_2 = st.columns(2)
            
            query1 = query_col_1.text_input("Enter first query:")
            query2 = query_col_2.text_input("Enter second query:")
            num_tests = st.number_input("Enter number of tests:", min_value=1, max_value=100)

            # Initialize SessionState for results if it doesn't exist
            if 'test_results' not in st.session_state:
                st.session_state.test_results = []

            # Get the last 20 described images
            described_images = fetch_and_filter_data(['described'])

            if st.button('Run Tests'):
                # Clear the current DataFrame
                st.session_state.test_results = []
                
                random_rows = described_images.sample(min(num_tests, len(described_images)))

                # Run the tests
                for i, row in random_rows.iterrows():
                    # Get the image path
                    image_path = row['image_0_location']
                            # Run the first query
                    result1 = describe_image_blip2(image_path, query1, "This is an image of a person.")
                    # Run the second query
                    result2 = describe_image_blip2(image_path, query2, "This is an image of a person.")

                    # Append the results to the list
                    st.session_state.test_results.append({
                        'test_num': i+1,
                        'image_path': image_path,
                        'query1': query1,
                        'query1_result': result1,
                        'query2': query2,
                        'query2_result': result2
                    })

                st.experimental_rerun()
            
            if st.session_state.test_results != []:
                for result in st.session_state.test_results:
                    # Create a new set of columns for each row
                    cols = st.columns(3)
                    # Display the image (replace 'image_path' with the actual image path)
                    cols[0].image(result['image_path'])
                            
                    cols[1].info(f"{result['query1']}")
                    cols[1].write(f"{result['query1_result']}")
                    
                    cols[2].info(f"{result['query2']}")
                    cols[2].write(f"{result['query2_result']}")

    elif section == 'Caption Pairs':

        def generate_captions(df, rows_to_process):
            system_message = "You are data tagging assistant who acts like a photographer. You give highly specific instructions, concise to your subjects in the correct JSON format based on image descriptions you're given."
            for index, row in rows_to_process.iterrows():
                print(f"Processing row {index}")
                image_0_description = row['image_0_description']
                image_1_description = row['image_1_description']
                original_caption = row['original_caption']
                prompt = (
                    "Below are descriptions of two images. Firstly, if these descriptions sound like they're the same, please just reply with 'the same' right away."
                    "If they sound different, here are some questions and answers about the images. They come from an AI image captioning so they may be inconsistent or inaccurate - keep that in mind and use common sense."
                    "Try to imagine what the images look like and imagine you're giving very basic instructions to a fashion model. "
                    "Give them succint, specific instructions to get from Description 1 to Description 2 and put the instructions as a json dictionary with three key pairs 'facial_expression', 'body_position','head_position' like this:"
                    "[{\"facial_expression\": \"open mouth and laugh\"},{\"body_position\": \"put right hand to face\"},{\"head_position\": \"no change\"}] \n\n"
                    "Only describe the change to make - not what the previous one was - and keep descriptions short and concise. If there's no change, just put 'no change' as the value. \n\n"
                    "Here's an overall caption for the video these were taken from: '{original_caption}'. These images were taken from within seconds one one another so changes will be minor.\n\n"
                    "And here are the descriptions of the two images: \n\n"
                    f"Description 1:\n\n{image_0_description}\n\n"
                    f"Description 2:\n\n{image_1_description}\n\n"
                    "As mentioned, please only reply with 'the same' if they're the same or json with the keys facial_expression, body_position and head_position if they're not:"
                )
                row['caption'] = prompt_gpt(prompt,system_message=system_message)
                # Update the row in the DataFrame
                if row['caption'] == 'the same' or row['caption'] == "'the same'":
                    row['status'] = 'rejected'                
                else:
                    row['status'] = 'captioned'
                df.loc[index] = row      
                
                df.to_csv('output.csv', index=False)

            st.success(f"Updated {len(rows_to_process)} rows in output.csv.")
            time.sleep(1)
            st.experimental_rerun()
        
        caption_all = st.checkbox('Caption All', value=True)

        if caption_all:
            num_to_caption = len(fetch_and_filter_data(['described']))        
        else:
            num_to_caption = st.number_input('How many would you like to caption?', min_value=1, max_value=1000, value=20)

        st.info(f"Captioning **{num_to_caption} pairs** will cost **${0.01 * num_to_caption:.2f}**.")
            
        if st.button('Generate Captions'):
            generate_captions(pd.read_csv('output.csv'), fetch_and_filter_data(['described']).head(num_to_caption))
            
        st.markdown("***")
        
        st.success('#### Previously captioned pairs:')

        st.markdown("***")

        for _, row in fetch_and_filter_data(['captioned']).tail(20).iterrows():
            # Create a new set of columns for each row
            cols = st.columns([1,1,1.5])
            # Display the images in the columns
            cols[0].image(row['image_0_location'])
            cols[1].image(row['image_1_location'])
            # Display the instructions in the third column
            if row['caption'].startswith('[') and row['caption'].endswith(']'):
                caption_dict = {k: v for d in json.loads(row['caption']) for k, v in d.items()}
                caption_df = pd.DataFrame(caption_dict, index=[0]).T
                cols[2].dataframe(caption_df)
            else:
                cols[2].error(f"Invalid JSON format in caption: {row['caption']}")
                            
            st.markdown("***")
        
    elif section == 'Send To Review Queue':

        def add_to_review_queue(df, rows_to_process):
            # Define the bucket name
            bucket_name = 'banodoco'

            # Iterate through the rows
            for index, row in rows_to_process.iterrows():
                video_id = row['video_id']
                image_0_frame_number = row['image_0_frame_number']
                image_1_frame_number = row['image_1_frame_number']

                # Define the file paths and S3 keys
                file_path_0 = row['image_0_location']
                file_path_1 = row['image_1_location']
                s3_key_0 = f'{video_id}_{image_0_frame_number}.png'
                s3_key_1 = f'{video_id}_{image_1_frame_number}.png'

                # Upload the files to S3 and get the URLs
                image_0_url = upload_image_to_s3(bucket_name, file_path_0, s3_key_0)
                image_1_url = upload_image_to_s3(bucket_name, file_path_1, s3_key_1)

                # Save the URLs in the dataframe
                df.loc[index, 'image_0_location'] = image_0_url
                df.loc[index, 'image_1_location'] = image_1_url

                # Set the row status
                df.loc[index, 'status'] = 'to_be_reviewed'

            # Save the dataframe to CSV
            df.to_csv('output.csv', index=False)
        
        st.info(f"There are **{len(fetch_and_filter_data(['captioned']))} pairs of images** at stage 'captioned'.")

        review_all = st.checkbox('Review All', value=True)

        if review_all:
            num_to_review = len(fetch_and_filter_data(['captioned']))        
        else:
            num_to_review = st.number_input('How many would you like to review?', min_value=1, max_value=1000, value=20)

        st.info(f"Reviewing **{num_to_review} images**.")
            
        if st.button('Review Images'):
            add_to_review_queue(pd.read_csv('output.csv'), fetch_and_filter_data(['captioned']).head(num_to_review))

        st.markdown("***")

        st.success('#### Current review queue:')

        # Count the total in the queue
        total_in_queue = len(fetch_and_filter_data(['to_be_reviewed']))

        st.info(f"There are **{total_in_queue} pairs of images** in the review queue.")
        
    elif section == 'Final Review':
    
        def manual_gpt_review():
            # Read the original DataFrame
            original_df = pd.read_csv('output.csv')

            # Get data with status 'Reviewed', rating = 1, and user_fix_suggestions != empty
            data = fetch_and_filter_data(statuses=['reviewed'], rating=1, filter_by_user_fix_suggestions=True)

            # Iterate over the rows
            for index, row in data.iterrows():
                system_message = "You are a data cleansing assistant who fulfils user requests and returns JSON - you only return JSON."
                
                # Construct the prompt
                prompt = f"What follows is JSON that contains a caption for a video: '{row['caption']}'\n\nUpon manual review, a reviewer suggested this change: '{row['user_fix_suggestions']}'\n\nCould you implement the changes that the user suggests to the input caption and return valid JSON with the fixes implemented? Don't copy exactly, try to update it with the changes in mind. Only return the JSON, nothing else."

                # Pass the prompt to gpt
                result = prompt_gpt(prompt, model="gpt-4", temperature=1, max_tokens=100, top_p=1, frequency_penalty=0, presence_penalty=0, system_message=system_message)

                # Save the result as a new caption in the original DataFrame
                original_df.at[index, 'caption'] = result

                # Update the 'user_rating' column in the original DataFrame
                user_rating = json.loads(original_df.at[index, 'user_rating'])
                user_rating[0]['rating'] = 2
                original_df.at[index, 'user_rating'] = json.dumps(user_rating)

            # Save the updated DataFrame to the csv file
            original_df.to_csv('output.csv', index=False)
        
        def trigger_inpainting(file_location, prompt):
            # Check if the image is a URL or a local file path
            image_input = file_location if file_location.startswith(('http://', 'https://')) else open(file_location, "rb")

            output_url = replicate.run(
                "subscriptions10x/sdxl-inpainting:733bba9bba10b10225a23aae8d62a6d9752f3e89471c2650ec61e50c8c69fb23",
                input={
                    "image": image_input,
                    "prompt": prompt,                
                    "mask_image": open("mask.png", "rb"),
                    "negative_keywords": "rectangle, blemish, watermark, logo, text, blotch"
                }
            )

            return output_url
        
        def process_accepted_images():
            # Step 1: Get all rows with status = "accepted" and watermark not removed
            accepted_df = fetch_and_filter_data(statuses=["approved"], filter_for_watermark_not_removed=True)

            # Step 2: Make a list of all unique image locations and corresponding prompts
            image_data = list(set((row['image_0_location'], row['original_caption']) for index, row in accepted_df.iterrows()) | 
                            set((row['image_1_location'], row['original_caption']) for index, row in accepted_df.iterrows()))

            # Step 3 and 4: Iterate through the list, trigger inpainting, and replace the original location with the new URL
            for image_location, prompt in image_data:
                # Get the UUIDs of the rows with the current image location
                row_uuids = accepted_df.loc[(accepted_df['image_0_location'] == image_location) | 
                                            (accepted_df['image_1_location'] == image_location), 'uuid'].values

                # Trigger inpainting
                new_url = trigger_inpainting(image_location, prompt)

                st.image(new_url)

                # Download the image
                response = requests.get(new_url[0])  # Access the URL by indexing the list
                img = Image.open(BytesIO(response.content))

                # Resize the image
                width = int((512 / img.height) * img.width)
                img = img.resize((width, 512))

                # Save the image to a temporary file
                img.save('temp.png')

                # Upload the image to S3
                upload_image_to_s3('banodoco', 'temp.png', image_location.split('/')[-1],overwrite=True)

                st.image(image_location)

                for uuid in row_uuids:
                    update_field_and_save(uuid, 'watermark_removed', True, csv_file='output.csv')
        
        tab1, tab2, tab3 = st.tabs(['Review Queue', 'GPT Enrichment', 'Remove Watermarks'])
        
        with tab1:
            
            reviewed_data = fetch_and_filter_data(statuses=['reviewed'], rating=2)
            
            if not reviewed_data.empty:            
                display_rows_with_editing('reviewed')
            else:
                st.success("There are no rows awaiting review.")

        with tab2:

            st.success(f"There are {len(fetch_and_filter_data(statuses=['reviewed'], rating=1, filter_by_user_fix_suggestions=True))} rows awaiting GPT enrichment.")
            
            if st.button('Run GPT Enrichment'):
                manual_gpt_review()
                st.experimental_rerun()

        with tab3:
            
            st.success(f"There are {len(fetch_and_filter_data(statuses=['approved'], rating=2, filter_for_watermark_not_removed=True))} rows awaiting watermark removal.")
            
            if st.button('Run Watermark Removal'):
                process_accepted_images()
            
    elif section == 'Approved Pairs':

        def prep_data_for_ip2p():
            # Create the data directory if it doesn't exist
            if not os.path.exists('data'):
                os.makedirs('data')

            # Fetch the DataFrame
            df = fetch_and_filter_data(statuses=['approved'])

            counter = 0
            for _, row in df.iterrows():
                dir_name = f'data/image_{counter}'
                counter += 1
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                                # Create prompt.json
                with open(f'{dir_name}/prompt.json', 'w') as file:
                    # Parse the JSON string in the 'caption' column
                    caption_data = json.loads(row['caption'])

                    # Extract the data from all pairs and join them with commas
                    output_caption = ', '.join([str(value) for pair in caption_data for value in pair.values() if value])

                    # Insert the extracted data into the JSON file
                    json.dump({
                        'caption': row['original_caption'],
                        'output': output_caption,
                        'edit': output_caption
                    }, file)
                # Generate a random 8-digit number
                seed = random.randint(10000000, 99999999)

                # Create metadata.jsonl
                with open(f'{dir_name}/metadata.jsonl', 'w') as file:
                    json.dump({
                        'seed': seed,
                        'p2p_threshold': 0.2,
                        'cfg_scale': 7.5,
                        'clip_sim_0': 0.8,
                        'clip_sim_1': 0.85,
                        'clip_sim_dir': 0.3,
                        'clip_sim_image': 0.7
                    }, file)

                # Download the images and save them as {seed}_0.png and {seed}_1.png
                for j, image_url in enumerate([row['image_0_location'], row['image_1_location']]):
                    response = requests.get(image_url)
                    with open(f'{dir_name}/{seed}_{j}.png', 'wb') as file:
                        file.write(response.content)

                    # Open the image and crop it to a 512x512 square at the center
                    img = Image.open(f'{dir_name}/{seed}_{j}.png')
                    width, height = img.size
                    left = (width - 512)/2
                    top = (height - 512)/2
                    right = (width + 512)/2
                    bottom = (height + 512)/2
                    img = img.crop((left, top, right, bottom))

                    # Save the cropped image
                    img.save(f'{dir_name}/{seed}_{j}.png')

        with st.sidebar:

            st.subheader("Data Preparation")

            st.info("Here, you can prepare data for IP2P or the motion module.")
            
            data_type = st.selectbox('Which type of data?', ['ip2p','motion_module'])

            if data_type == 'motion_module':
                grab_videos = st.checkbox('Grab full videos', value=False)

            if st.button('Prepare Data'):
                if data_type == 'ip2p':
                    prep_data_for_ip2p()
                    st.experimental_rerun()
                elif data_type == 'motion_module':
                    st.error("This feature is not yet implemented.")


            
        display_rows_with_editing('approved')


if __name__ == "__main__":
    main()