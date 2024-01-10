# imports for processing and utilities
import os 
import argparse 
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count, current_process

# imports to filter dataset
import pandas as pd
from jinja2 import Template

# imports to generate stories
import openai 
from config import OPENAI_API_KEY

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

# Define file paths
input_path = 'datasets/hcV3-stories.csv'
output_path = 'datasets/hcV3-imagined-stories.csv'

# Define prompt templates
system_prompt_template = """
Given a short prompt summary, write an imagined journal entry about an event. Writing instructions:
- The story must correspond to the summary. 
- Pretend the event happened to you, but do not write about something that actually happened to you.
- Write using first person perspective.  
- Use the timeline of when the event happened (e.g., "3 weeks ago", "6 months ago").

Now, write the journal entry below.:
- **Reminder** Please make sure to write a story that corresponds to the summary written above. 
- Don't write a story about something that actually happened to you.
- Story must be 15-25 sentences and 600-3000 characters including spaces.  
- Do not start off with any salutations (e.g., "Dear Diary") or dates (e.g., 3/1/2021").
- Do not include the specific date or time of the event in the story.
"""

system_prompt_template_few_shot = """
Given a short prompt summary, write an imagined journal entry about an event. Writing instructions:
- The story must correspond to the summary. 
- Pretend the event happened to you, but do not write about something that actually happened to you.
- Write using first person perspective.  
- Use the timeline of when the event happened (e.g., "3 weeks ago", "6 months ago").

Now, write the journal entry below.:
- **Reminder** Please make sure to write a story that corresponds to the summary written above. 
- Don't write a story about something that actually happened to you.
- Story must be 15-25 sentences and 600-3000 characters including spaces.  
- Do not start off with any salutations (e.g., "Dear Diary") or dates (e.g., 3/1/2021").
- Do not include the specific date or time of the event in the story.

Human Example 1: It was a beautiful Saturday morning. My family members and i decided to go to the beach. We got to the beach and got carried away by the fun. We forgot to apply the sunscreen and that caused a damage to our skin. We all got sunburned and look black. This unique burn on our skin made us look alike. After leaving the beach and got home. We decided to have a family reunion and this made us all have a burn to identify ourselves. We really love what the sunburn did to our skin. It made us love ourselves more. We took a family picture and placed it in the living room. Each member also had a copy of the pictures. We planned another visit to the beach. We applied sunscreen this time. We are ever happy after.

Human Example 2: Our end of summer family reunion came around and we decided to have it hosted at a restaurant.  It was beautiful, as we had it in a tent overlooking the lake.  My brother had maybe one too many drinks that day.  A waiter came over and asked if he would like a water.  My brother looked at him and told him that he knows when it's okay to quit drinking.  I know he didn't mean anything by it, but the waiter was horrified.  He thought he insulted my brother, when he was simply offering water to everyone on a tray.  My brother didn't mean to be mean.  Several people had already come up to him offering water, so he simply thought it was  hint to quit drinking.  It kind of set the tone for me for the rest of the day though.  I mean, it's been three weeks and I still get embarrassed thinking about that one moment.  One that the waiter has probably forgotten all about.  I won't go back to that restaurant though for a bit.  I wouldn't risk getting the same waiter just in case he would remember.  I don't want anyone to spit in my food if he still holds a grudge.

Human Example 3: I returned to school after not being there for a while. I studied real hard at the college. I met some new friends. We always went partying on the weekend. I always got blasted and puke a lot of the times. It was worth it. I studied to become an engineer. It was tough. The material was very challenging. I had a hard time getting the instructions right for the problems. I studied hard and I graduated. I had a fun time at the graduation. There were many people in the crowd. I waved to all of them. I threw my cap in the air and got my diploma. I hope to get a good job in the coming months.

Mimic the style of human stories (as shown in the three examples above) as much as possible - sentence length, difficulty of sentences, word choice, frequency of different parts of speeches, and more. You may decide to write short stories, use colloquial phrases, and have improper grammar. You may also choose to have short stories, as long as it of similar length to the examples above and meet the required guidelines.
"""

user_prompt_template = """
Summary (from {{ time_since_event }} days ago): {{ summary }}

Generated Story:
"""

def filter_and_save_dataset(input_path, output_path):
    '''This function filters the dataset to only include imagined stories 
    and saves the filtered dataset to a new CSV file.'''
    
    # Load data
    data = pd.read_csv(input_path)
    
    # Filter for 'imagined' stories
    imagined_stories = data[data['memType'] == 'imagined']
    
    # Keep only relevant columns
    kept_columns = ['AssignmentId', 'story', 'summary', 'timeSinceEvent']
    imagined_stories = imagined_stories[kept_columns]
    
    # Save filtered data to new CSV
    imagined_stories.to_csv(output_path, index=False)

    print(f"Saved {imagined_stories.shape[0]} rows to {output_path}")

def generate_story(summary, time_since_event, system_prompt_template=system_prompt_template, user_prompt_template=user_prompt_template):
    '''This function generates a story based on a summary and time since event.'''

    # Render prompt templates
    system_prompt = Template(system_prompt_template_few_shot).render()
    user_prompt = Template(user_prompt_template).render(
        summary=summary,
        time_since_event=time_since_event
    )

    # Generate story
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response['choices'][0]['message']['content']

def generate_stories_for_row(row):
    '''This function generates a story for a single row in the dataset.'''

    # Retrieve details from the row
    assignment_id = row['AssignmentId']
    summary = row['summary']
    time_since_event = row['timeSinceEvent']

    # Print the processing status and the core it's being processed on
    print(f"Processing {assignment_id} on {current_process().name}")

    # Generate the story using the summary and time since event
    generated_story = generate_story(summary, time_since_event)

    return assignment_id, generated_story

def generate_all_stories(data):
    '''This function generates stories for all rows in the dataset.'''
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPU cores for processing.")
    
    # Check if temp file exists and load it if it does
    temp_file_path = 'datasets/temp_generated_stories.csv'
    if os.path.exists(temp_file_path):
        temp_df = pd.read_csv(temp_file_path)
        completed_assignment_ids = set(temp_df['AssignmentId'])
        # Filter the data to only include rows not already processed
        data = data[~data['AssignmentId'].isin(completed_assignment_ids)]
        results = temp_df.values.tolist()
    else:
        results = []

    # Update progress bar total based on remaining data
    pbar = tqdm(total=len(data), desc="Progress")
    
    # Save periodically after processing each row
    with Pool(num_cpus) as pool:
        for assignment_id, generated_story in pool.imap_unordered(generate_stories_for_row, [row for _, row in data.iterrows()]):
            results.append((assignment_id, generated_story))
            
            # Update the progress bar
            pbar.update(1)
            
            # Save results to CSV every 20 rows processed
            if len(results) % 20 == 0:
                temp_df = pd.DataFrame(results, columns=['AssignmentId', 'generated_story'])
                temp_df.to_csv(temp_file_path, index=False)
    
    # Close the progress bar
    pbar.close()
    
    # Save the final results to a new dataframe and return it
    return pd.DataFrame(results, columns=['AssignmentId', 'generated_story'])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate stories based on summaries")
    parser.add_argument("--input_path", default="datasets/hcV3-stories.csv", help="Path to input CSV file")
    parser.add_argument("--output_path", default="datasets/hcV3-imagined-stories-with-generated-few-shot.csv", help="Path to output CSV file")
    return parser.parse_args()

def main(input_path, output_path):
    # Filter the dataset and save imagined stories
    imagined_stories_path = 'datasets/hcV3-imagined-stories.csv'
    filter_and_save_dataset(input_path, imagined_stories_path)
    
    # Load imagined stories
    imagined_stories = pd.read_csv(imagined_stories_path)

    # Generate stories for all rows in the dataframe
    stories_df = generate_all_stories(imagined_stories)

    # Merge the original dataframe with the generated stories on AssignmentId
    merged_df = imagined_stories.merge(stories_df, on="AssignmentId")

    # Save the updated dataframe with the generated stories to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Saved {merged_df.shape[0]} rows to {output_path}")

def few_shot_main(output_path):
    
    # Load imagined stories
    imagined_stories = pd.read_csv('datasets/hcV3-imagined-stories.csv')

    # Generate stories for all rows in the dataframe
    stories_df = generate_all_stories(imagined_stories)

    # Merge the original dataframe with the generated stories on AssignmentId
    merged_df = imagined_stories.merge(stories_df, on="AssignmentId")

    # Save the updated dataframe with the generated stories to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Saved {merged_df.shape[0]} rows to {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    few_shot_main(args.output_path)