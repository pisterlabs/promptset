import csv
import openai

openai.api_key = ''

common = '''You have to classify the given song to their most likely label according to the mood of the song. For each song I will provide song title, song artist and song lyrics. For interpreting the mood a song you either get it from your internal information or use the provided lyrics. The labels are 'romantic/love/passion/devotion','loss/sad/heartbreak/angst/protest','happy/celebration/party/dance','motivating/inspirational/uplifting/confidence/nostalgia'. You just have to provide the most probable answer in string format (single label), the answer shouldn't contain anything else (very important). You should perform the given task very accurately.'''

def create_question(artist, title, lyrics):
    question = f"Artist: {artist} ; Song-Title: {title} ; Lyrics: {lyrics}"
    return question

def ask_chatgpt(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        tokens_used = response.usage['total_tokens']
        return response.choices[-1].message['content'], None, tokens_used
    except Exception as e:
        print(f"API call failed. Error: {str(e)}")
        return None, str(e), 0

def process_csv(input_file, output_file):
    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['response', 'error']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        messages = [{"role": "system", "content": common}]
        count = 0
        total_tokens = 0

        for row in reader:
            count += 1
            if count>=214:
                question = create_question(row['artist'], row['title'], row['lyrics'])
                messages.append({"role": "user", "content": question})
                response, error, tokens = ask_chatgpt(messages)
                total_tokens += tokens
                
                row['response'] = response if response else "Error occurred"
                row['error'] = error if error else ""
                writer.writerow(row)
                messages.pop()  # Keep context relevant

                if count % 10 == 0:
                    print(f"Processed: {count} rows, Total tokens used: {total_tokens}")

            # Break after 10 for testing, remove this in production
            # if count == 10:
            #     break

process_csv('lyrics_final.csv', 'lyrics_gpt2.csv')