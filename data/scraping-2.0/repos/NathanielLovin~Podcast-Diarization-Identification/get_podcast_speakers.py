import openai
import csv
from time import sleep

def get_podcast_speakers_no_hints(file, model, speakers):
    print(file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        rows = [rows[i][3:] for i in range(len(rows))]
        rows = rows[1:]
        # Make a list of all the speakers as Don't Know
        current_speaker_guess = ["SPEAKER " + str(i) + ": Don't Know" for i in range(1, speakers+1)]
        # turn the list into a string
        current_speaker_guess = '\n'.join(current_speaker_guess)
        current = 0
        while "Don't Know" in current_speaker_guess and current < len(rows):
            next_rows = rows[current:current+15]
            next_rows = [next_rows[i][0] + ": " + next_rows[i][1] for i in range(len(next_rows))]
            next_rows = '\n'.join(next_rows)
            messages = [{"role": "system", "content": "You are given a podcast transcript broken down by speaker and need to identify the names of the speakers. If you don't know, identify the speaker as Don't Know. Only respond with the names of the speakers."},
                        {"role": "user", "content": "Current Speaker Guess: " + current_speaker_guess + "\n\nNext Lines of Transcript:\n" + next_rows}]
            try:
                result = openai.ChatCompletion.create(model=model, messages=messages)
            except:
                while True:
                    try:
                        sleep(10)
                        result = openai.ChatCompletion.create(model=model, messages=messages)
                        break
                    except:
                        pass
            current_speaker_guess = result.choices[0].message.content
            current += 15
            # print(current_speaker_guess)
            sleep(1)
        return current_speaker_guess
    
def get_podcast_speakers_with_hints(file, model, speakers):
    print(file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        rows = [rows[i][3:] for i in range(len(rows))]
        rows = rows[1:]
        # Make a list of all the speakers as Don't Know
        current_speaker_guess = ["SPEAKER " + str(i) + ": Don't Know" for i in range(1, len(speakers)+1)]
        # turn the list into a string
        current_speaker_guess = '\n'.join(current_speaker_guess)
        current = 0
        while "Don't Know" in current_speaker_guess and current < len(rows):
            next_rows = rows[current:current+15]
            next_rows = [next_rows[i][0] + ": " + next_rows[i][1] for i in range(len(next_rows))]
            next_rows = '\n'.join(next_rows)
            messages = [{"role": "system", "content": "You are given a podcast transcript broken down by speaker and need to identify the names of the speakers. If you don't know, identify the speaker as Don't Know. You are given the names of speakers but they are not in order. Only respond with the names of the speakers."},
                        {"role": "user", "content": "Possible Speakers: " + ', '.join(speakers) + "\nCurrent Speaker Guess: " + current_speaker_guess + "\n\nNext Lines of Transcript:\n" + next_rows}]
            # If api call fails, try again
            try:
                result = openai.ChatCompletion.create(model=model, messages=messages)
            except Exception as e:
                print(e)
                while True:
                    try:
                        sleep(10)
                        result = openai.ChatCompletion.create(model=model, messages=messages)
                        break
                    except Exception as e:
                        print(e)
                        pass
            current_speaker_guess = result.choices[0].message.content
            current += 15
            sleep(1)
            # print(current_speaker_guess)
        return current_speaker_guess
    
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-20.csv', 'gpt-3.5-turbo', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-27.csv', 'gpt-3.5-turbo', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-33.csv', 'gpt-3.5-turbo', 5))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-45.csv', 'gpt-3.5-turbo', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-53.csv', 'gpt-3.5-turbo', 4))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-3.5-turbo', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-27.csv', 'gpt-3.5-turbo', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-33.csv', 'gpt-3.5-turbo', ['Slime', 'Aiden', 'Ludwig', 'Nick', 'Asa Butterfield']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-45.csv', 'gpt-3.5-turbo', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-53.csv', 'gpt-3.5-turbo', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-5.csv', 'gpt-3.5-turbo', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-70.csv', 'gpt-3.5-turbo', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-102.csv', 'gpt-3.5-turbo', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-117.csv', 'gpt-3.5-turbo', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-121.csv', 'gpt-3.5-turbo', 2))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-5.csv', 'gpt-3.5-turbo', ['Hank', 'Maureen']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-70.csv', 'gpt-3.5-turbo', ['John', 'Ashley']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-102.csv', 'gpt-3.5-turbo', ['Hank', 'John']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-117.csv', 'gpt-3.5-turbo', ['Hank', 'Katherine']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-121.csv', 'gpt-3.5-turbo', ['Hank', 'John']))

# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-20.csv', 'gpt-4', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-27.csv', 'gpt-4', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-33.csv', 'gpt-4', 5))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-45.csv', 'gpt-4', 4))
# print("Final Result: " + get_podcast_speakers_no_hints('yard-ep-53.csv', 'gpt-4', 4))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-27.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-33.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick', 'Asa Butterfield']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-45.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-53.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-5.csv', 'gpt-4', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-70.csv', 'gpt-4', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-102.csv', 'gpt-4', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-117.csv', 'gpt-4', 2))
# print("Final Result: " + get_podcast_speakers_no_hints('dear-hank-and-john-ep-121.csv', 'gpt-4', 2))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-5.csv', 'gpt-4', ['Hank', 'Maureen']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-70.csv', 'gpt-4', ['John', 'Ashley']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-102.csv', 'gpt-4', ['Hank', 'John']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-117.csv', 'gpt-4', ['Hank', 'Katherine']))
# print("Final Result: " + get_podcast_speakers_with_hints('dear-hank-and-john-ep-121.csv', 'gpt-4', ['Hank', 'John']))

print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
print("Final Result: " + get_podcast_speakers_with_hints('yard-ep-20.csv', 'gpt-4', ['Slime', 'Aiden', 'Ludwig', 'Nick']))
