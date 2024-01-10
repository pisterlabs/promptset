import openai
import json
from transformers import pipeline
import newspaper

openai.api_key = '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'


def convertArticleText(link):
    article = newspaper.Article(url=link)

    article.download()

    article.parse()

    article_text = article.text

    return(article_text)


def extract_features(emotions_dict):
    score_message = 'The scores for the intensity of each emotion\'s level, separated by \';\' are: '
    for i, emotion in enumerate(emotions_dict):
        score_message += f'{emotion}: {emotions_dict[emotion]}'
        if i == len(emotions_dict) - 1:
            score_message += '.'
        else:
            score_message += '; '
    return(score_message)
    

def generate_sentence_with_emotion(original_sentence, new_dict):
    neutrality_value=new_dict['neutral']
    string_neut=str(neutrality_value)

    prompt = f'''The following in quotes is a news article: \'{original_sentence}\'. 
    
    The intensity of each emotion present in the article above was scaled on a scale from 0-1, with 0 being a low-scoring emotion and 1 being a high-scoring emotion.  the following passage scored a'+ string_neut +' on the neutral scale, can you translate this text to make it have a neutral value of .7 which would make the article much more neutral
    '''

    response = openai.Completion.create(
        engine="text-davinci-002",  # Choose the appropriate engine
        prompt=prompt,
        temperature=0.7,
        max_tokens=300
    )

    rewritten_sentence = response.choices[0].text.strip()
    return rewritten_sentence


def robertaClassifier(passage):
    # HuggingFace Roberta Classifier
    classifier = pipeline(task="sentiment-analysis", model="SamLowe/roberta-base-go_emotions", top_k=None)

    # generate emotion scores
    model_outputs = classifier(passage, truncation=True, padding=True, max_length=512)

    new_dict = {}
    for item in model_outputs[0]:
        new_dict[item['label']] = item['score']
    # new_dict['neutral'] = 0.7

    print(new_dict['neutral'])
    return new_dict


# temporary link
# passage = convertArticleText("https://www.cnn.com/2023/10/20/opinions/israel-gaza-biden-ukraine-russia-mark/index.html")

# extract emotion scores and generate score message
# score_message = extract_features(robertaClassifier(passage))
# print(score_message)
# new_article = generate_sentence_with_emotion(passage, new_dict)

# print(new_article)
