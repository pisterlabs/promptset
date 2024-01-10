from Access_to_models import get_embedding_gensim_list
from Access_to_models import calc_cosine_similarity
import pickle
from nltk.corpus import stopwords
import os
import time
import nltk
import openai
import pandas as pd
import wordfreq
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from Access_to_models import chatgpt

prompt_creative_short_story = 'Act as a very creative person. Write a short story about topic.'

prompt_uncreative_short_story = 'Act as a very uncreative person. Write a short story about topic.'

prompt_creative_poem = 'Act as a very creative person. Write a poem about topic.'

prompt_uncreative_poem = 'Act as a very uncreative person. Write a poem about topic.'

prompt_creative_article = 'Act as a very creative person. Write an article about topic.'

prompt_uncreative_article = 'Act as a very uncreative person. Write an article about topic.'


def save_storys(prompt, filename_raw, num_responses):
    print(prompt)
    try:
        with open(filename_raw, 'r') as file:
            lines = len(file.readlines())
            print('curr num of lines', lines)
    except FileNotFoundError:
        lines = 0
    if lines < num_responses:
        try:
            response = chatgpt(prompt)
            response = response.strip()
            response = response.replace('\n', ' ')
            with open(filename_raw, 'a+') as f:
                # print(i)
                if lines >= 1:
                    # Write the string to the file
                    f.write('\n' + response)
                else:
                    f.write(response)
            save_storys(prompt, filename_raw, num_responses)
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(20)
            save_storys(prompt, filename_raw, num_responses)
            # response = chatgpt(prompt, temperature)
    else:
        return None


topics = ['how AI might change the world', 'climate change', 'social media']


# for topic in topics:
#     # save_storys(prompt_creative_poem.replace('topic', topic), 'poems/creative_poems_' + topic + '.txt', 10)
#     save_storys(prompt_uncreative_poem.replace('topic', topic), 'poems/uncreative_poems_' + topic + '.txt', 10)


def get_avg_distance_text(text, filename, file_name_list):
    distances = []
    words = nltk.word_tokenize(text)
    # print(words)
    # Filter out non-words, such as punctuation symbols and numbers
    # filtered_words = [word for word in words if word.isalpha()]
    filtered_words = [word for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    # print(filtered_words)
    # print(filtered_words)
    if os.path.exists(file_name_list):
        # File exists, read the list from the file
        with open(file_name_list, 'rb') as file:
            vectors = pickle.load(file)
        # print("List loaded from the existing file:")
        # print(vectors)
    else:
        vectors = get_embedding_gensim_list(filtered_words)
        with open(file_name_list, 'wb') as file:
            pickle.dump(vectors, file)
        # print("New list created and saved to a file.")
    frequencies = [wordfreq.zipf_frequency(w, 'en', wordlist='best', minimum=0.0) for w in filtered_words]
    lower_words = [w.lower() for w in filtered_words]
    unique_words = set(lower_words)
    for i in range(0, len(vectors), 2):
        if (i + 1) < len(vectors):
            # print(vectors[i],vectors[i + 1])
            distances.append(1 - calc_cosine_similarity(vectors[i], vectors[i + 1]))
    # print(distances)
    avg_dist = sum(distances) / len(distances)
    avg_freq = sum(frequencies) / len(frequencies)
    print("Average Frequency:", avg_freq, "Average Distance:", avg_dist, "Unique Words:", len(unique_words), 'Length',
          len(text))
    return avg_dist, avg_freq, len(text), len(unique_words)


def plot_act_as(df_creative, df_uncreative, topic):
    fontsize = 19
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 18), gridspec_kw={'top': 0.95})
    bar_width = 0.4
    categories = ['Stories', 'Articles', 'Poems']
    # Plot creative data in the left column
    for count, (index, row) in enumerate(df_creative.iterrows(), start=1):
        if count - 1 == 0:
            ax = axes[(0, 0)]
        elif count - 1 == 1:
            ax = axes[(0, 1)]
        elif count - 1 == 2:
            ax = axes[(1, 0)]
        elif count - 1 == 3:
            ax = axes[(1, 1)]
        x_positions = [x - 0.2 for x in range(3)]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories)
        ax.bar(x_positions, row, width=bar_width, color='red', edgecolor='black', label='creative')
        ax.set_xlabel('Categories', fontsize=fontsize)  # Increase font size for x-axis label
        ax.set_ylabel(f'Mean {index}', fontsize=fontsize)  # Increase font size for y-axis label
        ax.set_title(f'{index} Mean Values', fontsize=fontsize)
        # Plot uncreative data in the right column
    for count, (index, row) in enumerate(df_uncreative.iterrows(), start=1):
        if count - 1 == 0:
            ax = axes[(0, 0)]
        elif count - 1 == 1:
            ax = axes[(0, 1)]
        elif count - 1 == 2:
            ax = axes[(1, 0)]
        elif count - 1 == 3:
            ax = axes[(1, 1)]
        x_positions = [x + 0.2 for x in range(3)]
        # x_positions = [x + bar_width for x in categories]
        ax.bar(x_positions, row, width=bar_width, color='green', edgecolor='black', label='uncreative')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, fontsize=fontsize)
        ax.set_xlabel('Categories', fontsize=fontsize)  # Increase font size for x-axis label
        ax.set_ylabel(f'Mean {index}', fontsize=fontsize)  # Increase font size for y-axis label
        ax.set_title(f'Comparison {index}', fontsize=fontsize)  # Increase font size for title
    # axes[0, 1].set_ylim(4, 5)  # Adjust the limits as needed
    for ax in axes.flatten():
        # ax.legend()
        ax.legend().set_visible(False)
    # plt.suptitle('Texts about ' + topic, fontsize=22, y=0.99)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    dpi = 450  # Adjust the DPI value as needed
    # plt.savefig('Creative and uncreative texts about ' + topic + '.png', dpi=dpi, bbox_inches='tight')
    # Show the figure with subplots
    plt.show()


def ttest(data1, data2):
    t_statistic, p_value = ttest_ind(data1, data2)
    # Print the results
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)
    print(p_value < 0.05)
    # if p_value < 0.05:
    #     print("Result: Significant")
    # else:
    #     print("Result: Not Significant")


def main(filename_texts, filename_distances):
    print('Creative')
    data_creative = []
    data_uncreative = []
    with open(filename_texts, 'r') as file:
        for i, line in enumerate(file):
            curr_ff, curr_freq, curr_len, curr_uw = get_avg_distance_text(line, '',
                                                                          filename_distances + str(i) + '.pickle')
            data_creative.append([curr_ff, curr_freq, curr_len, curr_uw])
    filename_texts = filename_texts.replace('creative', 'uncreative')
    filename_distances = filename_distances.replace('creative', 'uncreative')
    print('Uncreative')
    with open(filename_texts, 'r') as file:
        for i, line in enumerate(file):
            curr_ff, curr_freq, curr_len, curr_uw = get_avg_distance_text(line, '',
                                                                          filename_distances + str(i) + '.pickle')
            data_uncreative.append([curr_ff, curr_freq, curr_len, curr_uw])
    return data_creative, data_uncreative


topics = ['how AI might change the world', 'climate change', 'social media']
for topic in topics:
    print('Stories about', topic)
    data_creative_stories, data_uncreative_stories = main('stories/creative_story_' + topic + '.txt',
                                                          'stories/creative_distances_' + topic + 'story')
    # print(data_creative_stories[0])
    df_creative_stories = pd.DataFrame(data=data_creative_stories,
                                       columns=['Distance between Words', 'Word Frequencies', 'Length',
                                                'Number of Unique Words'])
    df_uncreative_stories = pd.DataFrame(data=data_uncreative_stories,
                                         columns=['Distance between Words', 'Word Frequencies', 'Length',
                                                  'Number of Unique Words'])
    ttest(df_creative_stories, df_uncreative_stories)
    df_creative_stories = df_creative_stories.mean()
    df_uncreative_stories = df_uncreative_stories.mean()
    # print(df_creative_stories)
    print('Articles about', topic)
    data_creative_articles, data_uncreative_articles = main('articles/creative_articles_' + topic + '.txt',
                                                            'articles/creative_distances_' + topic + 'article')
    df_creative_articles = pd.DataFrame(data=data_creative_articles,
                                        columns=['Distance between Words', 'Word Frequencies', 'Length',
                                                 'Number of Unique Words'])
    df_uncreative_articles = pd.DataFrame(data=data_uncreative_articles,
                                          columns=['Distance between Words', 'Word Frequencies', 'Length',
                                                   'Number of Unique Words'])
    ttest(df_creative_articles, df_uncreative_articles)
    df_creative_articles = df_creative_articles.mean()
    df_uncreative_articles = df_uncreative_articles.mean()
    print('Poems about', topic)
    data_creative_poems, data_uncreative_poems = main('poems/creative_poems_' + topic + '.txt',
                                                      'poems/creative_distances_' + topic + 'poem')
    df_creative_poems = pd.DataFrame(data=data_creative_poems,
                                     columns=['Distance between Words', 'Word Frequencies', 'Length',
                                              'Number of Unique Words'])
    df_uncreative_poems = pd.DataFrame(data=data_uncreative_poems,
                                       columns=['Distance between Words', 'Word Frequencies', 'Length',
                                                'Number of Unique Words'])
    ttest(df_creative_poems, df_uncreative_poems)
    df_creative_poems = df_creative_poems.mean()
    df_uncreative_poems = df_uncreative_poems.mean()
    df_total_creative = pd.concat([df_creative_stories, df_creative_articles, df_creative_poems], axis=1)
    df_total_creative.columns = ['Stories', 'Articles', 'Poems']
    df_total_uncreative = pd.concat([df_uncreative_stories, df_uncreative_articles, df_uncreative_poems], axis=1)
    df_total_uncreative.columns = ['Stories', 'Articles', 'Poems']
    # print(df_total)
    plot_act_as(df_total_creative, df_total_uncreative, topic)
