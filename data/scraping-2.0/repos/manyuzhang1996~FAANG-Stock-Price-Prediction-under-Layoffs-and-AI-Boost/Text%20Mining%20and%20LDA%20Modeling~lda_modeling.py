import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import CoherenceModel
from wordcloud import WordCloud

def calculate_coherence(corpus, dictionary, texts, num_topics):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score

def lda_modeling_experiment(df):
    texts = df['title'].tolist()
    texts = [word_tokenize(text) for text in texts]
    texts = [[word for word in text if word.isalpha()] for text in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    seed = 42
    num_topics_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    coherence_scores = [calculate_coherence(corpus, dictionary, texts, num_topics) for num_topics in num_topics_list]
    optimal_topics = num_topics_list[coherence_scores.index(max(coherence_scores))]

    # Visualize the results
    plt.plot(num_topics_list, coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Optimal Number of Topics Selection')
    plt.show()
    print(f"The optimal number of topics is {optimal_topics}")
    return corpus, optimal_topics, dictionary

def optimal_lda_modeling(corpus, optimal_topics, dictionary):
    lda_model = models.LdaModel(corpus, num_topics=optimal_topics, id2word=dictionary, passes=15)
    for topic_id, topic_words in lda_model.print_topics():
        print(f'\nTopic {topic_id + 1}: {topic_words}')
    return lda_model

def visualize_lda_topics(lda_model, num_topics):
    topics = [lda_model.show_topic(topic_id) for topic_id in range(num_topics)]
    for i, topic_words in enumerate(topics):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(topic_words))
        plt.figure(figsize=(6, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i + 1} Word Cloud')
        plt.axis('off')
        plt.show()