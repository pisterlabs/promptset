import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import CoherenceModel

def get_top_words_for_topics(lda_model, num_words=5):
    top_words_for_topics = []
    for topic_id in range(lda_model.num_topics):
        top_words = [word for word, prob in lda_model.show_topic(topic_id, topn=num_words)]
        top_words_for_topics.append(top_words)
    return top_words_for_topics

def main():
    # Load data from a CSV file
    df = pd.read_csv('wb_neutral_df.csv')

    # Assuming your CSV file has a 'text' column containing the text data
    documents = df['0'].tolist()

    # Tokenize the input documents (split by space)
    tokenized_documents = [document.split() for document in documents[1:]]

    # Create a Gensim dictionary
    dictionary = corpora.Dictionary(tokenized_documents)

    # Create a Gensim corpus (bag of words)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    # Perform LDA topic modeling
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=40)

    # Print the topics
    topics = lda_model.print_topics()
    for topic in topics:
        print(topic)

    # Calculate and print coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score (LDA): {coherence_lda:.4f}')

     # Get the top words for each topic
    top_words_for_topics = get_top_words_for_topics(lda_model, num_words=5)

    # Convert top words into a DataFrame
    top_words_df = pd.DataFrame(top_words_for_topics, columns=[f'Top_Word_{i}' for i in range(1, 6)])

    # Print the top words for each topic
    print(top_words_df)

if __name__ == '__main__':
    main()
