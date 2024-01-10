import nltk
import re
import string
# nltk.download('stopwords')
# nltk.download('punkt')
# import spacy.cli
# spacy.cli.download("en_core_web_sm")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm') # using script python -m spacy download en

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint

DATA_DIRECTORY = "./dags/services/data/"
VECTOR_STORE_PATH  = "./dags/services/vector_store_folder"

ALl_RAW_DATA_PATH = './dags/services/data/all_raw_data_2015-2023.csv'
CURRENT_YEAR_RAW_DATA_PATH = './dags/services/data/raw_data_2023.csv'

LDA_ALL_DATA_YEAR_PATH = './dags/services/data/lda_all_data_2015_2023.csv'
LDA_CURRENT_YEAR_DATA_PATH = './dags/services/data/lda_data_2023.csv'

def LDA(file_path=ALl_RAW_DATA_PATH,lda_file_path=LDA_ALL_DATA_YEAR_PATH,n=10,alpha=0.3,beta="auto"): #Using raw title data
    df = pd.read_csv(file_path)
    data_title = df.title.apply(lambda t : t.lower())

    stop_words = set(stopwords.words('english'))
    stop_words.add('inf')

    def remove_stopwords(article):
        article_tokens = word_tokenize(article)
        filtered_article = [word for word in article_tokens if not word in stop_words]
        return " ".join(filtered_article)
    def remove_extra_marks(article):
        extra_keys = ["’","—","”","“"]
        article_tokens = word_tokenize(article)
        filtered_article = [word for word in article_tokens if not word in extra_keys]
        return " ".join(filtered_article)
    def lemmatize(text):
        """Return text after performing the lemmztiztion"""
        doc = nlp(text)
        tokens = [token for token in doc]
        return  " ".join([token.lemma_ for token in doc])

    #removing stopwords
    data_title = data_title.apply(remove_stopwords)

    #removing Punctuations
    data_title = data_title.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

    #removing digits
    data_title = data_title.apply(lambda x: re.sub('\w*\d\w*','', x))
    
    #remove extra marks
    data_title = data_title.apply(remove_extra_marks)

    data_title = data_title.apply(lemmatize)

    #tokenize articles
    tokenize_title = data_title.apply(lambda x : x.split())
    id2word = corpora.Dictionary(tokenize_title)

    # Create Corpus
    texts = tokenize_title

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # LDA model   
    lda_model =  gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=n,
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=alpha,
                                            per_word_topics=True,
                                            eta = beta,
                                            update_every=1,
                                            )
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    #save lda result
    word_id_pairs = []
    for id_group, words in lda_model.print_topics():
        extracted_words = [word.split('"')[1] for word in words.split(' + ')]
        word_id_pairs.extend([(id_group, word) for word in extracted_words])

    # Create a DataFrame from the extracted word-ID pairs
    df = pd.DataFrame(word_id_pairs, columns=['ID Group', 'Word'])

    # Save the DataFrame to a CSV file
    df.to_csv(lda_file_path,index=False)
    print(f"Succesfully added new data!! to {lda_file_path}")
    print(df.head(10))
    return word_id_pairs



if __name__ == "__main__":
    print()
    LDA(file_path=ALl_RAW_DATA_PATH,n=10,alpha=0.3,beta="auto")