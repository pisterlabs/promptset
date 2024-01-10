from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

STOP = stopwords.words('english')
lemma = WordNetLemmatizer()

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [lemma.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in STOP] # remove stopwords
    return tokens


class ContentBasedRecommendations():
    def __init__(self, df_content):
        df_content['doc_body'] = df_content['doc_body'].fillna('')
        df_content['doc_description'] = df_content['doc_description'].fillna('')
        df_content['all_text'] = df_content['doc_full_name'] + ' ' + df_content['doc_body'] + ' ' + df_content['doc_description'] 
        self.df_content = df_content
        self.indices = pd.Series(df_content.index, index=df_content['doc_full_name']).drop_duplicates()

    def get_tfidf(self):
        # Compute tfidf 
        vect = TfidfVectorizer(stop_words='english')
        count_matrix = vect.fit_transform(self.df_content.all_text.values)
        return count_matrix

    def get_cosine_similarity_matrix(self, count_matrix):
        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(count_matrix, count_matrix)

    def get_recommendations_tfidf(self, name):
        if hasattr(self, 'cosine_sim'):
            # Get the index of the movie that matches the title
            idx = self.indices[name]
            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Get the scores of the 10 most similar movies remove own
            print('Making Reccomendations...')
            sim_scores = sim_scores[1:11]
            # Get the movie indices
            artcle_indices = [i[0] for i in sim_scores]
            # Return the top 10 most similar movies
            return df_content['doc_full_name'].iloc[artcle_indices]
        else:    
            print('building tfidf model')
            count_matrix = self.get_tfidf()
            self.get_cosine_similarity_matrix(count_matrix)
            return self.get_recommendations_tfidf(name)

    def tokenize(self, raw_text):
        all_tokens = []
        for text in raw_text:
            all_tokens.append(my_tokenizer(text))
        return all_tokens

    def build_doc2vec(self):
        all_tokens = self.tokenize(self.df_content.all_text.values)
        sentences = []
        for i, line in enumerate(all_tokens):
            sentences.append(TaggedDocument(line, [i]))

        self.model = Doc2Vec(documents=sentences)
        print('model built')

    def get_recommendations_doc2vec(self, name):
        if hasattr(self, 'model'):
            tokens = name.lower().split()
            new_vector = self.model.infer_vector(tokens)
            sims = self.model.docvecs.most_similar([new_vector])
            print('getting recs')
            recs = []
            for sim in sims:
                name = self.df_content.loc[self.df_content.article_id == sim[0], 'doc_full_name'].values[0]
                recs.append((name, round(sim[1], 2)))
            return recs
        else:
            print('building doc2vec model')
            self.build_doc2vec()
            return self.get_recommendations_doc2vec(name)
