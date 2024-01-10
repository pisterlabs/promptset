
from os.path import exists
import csv #CSV reader
from collections import Counter #Python Collection Counter
import re #Regex library

#NLTK Library
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 

#Gensim Library
import gensim
import gensim.models as models
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint


#Few dependencies for Library
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords') 



#Creating LDAModeler class
#LDAModeler class is used to train and test the models and also to print the metrics
class LdaModeler():
  #Arguments: corpus: Doc2bow_Corpus, dictionary: Vectorized Dictionary, split: Train/Test, passes: LDA_Model hyperparameter, iterations: LDA_Model hyperparameter
  def __init__(self, corpus, dictionary, texts, split, model_file="LDA_model_params_8", passes=50, iterations=400):
    self.corpus = corpus
    self.model_file = model_file
    self.dictionary = dictionary
    self.texts = texts
    if(split=="train"):
      self.passes = passes
      self.iterations = iterations

      self.num_topics = 9
      self.random_state = 0
      self.chunksize = 500
      self.alpha = 'auto'
      self.eta = 'auto'
      self.per_word_topics = True
      self.model = self.train_model() #Model Training is called here
    elif(split=="test"):
      self.model = models.ldamodel.LdaModel.load(self.model_file)
      self.test_model()
    
    self.print_metrics()

  #The train_model function is called from the constructor during the object creation
  def train_model(self):
    lda_model = models.ldamodel.LdaModel(self.corpus, num_topics=self.num_topics,
                                         id2word=self.dictionary,
                                         passes=self.passes, 
                                         iterations=self.iterations,
                                         random_state=self.random_state, 
                                         chunksize=self.chunksize, 
                                         alpha='auto', eta='auto',
                                         per_word_topics=True)

    for idx, topic in lda_model.print_topics(-1): #Top 10 words for each topic and their probabilities are printed for each topic as well
      print('Topic: {} Word: {}'.format(idx, topic))

      lda_model.save(self.model_file) # Saving the trained model
    
    return lda_model

  def test_model(self):
    for idx, topic in self.model.print_topics(-1): #Top 10 words for each topic and their probabilities are printed for each topic as well
      print('Topic: {} Word: {}'.format(idx, topic))
      
  #print_metrics function is used to test the coherence(c_v and u_mass) and the perplexity score for the test data and the training data
  def print_metrics(self):
    """cmcv = CoherenceModel(model=self.model, 
                          texts=self.texts, 
                          dictionary=self.dictionary,
                          coherence='c_v')
    print("C_v coherence: ",cmcv.get_coherence())

    cmumass = CoherenceModel(model=self.model, 
                             texts=self.texts, 
                             dictionary=self.dictionary,
                             coherence='u_mass')
    print("u_mass coherence: ",cmumass.get_coherence())"""
    print("Perplexity: ",self.model.log_perplexity(self.corpus))



#Vectorizer class converts each email to vectors(Doc2Bow)
class Vectorizer():
  def __init__(self, docs):
    self.docs=docs
    self.dictionary, self.bow_corpus = self.vectorize()
  
  def vectorize(self): #This function takes the Doc2bo corpus and converts into vectors for each email
    dct = Dictionary(self.docs) #Converting thw words into dictionary
    dct.filter_extremes(no_below=5, no_above=0.5, keep_n=5000) # removing the words which are less than 5 and have frequency percentage of more than 50%
    corpus = [dct.doc2bow(doc) for doc in self.docs] #Converting itno doc2bow

    return dct, corpus

# This class is used to proprocess the enter email data
class EmailPreprocessor():
  def __init__(self, data_file):
    self.data_file = data_file
    emailstemp = []
    with open(data_file, newline='\n') as csvfile: #Converting into CSV file
      docs = csv.reader(csvfile, delimiter=',')
      for doc in docs:
        emailstemp.append(', '.join(doc))
    
    self.df =  emailstemp
    if("train" in data_file): #Checking if the data_file is training or testing
      self.split = "train"
    else:
      self.split = "test"
    self.emails_df = self.create_dataframe()
    self.emails = self.preprocess_email()
    
  #Applying filter rules using regex 
  def apply_filter_rules(self, index):
    sentence = re.sub('http://\S+|https://\S+', '', index)  #Replacing URLS
    sentence = re.sub('\S*@\S*\s?','',sentence)   #Replacing emails
    sentence = re.sub('\<\S*@\S*\s?>','',sentence) #Replacing <emails and strings >
    sentence = re.sub('\([^)]*\)','',sentence) #Replacing (strings)
    sentence = re.sub('[^a-z]+',' ',sentence) #Replacing alphanumeric characters
    sentence = ' '.join(w for w in sentence.split() if len(w)>2)
    return sentence

  #DataFrame is created from the corpus 
  def create_dataframe(self):
    from_element = []
    subject_element = []
    email_element = []


    for rows in self.df[0:]:
      from_count = 0
      subject_count = 0
      line_check = False
      email_string = ''
      #Looping each email and splitting it using "\n" operator
      for index in rows.split('\n'):
        index =index.lower()
        if re.search('^from:', index) and from_count == 0: # Checking for from
          from_count = 1
          from_element.append(re.search('^from:', index).string.split(":")[1])
        if re.search('^subject:', index) and subject_count == 0: # Checking for Subject
          subject_count += 1
          string = re.search('^subject:', index).string.split("subject:")[1]
          string = string.replace("re:","")
          subject_element.append(self.apply_filter_rules(string))
        if re.search('^lines:', index): #Checking for lines
            line_check = True
        if line_check == True:
          if index != "":
            if not re.search('[nntp-posting-host|lines|reply\-to|organization]:\s', index): #Removing all the other email headings
              email_string = email_string+" "+ self.apply_filter_rules(index)
      email_element.append(email_string)
    email_df = {'from':from_element, 'subject':subject_element, 'email':email_element}
    return email_df

  def make_trigrams(self, texts, trigram_mod, bigram_mod): # Making trigrams
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

  def get_wordnet_pos(self, word): #Checking POS for the word
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

  def lemmatize_tokens(self, tokens): #Lemmatizing the tokens
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens]


  def preprocess_email(self): #Appending all the email into list of list
    emails_arr = []
    stops = stopwords.words('english')
    stops += ['article', 'writes', 'know', 'could', 'would', 'well']
    for idx, doc in enumerate(self.emails_df['email']):
      email = []
      words = nltk.word_tokenize(doc)
      for word in words:
        if word not in stops:
          email.append(word)
    
      emails_arr.append(self.lemmatize_tokens(email))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(emails_arr, min_count=10, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[emails_arr], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    trigrams =  self.make_trigrams(emails_arr, trigram_mod, bigram_mod)
   
    return trigrams # Returning the trigram model which also contains unigram, bigram and trigram
      

class TopicClassifer(): # This class is used to call the other classes for training, testing, preprocessing and vectorizing
  def __init__(self, split, data_file="T2_train.csv", model_file="LDA_model_params_8"): 
    if(split=="train"): # Checking for saved model 

      self.email_preprocessor = EmailPreprocessor(data_file)
      self.vectorizer = Vectorizer(self.email_preprocessor.emails)
      self.lda_model = LdaModeler(self.vectorizer.bow_corpus, 
                                  self.vectorizer.dictionary,
                                  self.email_preprocessor.emails,
                                  split, model_file, 50, 400)
      
    else:
      self.email_preprocessor = EmailPreprocessor(data_file)
      self.vectorizer = Vectorizer(self.email_preprocessor.emails)
      self.lda_model = LdaModeler(self.vectorizer.bow_corpus, 
                                  self.vectorizer.dictionary,
                                  self.email_preprocessor.emails,
                                  split, model_file)
  
if __name__ == "__main__":

  TopicClassifer("train", "T2_train.csv", model_file="LDA_model_params_8")





