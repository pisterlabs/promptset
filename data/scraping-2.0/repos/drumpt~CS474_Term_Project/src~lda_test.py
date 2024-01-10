import gensim
import preprocess
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.util import ngrams
import utils
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))


class LDA:
    def __init__(self, data_dir, num_trends, num_passes):
        a = time.time()
        self.num_trends = num_trends
        self.num_passes = num_passes
        self.dfs_2015, self.dfs_2016, self.dfs_2017, self.tdc_2015, self.tdc_2016, self.tdc_2017 = self.pre_process(data_dir)
        
        self.dict, self.corpus = self.encoding(self.tdc_2015)
        self.topics, self.lda_model = self.train_lda(self.dict, self.corpus)
        self.topic_dict = self.build_topic_dict()
        self.merge_dict = self.merge_clustering()
        self.trend_find(self.merge_dict, "2015", self.tdc_2015)


        self.dict, self.corpus = self.encoding(self.tdc_2016)
        self.topics, self.lda_model = self.train_lda(self.dict, self.corpus)
        self.topic_dict = self.build_topic_dict()
        self.merge_dict = self.merge_clustering()
        self.trend_find(self.merge_dict, "2016", self.tdc_2016)


        self.dict, self.corpus = self.encoding(self.tdc_2017)
        self.topics, self.lda_model = self.train_lda(self.dict, self.corpus)
        self.topic_dict = self.build_topic_dict()
        self.merge_dict = self.merge_clustering()
        self.trend_find(self.merge_dict, "2017", self.tdc_2017)

        b = time.time()
        print("Execution time: ")
        print(b-a)
        

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        # For fast check !
        # dfs_2015 = dfs_2015
        # dfs_2016 = dfs_2016
        # dfs_2017 = dfs_2017

        Processor = preprocess.Preprocessor()
        tdc_2015 = dfs_2015["body"].apply(lambda x: Processor.preprocess(x))
        tdc_2016 = dfs_2016["body"].apply(lambda x: Processor.preprocess(x))
        tdc_2017 = dfs_2017["body"].apply(lambda x: Processor.preprocess(x))
        
        return dfs_2015, dfs_2016, dfs_2017, tdc_2015, tdc_2016, tdc_2017

    def encoding(self, tokenized_doc):
        dictionary = corpora.Dictionary(tokenized_doc)
        corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
        # for i, c in enumerate(corpus):
        #   print(f"{i}th corpus data is {c}")
        #   if i>10:
        #     break
        return dictionary, corpus

    def train_lda(self, dict, corpus):
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_trends, id2word=dict, passes=self.num_passes)
        topics = lda_model.print_topics(num_words=10)
        
        return topics, lda_model

    def build_topic_dict(self):
        topic_dict = {}

        for i in range(self.num_trends):
          topic_dict[i]=[]
        # print(topic_dict.keys())
        # print("TOPIC DICT IS BUILDING")
        standard = 0.6
        for i, topic_list in enumerate(self.lda_model[self.corpus]):
          topic_list = sorted(topic_list, key=lambda x: x[1], reverse=True)
          # if topic_list[0][1] < standard:
            # print("It is smaller than standard")
            # continue
          if len(topic_list) == 0:
              continue
          if topic_list[0][1] < standard and len(topic_list) >= 2:
            topic_dict[topic_list[0][0]].append((i, topic_list[0][1]))
            topic_dict[topic_list[1][0]].append((i, topic_list[1][1]))
          elif topic_list[0][0] in topic_dict:
            topic_dict[topic_list[0][0]].append((i, topic_list[0][1]))
        return topic_dict    

    def merge_clustering(self):
      lda_model = self.lda_model
      topic_dict = self.topic_dict

      # for key in topic_dict:
      #   print(f"{key}th num of data is {len(topic_dict[key])}")

      info_mat = lda_model.get_topics() # topic vector
      # print(info_mat[0])
      while len(info_mat) > 10:
        # print(len(info_mat))
        cos_mat = np.zeros((len(info_mat), len(info_mat)))
        for i in range(len(info_mat)):
          for j in range(len(info_mat)):
            if i==j:
              continue
            else:
              cos_mat[i][j] = cos_sim(info_mat[i], info_mat[j])
        ind = np.unravel_index(np.argmax(cos_mat, axis=None), cos_mat.shape)
        new_one = (info_mat[ind[0]] + info_mat[ind[1]]) / 2
        del_index = [ind[0], ind[1]]
        # print(topic_dict.keys())
        # print(del_index)
        new_topic_dict = {}
        for key in topic_dict:
          if not key in del_index:
            new_topic_dict[len(new_topic_dict)] = topic_dict[key]
        new_topic_dict[len(new_topic_dict)] = topic_dict[ind[0]] + topic_dict[ind[1]]

        info_mat = np.delete(info_mat, del_index, axis=0)
        info_mat = np.concatenate((info_mat, [new_one]), axis=0)
        topic_dict = new_topic_dict

      # for key in topic_dict:
      #   print(f"{key}th num of data is {len(topic_dict[key])}")
      return topic_dict


    def trend_find(self, finding_dict, year_string, tokenized_doc):
      f = open("../output/" + year_string + "_Issues.txt", 'w')  
      for k in finding_dict:
        pre_A = sorted(finding_dict[k], key=lambda x: x[1], reverse=True)
        if len(pre_A) > 50:
          pre_A = pre_A[:50]
        A = tokenized_doc[[index[0] for index in pre_A]]
        pre_B = []
        B = []
        for a_key in finding_dict:
          if k == a_key:
            continue
          else:
            for elem in finding_dict[a_key]:
              pre_B.append(elem)
        pre_B = sorted(pre_B, key=lambda x: x[1], reverse=True)
        if len(pre_B) > 50:
          pre_B = pre_B[:50]
        B = tokenized_doc[[index[0] for index in pre_B]]
        # print(f"{key}th Topic Important Bigram is")
        result, result_2, result_3 = self.common_phrase(A, B, 2)
        result_3 = sorted(list(result_3.items()), key=lambda x: x[1], reverse=True)
        writing = result_3[:1]
        data = ""
        for w in writing:
          data = w[0][0] + " "+ w[0][1] + "\n"
        f.write(data)
      f.close()
      return result_3

    def common_phrase(self, text_list, untext_list, phrase_length):
      phrase_dict = {}
      cnt_n = 0
      for (i, t) in enumerate(text_list):
        ngram = ngrams(t, phrase_length)
        # cnt_n += len(ngram)
        for n in ngram:
          cnt_n += 1  
          n = tuple(n)
          if n not in phrase_dict:
            phrase_dict[n] = 0
          for (j, t_t) in enumerate(text_list):
            if i == j:
              continue
            mgram = ngrams(t_t, phrase_length)
            for m in mgram:
              m = tuple(m)
              if n == m:
                phrase_dict[n] += 1

      anti_phrase_dict = {}
      for key in phrase_dict:
        anti_phrase_dict[key] = 1
      for (k, t) in enumerate(untext_list):
        ngram = ngrams(t, phrase_length)
        for n in ngram:
          n = tuple(n)
          if n in anti_phrase_dict:
            anti_phrase_dict[n] += 1
            continue

      final_dict = {}
      N = len(text_list) + len(untext_list)
      for key in phrase_dict:
        tf = phrase_dict[key] / cnt_n
        df = anti_phrase_dict[key]
        idf = N / df
        idf = np.log10(idf)
        final_dict[key] = tf * idf

      return phrase_dict, anti_phrase_dict, final_dict


"""
    def trend_find(self):
        topic_dict = {}
        print("TOPIC DICT IS BUILDING")
        standard = 0.75
        for i, topic_list in enumerate(self.lda_model[self.corpus]):
            topic_list = sorted(topic_list, key=lambda x: x[1], reverse=True)
            if topic_list[0][1] < standard:
                # print("It is smaller than standard")
                continue
            if not topic_list[0][0] in topic_dict:
                topic_dict[topic_list[0][0]] = [i]
            else:
                temp_list = topic_dict[topic_list[0][0]]
                temp_list.append(i)
                topic_dict[topic_list[0][0]] = temp_list
        print("TOPIC DICT HAS BEEN BUILT")

        return topic_dict

    def merge_clustering(self):
        lda_model = self.lda_model
        topic_dict = self.topic_dict

        for key in topic_dict:
            print(f"{key}th num of data is {len(topic_dict[key])}")

        info_mat = lda_model.get_topics()  # topic vector
        while len(info_mat) > 10:
            print(len(info_mat))
            cos_mat = np.zeros((len(info_mat), len(info_mat)))
            for i in range(len(info_mat)):
                for j in range(len(info_mat)):
                    if i == j:
                        continue
                    else:
                        cos_mat[i][j] = cos_sim(info_mat[i], info_mat[j])
            ind = np.unravel_index(np.argmax(cos_mat, axis=None), cos_mat.shape)
            new_one = (info_mat[ind[0]] + info_mat[ind[1]]) / 2
            del_index = [ind[0], ind[1]]

            new_topic_dict = {}
            for key in topic_dict:
                if not key in del_index:
                    new_topic_dict[len(new_topic_dict)] = topic_dict[key]
            new_topic_dict[len(new_topic_dict)] = topic_dict[ind[0]] + topic_dict[ind[1]]

            info_mat = np.delete(info_mat, del_index, axis=0)
            print(f"Topic {ind[0]} and {ind[1]} are most similar!")
            info_mat = np.concatenate((info_mat, [new_one]), axis=0)
            topic_dict = new_topic_dict

        for key in topic_dict:
            print(f"{key}th num of data is {len(topic_dict[key])}")

    # def trend_find(self):
    # for k in topic_dict:
    #     # Randomly sampling the other topics'
    #     single = self.common_phrase(self.tokenized_doc[topic_dict[k]], 1)
    #     bi = self.common_phrase(self.tokenized_doc[topic_dict[k]], 2)
    #     tri = self.common_phrase(self.tokenized_doc[topic_dict[k]], 3)
    #     quad = self.common_phrase(self.tokenized_doc[topic_dict[k]], 4)
    #     result_arr = [single[0], bi[0], tri[0], quad[0]]
    #     result_arr.sort(key=lambda x:x[1], reverse=True)
    #     print(f"Best ngram was {result_arr[0]}")
    # return topic_dict

    def common_phrase(self, text_list, phrase_length):
        a = time.time()
        phrase_dict = {}
        for t in text_list:
            ngram = ngrams(t, phrase_length)
            for n in ngram:
                if n not in phrase_dict:
                    phrase_dict[n] = 0
                for t_t in text_list:
                    for i in range(len(t_t) - phrase_length + 1):
                        if t_t[i:i + phrase_length] == n[i:i + phrase_length]:
                            phrase_dict[n] += 1
        result = sorted(phrase_dict.items(), key=lambda x: x[1], reverse=True)
        b = time.time()
        print(f"Function {phrase_length}-length common_phrase time : {b - a}")
        return result[:5]
"""


class LDA_scikit():
    def __init__(self, directory, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.pre_process(directory)
        print("Preprocess ended")
        self.detokenized = self.detokenization(self.tokenized_doc)
        print("Detokenization ended")
        self.tfidf = self.tfidf_LDA(self.detokenized)
        print("Title based LDA ended")

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        # dfs_2015 = dfs_2015
        Processor = preprocess.Preprocessor()
        tokenized_doc = dfs_2015["title"].apply(lambda x: Processor.preprocess(x))
        tokenized_doc = tokenized_doc.apply(lambda x: [word for word in x if len(word) > 3])
        return dfs_2015, tokenized_doc

    def detokenization(self, tokenized_doc):
        detokenized = []
        for i in range(len(tokenized_doc)):
            t = ' '.join(tokenized_doc[i])
            detokenized.append(t)
        return detokenized

    def tfidf_LDA(self, detokenized):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # 상위 1,000개의 단어를 보존
        X = vectorizer.fit_transform(detokenized)
        lda_model = LatentDirichletAllocation(n_components=self.num_trends, learning_method='online',
                                              random_state=777, max_iter=1)
        lda_top = lda_model.fit_transform(X)
        terms = vectorizer.get_feature_names()
        n = 5
        for idx, topic in enumerate(lda_model.components_):
            print("Topic %d:" % (idx + 1), [(terms[i], topic[i].round(2))
                                            for i in topic.argsort()[:-n - 1:-1]])
        # print(f"LDA top is {lda_top}")
        return 1


if __name__ == "__main__":
    lda_class = LDA("../dataset", 100, 30)
    # lda_scikit = LDA_scikit("../dataset", 30)
