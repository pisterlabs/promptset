from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from .config import OPENAI_API_KEY
from langchain.vectorstores import Chroma
import pickle
from soylemma import Lemmatizer
from konlpy.tag import Okt
from konlpy.tag import Mecab
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os.path   
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np


#"/prj/out/exp_finetune"
#"beomi/kcbert-base"
#BM-K/KoSimCSE-roberta-multitask
class Embedding_Document:

    def __init__(self, save_tfvector_dir, save_doc2vec_dir, save_bert_dir, save_bm25_dir):
        self.embedd_model = 'BM-K/KoSimCSE-roberta-multitask'
        self.save_tfvector_dir = save_tfvector_dir
        self.save_doc2vec_dir = save_doc2vec_dir
        self.save_bert_dir = save_bert_dir
        self.save_bm25_dir = save_bm25_dir

        self.embeddings = SentenceTransformer(self.embedd_model)
    
    # def __get_embedding_model(self):
    #     print("embedding model load")
    #     model_kwargs = {'device': 'cuda'}
    #     encode_kwargs = {'normalize_embeddings': True}

    #     self.embeddings = HuggingFaceEmbeddings(model_name=self.embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    #     self.embeddings.client.tokenizer.pad_token = self.embeddings.client.tokenizer.eos_token
    #     self.embeddings.client.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __read_pdfs(self, pdf_files):
        if not pdf_files:
            return []
        
        pages = []
        for path in pdf_files:
            loader = PyPDFLoader(path)
            for page in loader.load_and_split():
                pages.append(page)

        return pages


    def __split_pages(self, pages):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap  = 100,
            length_function = len,
            is_separator_regex = False,
            separators = ["\n\n"]
        )
        texts = text_splitter.split_documents(pages)
        return texts



    # def bert_embedding(self, pdf_files):
    #     try:
    #         print("embedding model load")
    #         model_kwargs = {'device': 'cuda'}
    #         encode_kwargs = {'normalize_embeddings': True}

    #         embeddings = HuggingFaceEmbeddings(model_name=self.embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    #         embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
    #         embeddings.client.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    #         print("read_pdfs")
    #         pages = self.__read_pdfs(pdf_files)
    #         print("split_pdfs")
    #         texts = self.__split_pages(pages)

    #         persist_directory = "/prj/src/data_store" 
    #         print("pdf embedding")
    #         db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
    #             documents=texts,
    #             embedding=embeddings,
    #             persist_directory=persist_directory)

    #         print("pdf embedd and saved")

    #         return True, None
    #     except Exception as e:
    #         print("error to embedding")
    #         print(f"error_msg: {e}")
    #         return False, e
        

    def __find_elements_with_specific_value(self, tuple_list, target_value):
        result_list = [t[0] for t in tuple_list if t[1] == target_value]
        return result_list
    
    def __find_highest_doc_index(self, result_list):
        max_value = float('-inf') #초기 최댓값을 음의 무한대로 설정
        max_index = None

        for i, sublist in enumerate(result_list):
            if len(sublist) > 0 and isinstance(sublist[0], (int, float)):
                value = sublist[0]
                if value > max_value:
                    max_value = value
                    max_index = i

        return max_index

    def __sentence_tokenizing(self, query, mode="string"):
        lemmatizer = Lemmatizer()

        t = Okt()
        stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        query = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", query)

        if mode == "string":
            lemm_sentence = ''
            for text in t.pos(query):
                if text[0] in stopwords:
                    continue
                result_lemm = self.__find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
                if len(result_lemm) == 0:
                    lemm_sentence += f"{text[0]} "
                else:
                    # print(result_lemm)
                    lemm_sentence += f"{result_lemm[0]} "
        elif mode == "array":
            lemm_sentence = []
            for text in t.pos(query):
                if text[0] in stopwords:
                    continue
                result_lemm = self.__find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
                if len(result_lemm) == 0:
                    lemm_sentence.append(text[0])
                else:
                    lemm_sentence.append(result_lemm[0])

        return lemm_sentence


    def embedding_doc2vec(self, pdf_files):
        try:
            print("doc2vec embedding")
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)

            content = []
            source = []
            origin_content = []

            if os.path.isfile(f'{self.save_doc2vec_dir}/content.pkl'):
                with open(f'{self.save_doc2vec_dir}/content.pkl', 'rb') as f:
                    load_doc_info = pickle.load(f)
                    content = load_doc_info["content"]
                    origin_content = load_doc_info["origin_content"]
                    source = load_doc_info["source"]

            for text in texts:
                origin_content.append(text.page_content)
                result_sentence = self.__sentence_tokenizing(text.page_content,"array")
                content.append(result_sentence)
                source.append((text.metadata['source'],text.metadata['page']))

            tagged_data = [TaggedDocument(words=content, tags=[str(id)]) for id, content in enumerate(content)]

            max_epochs = 10

            model = Doc2Vec(
                window=20, #모델 학습할 때 앞뒤로 보는 단어의 수
                vector_size=300, #벡터 차원의 크기
                alpha=0.025, #lr
                min_alpha=0.025,
                min_count=5, #학습에 사용할 최소 단어 빈도 수
                dm=0, #학습방법 1=PV-DM, 0=PV_DBOW
                negative=5, #Complexity Reduction 방법, negative sampling
                seed=9999
            )

            model.build_vocab(tagged_data)

            for epoch in range(max_epochs):
                print(f'iteration {epoch}')
                model.train(tagged_data, 
                            total_examples=model.corpus_count,
                            epochs=max_epochs)

                model.alpha -= 0.002
                model.min_alpha = model.alpha

            doc_info = {
                "content": content,
                "origin_content": origin_content,
                "source": source
            }

            #save
            with open(f'{self.save_doc2vec_dir}/content.pkl', 'wb') as f:
                pickle.dump(doc_info, f)

            model.save(f'{self.save_doc2vec_dir}/model.doc2vec')
            # with open(f'{self.save_doc2vec_dir}/model.pkl', 'wb') as f:
            #     pickle.dump(model, f)
        except Exception as e:
            print(f"error:{e}")
            return False, e

        return True, None



    def embedding_bert(self, pdf_files):
        try:
            print("bert embedding")
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)


            content = []
            source = []
            origin_content = []
            content_vectors= []

            if os.path.isfile(f'{self.save_bert_dir}/content.pkl'):
                with open(f'{self.save_bert_dir}/content.pkl', 'rb') as f:
                    load_doc_info = pickle.load(f)
                    content = load_doc_info["content"]
                    origin_content = load_doc_info["origin_content"]
                    source = load_doc_info["source"]   

            # if os.path.isfile(f'{self.save_bert_dir}/contentvector.pkl'):
            #     with open(f'{self.save_bert_dir}/contentvector.pkl', 'rb') as f:
            #         content_vectors = pickle.load(f)

            for text in texts:
                origin_content.append(text.page_content)
                result_sentence = self.__sentence_tokenizing(text.page_content,"string")
                content.append(result_sentence)
                source.append((text.metadata['source'],text.metadata['page']))

            # for sentence in content:
            #     vectors = self.embeddings.encode(sentence)
            #     content_vectors.append(vectors)
            for text in texts:
                vectors = self.embeddings.encode(text.page_content)
                content_vectors.append(vectors)


            doc_info = {
                "content": content,
                "origin_content": origin_content,
                "source": source
            }

            with open(f'{self.save_bert_dir}/content.pkl', 'wb') as f:
                pickle.dump(doc_info, f)
            with open(f'{self.save_bert_dir}/contentvector.pkl', 'wb') as f:
                pickle.dump(content_vectors, f)
        except Exception as e:
            print(f"error:{e}")
            return False, e
        
        return True, None



    def embedding_tf_idf(self, pdf_files):
        try:
            print("tf-idf embedding")
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)

            content = []
            source = []
            origin_content = []

            if os.path.isfile(f'{self.save_tfvector_dir}/content.pkl'):
                with open(f'{self.save_tfvector_dir}/content.pkl', 'rb') as f:
                    load_doc_info = pickle.load(f)
                    content = load_doc_info["content"]
                    origin_content = load_doc_info["origin_content"]
                    source = load_doc_info["source"]

            
            for text in texts:
                origin_content.append(text.page_content)
                result_sentence = self.__sentence_tokenizing(text.page_content,"string")
                content.append(result_sentence)
                source.append((text.metadata['source'],text.metadata['page']))


            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(content)
            doc_info = {
                "content": content,
                "origin_content": origin_content,
                "source": source
            }

            
            # # save
            with open(f'{self.save_tfvector_dir}/content.pkl', 'wb') as f:
                pickle.dump(doc_info, f)
            with open(f'{self.save_tfvector_dir}/test2.pkl', 'wb') as f:
                pickle.dump(tfidf_matrix, f)
            with open(f'{self.save_tfvector_dir}/tfidf_vectorizer.pkl', 'wb') as file:
                pickle.dump(vectorizer, file)
        except Exception as e:
            print(f"error:{e}")
            return False, e
        
        return True, None
    
    def bm25_embedding(self, pdf_files):
        try:
            print("bm25-idf embedding")
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)

            content = []
            source = []
            origin_content = []

            if os.path.isfile(f'{self.save_bm25_dir}/content.pkl'):
                with open(f'{self.save_bm25_dir}/content.pkl', 'rb') as f:
                    load_doc_info = pickle.load(f)
                    content = load_doc_info["content"]
                    origin_content = load_doc_info["origin_content"]
                    source = load_doc_info["source"]

            for text in texts:
                origin_content.append(text.page_content)
                result_sentence = self.__sentence_tokenizing(text.page_content,"array")
                content.append(result_sentence)
                source.append((text.metadata['source'],text.metadata['page']))

            doc_info = {
                "content": content,
                "origin_content": origin_content,
                "source": source
            }


            bm25 = BM25Okapi(content)

            # # save
            with open(f'{self.save_bm25_dir}/content.pkl', 'wb') as f:
                pickle.dump(doc_info, f)
            with open(f'{self.save_bm25_dir}/bmvector.pkl', 'wb') as file:
                pickle.dump(bm25, file)

        except Exception as e:
            print(f"error:{e}")
            return False, e
        
        return True, None    


    def bm25_search_doc(self, query, k):
        with open(f'{self.save_bm25_dir}/bmvector.pkl', 'rb') as file:
            bm25 = pickle.load(file)
        with open(f'{self.save_bm25_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)


        origin_content = doc_info["origin_content"]
        # content = doc_info["content"]
        source = doc_info["source"]

        new_query = self.__sentence_tokenizing(query, "array")
        scores = bm25.get_scores(new_query)
        max_score_index = scores.argmax()


        return origin_content[max_score_index], source[max_score_index][0], source[max_score_index][1], scores[max_score_index]

    def bert_search_doc(self, query, k):
        with open(f'{self.save_bert_dir}/contentvector.pkl', 'rb') as f:
            contentvector = pickle.load(f)
        with open(f'{self.save_bert_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)

        origin_content = doc_info["origin_content"]
        # content = doc_info["content"]
        source = doc_info["source"]
        print(origin_content)

        # new_query = self.__sentence_tokenizing(query, "string")
        query_vector = self.embeddings.encode([query])

        similarity_scores = cosine_similarity(contentvector, query_vector)

        # result_index = self.__find_highest_doc_index(similarity_scores)
        result_index = similarity_scores.argmax()

        return origin_content[result_index], source[result_index][0], source[result_index][1], float(similarity_scores[result_index][0])



    def doc2vec_search_doc(self, query, k):
        # with open(f'{self.save_doc2vec_dir}/model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        model = Doc2Vec.load(f'{self.save_doc2vec_dir}/model.doc2vec')
        with open(f'{self.save_doc2vec_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)

        model.random.seed(9999)

        new_query = self.__sentence_tokenizing(query, "array")
        inferred_vector = model.infer_vector(new_query)
        return_docs = model.docvecs.most_similar(positive=[inferred_vector], topn=k)

        origin_content = doc_info["origin_content"]
        source = doc_info["source"]
        index = int(return_docs[0][0])

        # print(f"문서내용: {origin_content[index]}\n")
        # print(f"점수: {return_docs[0][1]}")

        return origin_content[index], source[index][0], source[index][1], return_docs[0][1] 


    def search_doc_bm_bert(self, query, k, bm_k, alpha=0.9):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)


        with open(f'{self.save_bm25_dir}/bmvector.pkl', 'rb') as file:
            bm25 = pickle.load(file)
        with open(f'{self.save_bm25_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)


        origin_content = doc_info["origin_content"]
        # content = doc_info["content"]
        source = doc_info["source"]

        arr_new_query = self.__sentence_tokenizing(query, "array")
        doc_scores = bm25.get_scores(arr_new_query)
        document_idx = np.argpartition(-doc_scores, range(k))[0:bm_k]
        # top_documents = [origin_content[i] for i in document_idx]
        doc_scores = [doc_scores[i] for i in document_idx]

        #rerank
        top_source = []
        top_vector = []
        top_origin = []


        if os.path.isfile(f'{self.save_bert_dir}/content.pkl') and os.path.isfile(f'{self.save_bert_dir}/contentvector.pkl'):
            with open(f'{self.save_bert_dir}/content.pkl', 'rb') as f:
                load_doc_info = pickle.load(f)
                origin_content = load_doc_info["origin_content"]
                source = load_doc_info["source"]   
            with open(f'{self.save_bert_dir}/contentvector.pkl', 'rb') as f:
                contentvector = pickle.load(f)
        else:
            raise Exception("No Bert Datas!!")

        for id in document_idx:
            top_vector.append(contentvector[id])
            top_source.append(source[id])
            top_origin.append(origin_content[id])

        print(f"bm_result: {top_origin[0]}") #bm25의 결과

        query_vector = self.embeddings.encode([query])
        similarity_scores = cosine_similarity(query_vector, top_vector)
        
        cos_score = similarity_scores[0]
        cos_score = softmax(cos_score)

        result_scores = (1-alpha)*np.array(doc_scores) + alpha*cos_score 
        top_results = np.argpartition(-result_scores, range(k))[0:k]
        rerank_documents = [top_origin[i] for i in top_results]
        rerank_scores = [result_scores[i] for i in top_results]
        rerank_sources = [top_source[i][0] for i in top_results]
        rerank_pages = [top_source[i][1] for i in top_results]
        rerank_scores = list(map(lambda x: float(x), rerank_scores))

        
        return rerank_documents, rerank_sources, rerank_pages, rerank_scores

        
        



    def tf_idf_search_doc(self, query, k):
        with open(f'{self.save_tfvector_dir}/test2.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(f'{self.save_tfvector_dir}/tfidf_vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
        with open(f'{self.save_tfvector_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)


        origin_content = doc_info["origin_content"]
        # content = doc_info["content"]
        source = doc_info["source"]

        new_query = self.__sentence_tokenizing(query, "string")
        query_vector = loaded_vectorizer.transform([new_query])

        similarity_scores = cosine_similarity(tfidf_matrix, query_vector)
        result_index = self.__find_highest_doc_index(similarity_scores)

        # print(origin_content[result_index])
        # print(source[result_index])
        # print(similarity_scores[result_index])

        return origin_content[result_index], source[result_index][0], source[result_index][1], similarity_scores[result_index][0]

