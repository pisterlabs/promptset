import os, qdrant_client, re
import pandas as pd
from typing import Iterable
from openai import OpenAI
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
#código feito para carregar novos documentos na qdrant cloud

"""
TODO
fazer um arquivo/função de ETL, que extraia o PDF e carregue no vectorDB 
"""


class QdrantTextLoader:
    """
    Classe para carregar arquivos txt com o padrão de questões do ENEM no vectorDB Qdrant, permitindo RAG com esses textos
    
    Os arquivos são quebrados em pedaços correspondentes a cada questão e são carregados individualmente, cada questão sendo um vetor
    
     Args:
           collection_name (str) : nome da coleção do Qdrant que vai receber os dados
           
           QDclient (qdrant_client.QdrantClient) : objeto do client QDrant que contém a coleção

    """
    
    __OPENAI_VECTOR_PARAMS = VectorParams(size=1536, distance= Distance.COSINE, hnsw_config= None, quantization_config=None, on_disk=None)
    __YEAR_PATTERN:str = "20\d{2}" #padrões REGEX para pegar o ano e a matéria de cada arquivo das questões
    __SUBJECT_PATTERN: str = "_(.{3,}?)_" #padrão para achar a matéria da questão eng, lang, huma.....
    __CORRECT_ANSWER_STR: str = "(RESPOSTA CORRETA)"
    __EMBEDDINGS_MODEL:str = "text-embedding-ada-002"
    __QUESTION_SUBJECTS: set[str] = {"eng", "lang", "spani", "natu", "math", "huma"}

    QDclient: qdrant_client.QdrantClient

    def __init__(self, QDclient: qdrant_client.QdrantClient)->None:
        """
        Construtor para a classe QdrantTextLoader

        Args:
           QDclient (qdrant_client.QdrantClient) : objeto do client QDrant que contém a coleção
        """
        if not isinstance(QDclient,qdrant_client.QdrantClient):
            raise TypeError("Argumento do cliente do Qdrant não é do tipo correto")
        
        self.QDclient =  QDclient
        load_dotenv(os.path.join("keys.env"))

    def __qdrant_create_collection(self)->bool:
        return self.QDclient.create_collection( 
            collection_name= self.collection_name,
            vectors_config=self.__OPENAI_VECTOR_PARAMS
        )
    
    def __get_openAI_embeddings(self,text:str)->list[float]:
        embed_client = OpenAI()
        response = embed_client.embeddings.create(
            input= text,
            model= self.__EMBEDDINGS_MODEL
        )
        return response.data[0].embedding

    def QDvector_search(self, vector:list[float], vector_num:int = 1)->list:
        """
        Função para realizar uma busca por vetores dentro da coleção da instância da classe
        
        Args:
           vector (list[float]) : um vetor de embeddings do tipo da openAI, vão ser buscados outros vetores similar a esse
           vector_num (opcional, por padrão =1 ) (int) : o número de vetores que vão ser retornados
        
        Retorno: 
            Lista de vetores (parte númerica , payload e metadados) retornados 
        """
        

        """
        TODO
        fazer checagem se os vetores de input são os da openAI
        """
        search = self.QDclient.search(
          collection_name=self.collection_name,
          query_vector = vector,
          limit= vector_num
        )
        return search

    def __text_chunk_splitter(self, text:str, split_key:str)->Iterable[str]: #a split key vai ser (RESPOSTA CORRETA)
        """
        Iterador que retorna a proxima substr que corresponde a uma questão inteira
        """
        current_key_posi: int = 0
        CORRECT_ALTERNATIVE_BUFF: int = 22  #tamanho da split key até a alternativa correta:  (RESPOSTA CORRETA): D
        
        while (next_key_posi := text.find(split_key, current_key_posi)) != -1:
            str_slice:str = text[current_key_posi: next_key_posi + CORRECT_ALTERNATIVE_BUFF]
            current_key_posi = next_key_posi + CORRECT_ALTERNATIVE_BUFF  #começar a procurar da posição atual + buffer
            yield str_slice

    def __etl_metadata_saving(
        self,
        stats_csv_path:str, 
        current_year:int, 
        subject: str, 
        all_questions:int,
        added_questions:int
    )->None:
    
        if not stats_csv_path:
            raise IOError("Caminho para o arquivo não pode ser vazio")

        if ".csv" not in stats_csv_path:
          raise IOError("Arquivo não é do tipo CSV")
        
        ALL_QUESTIONS_INDEX: str = f"{current_year} todas questoes" #esses vão ser o nome dos indexes (linhas) do DF para guardar os valores do total de questões de um arquivo e do total add desse arqui
        ADDED_QUESTIONS_INDEX: str = f"{current_year} questoes add"

        if os.path.exists(stats_csv_path) and os.path.getsize(stats_csv_path) > 0:
          df = pd.read_csv(stats_csv_path, index_col= 0)
        else:
          df = pd.DataFrame()
        
        if subject not in df.columns: # se a matéria não existir no dataframe/csv, cria uma coluna vazia com o nome desse matéria 
           df[subject] = None
        
        if ALL_QUESTIONS_INDEX not in df.index:
            df.loc[ALL_QUESTIONS_INDEX] = None
            df.loc[ADDED_QUESTIONS_INDEX] = None
            
        df.at[ALL_QUESTIONS_INDEX, subject] = all_questions #coloca o valor de questões adicionadas na coluna e linha específica
        df.at[ADDED_QUESTIONS_INDEX, subject] = added_questions
 
        df.to_csv(stats_csv_path, index=True)

    def file_to_vectorDB(self, QD_collection:str , txt_file_path:str, save_extraction_stats: bool = False , stats_csv_path: str = "")->None:
        
        if ".txt" not in txt_file_path:
            raise IOError("essa função apenas aceita arquivos .TXT")

        if not isinstance(txt_file_path, str):
            raise TypeError("path para o arquivo não é uma string")

        if not isinstance(self.QDclient,qdrant_client.QdrantClient):
            raise TypeError("Argumento do cliente do Qdrant não é do tipo suportado")
        
        try:
           collection = self.QDclient.get_collection(QD_collection)
           vector_count:int  = collection.vectors_count

        except Exception as e:
           if isinstance(e, qdrant_client.http.exceptions.UnexpectedResponse ):
              print("coleção não existe, tentando criar uma nova")
            
              if self.__qdrant_create_collection():
                  vector_count:int  = 0
                  print("nova coleção criada")
              else:
                raise Exception("Não foi possível criar uma nova coleção")
    
        backslash_index: int = txt_file_path.rfind("/") #vai da direita pra esquerda até ahcar o primeiro / , isso é para isolar apenas o nome do arquivo no path e permitir achar a matéria
        file_str:str = txt_file_path[backslash_index+1 : len(txt_file_path)]

        if not (year_matches_list := re.findall(self.__YEAR_PATTERN , file_str)):
             raise IOError("Nome do arquivo não tem referencia ao ano da prova")
        else:
             test_year: int = int(year_matches_list[0])
        
        if not (subject_matches_list := re.findall(self.__SUBJECT_PATTERN, file_str)):
          raise IOError("Nome do arquivo não tem referencia à matéria das questões")
        else:  
            subject:str = subject_matches_list[0]

        print(f"qntd inicial de vetores {vector_count}") # Os IDs dos vetores a serem inseridos correspondem a qntd de vetores já existentes na collection do Qdrant

        start_amount: int = vector_count  #o primeiro vetor foi add com id=0 , o segundo com id=1, então o ID do novo vai ser a qntd de vetores ja existentes
        
        with open(txt_file_path, "r") as f:
            entire_text: str = f.read()

        text_and_embedings : dict[str,list[float]] = {chunk : self.__get_openAI_embeddings(chunk) for chunk in self.__text_chunk_splitter(entire_text, self.__CORRECT_ANSWER_STR)}
        #dict comprehension para gerar um dict com os pedaços de texto das questões como key e os seus embeddings como values

        self.QDclient.upsert( #a função de upsert é chamada apenas uma vez com toda  lista de pontos/vetores
                collection_name = QD_collection,
                points= [ #varios objetos da classe de pontos são instanciadas, cada uma com os keys e values do dicionário de text e embeddings
                    PointStruct(
                      id = idx,
                      vector = text_and_embedings[text_chunk],
                      payload= {"page_content":text_chunk, "metadata": {"materia": subject, "ano": test_year}}
                    )
                for idx, text_chunk in enumerate(text_and_embedings , vector_count)  
                ] 
        )
        
        vector_count += len(text_and_embedings) #tamanho do dicionário é o número de questões no arquivo de texto
   
        all_new_questions: int = vector_count - start_amount
        print(f"Tentou inserir {all_new_questions} questões no vectorDB") #quantidade de novas questões que eram para ser adicionadas

        final_vector_count: int  = self.QDclient.get_collection(QD_collection).vectors_count #mudança na qntd de vetores no vector db
        questions_added: int = final_vector_count - start_amount
        print(f"Foram inseridas  {questions_added} questões no vectorDB, para um total de {final_vector_count} questões")   

        vector_count = final_vector_count  #caso não seja possível colocar todos os vetores na collection, a variavel de classe que conta os vetores é corrigida com o real número
       
        if save_extraction_stats: #caso o argumento de salvar os dados da inserção de texto seja true, vamos chamar a função para fazer isso
            self.__etl_metadata_saving(
                stats_csv_path=stats_csv_path,
                current_year= test_year,
                subject= subject,
                all_questions= all_new_questions,
                added_questions= questions_added 
            )  

        if all_new_questions !=  questions_added:
            print("Não foi possível adicionar todas as questões no vectorDB") 

    def dict_to_vectorDB(self, QD_collection:str, subjects_and_questions: dict[str,str] ,save_extraction_stats: bool = False , stats_csv_path: str = "" )->None:
        """
        Função que recebe um dicionário de matérias do ENEM e questões associadas e o ano da prova e carrega elas num vector DB
        """

        if not isinstance(subjects_and_questions, dict):
            raise TypeError("variável de input não é um dicionário")
        
        if not subjects_and_questions.get("test_year"):
            raise IOError("dicionário não contem o ano do teste")
                
        test_year: int = int(subjects_and_questions.pop("test_year"))
        
        for subject in  subjects_and_questions:
            if subject not in self.__QUESTION_SUBJECTS:
                raise IOError("o dicionário contém matérias não suportadas")
        try:
           collection = self.QDclient.get_collection(QD_collection)
           vector_count:int  = collection.vectors_count

        except Exception as e:
           if isinstance(e, qdrant_client.http.exceptions.UnexpectedResponse ):
              print("coleção não existe, tentando criar uma nova")
            
              if self.__qdrant_create_collection():
                  vector_count:int  = 0
                  print("nova coleção criada")
              else:
                raise Exception("Não foi possível criar uma nova coleção")
            
        print(f"qntd inicial de vetores {vector_count}") # Os IDs dos vetores a serem inseridos correspondem a qntd de vetores já existentes na collection do Qdrant
        #o primeiro vetor foi add com id=0 , o segundo com id=1, então o ID do novo vetor vai ser a qntd de vetores ja existentes
       
        for subject in subjects_and_questions:
            start_amount: int = vector_count #atualiza a qntd de vetores inicial 
            
            entire_text: str = subjects_and_questions[subject]
            questions_and_embedings : dict[str,list[float]] = {chunk : self.__get_openAI_embeddings(chunk) for chunk in self.__text_chunk_splitter(entire_text, self.__CORRECT_ANSWER_STR)}
            
            self.QDclient.upsert( #a função de upsert é chamada apenas uma vez com toda  lista de pontos/vetores
                collection_name= QD_collection,
                points= [ #varios objetos da classe de pontos são instanciadas, cada uma com os keys e values do dicionário de text e embeddings
                    PointStruct(
                      id = idx,
                      vector = questions_and_embedings[question],
                      payload= {"page_content": question, "metadata": {"materia": subject, "ano": test_year}}
                    )
                for idx, question in enumerate(questions_and_embedings , vector_count)  
                ] 
            )   
            vector_count += len(questions_and_embedings) #tamanho do dicionário é o número de questões no arquivo de texto
        
            all_new_questions: int = vector_count - start_amount
            print(f"Tentou inserir {all_new_questions} questões do assunto {subject} no vectorDB") #quantidade de novas questões que eram para ser adicionadas

            final_vector_count: int  = self.QDclient.get_collection(QD_collection).vectors_count #mudança na qntd de vetores presentes no vector db
            questions_added: int = final_vector_count - start_amount
            print(f"Foram inseridas  {questions_added} questões do assunto {subject} no vectorDB, para um total de {final_vector_count} questões")  
           
            vector_count = final_vector_count  #caso não seja possível colocar todos os vetores na collection, a variavel de classe que conta os vetores é corrigida com o real número
           
            if save_extraction_stats: 
                self.__etl_metadata_saving(
                    stats_csv_path=stats_csv_path,
                    current_year= test_year,
                    subject= subject,
                    all_questions= all_new_questions,
                    added_questions= questions_added 
                )  
                
            if all_new_questions !=  questions_added:
               print(f"Não foi possível adicionar todas as questões da matéria {subject} no vectorDB") 
            
            
