# file lib_OpenAI_Embeddings
import pandas as pd

import openai

from scipy import spatial  # for calculating vector similarities for search
from ast import literal_eval # Function used to cast embedding vectors stored as strings to arrays
from nl2sql.OpenAI_Func import Num_Tokens_From_String, OpenAI_Embeddings_Cost, OpenAI_Embeddings_Cost

#############################################################################
class OpenAI_Embeddings():
    def __init__(self, Filename, Encoding_Base, Model, Token_Cost, Max_Tokens):
        self._DBFilename = Filename
        self._Encoding_Base = Encoding_Base
        self._Model = Model
        self._Token_Cost = Token_Cost
        self._Max_Tokens = Max_Tokens
        self._VDS_DF = pd.DataFrame()
        self._Embedding = []

#############################################################################
# OpenAI Embeddings - returns list
    def OpenAI_Get_Embedding(self, Text='', Verbose=False):
        # replace line return
        if Text != '':
            Text = Text.replace("\n", " ")
            # check if tokens is under max tokens for model
            ntokens = Num_Tokens_From_String(Text, self._Encoding_Base)
            if ntokens < self._Max_Tokens:  
                try:
                    #Make your OpenAI API request here
                    response= openai.Embedding.create(input=[Text],model=self._Model)
                except openai.error.APIError as e:
                    #Handle API error here, e.g. retry or log
                    print(f"OpenAI API returned an API Error: {e}")
                    return []
                except openai.error.APIConnectionError as e:
                    #Handle connection error here
                    print(f"Failed to connect to OpenAI API: {e}")
                    return []
                except openai.error.RateLimitError as e:
                    #Handle rate limit error (we recommend using exponential backoff)
                    print(f"OpenAI API request exceeded rate limit: {e}")
                    return []
                
                embeddings = response['data'][0]['embedding']
                cost, tokens = OpenAI_Embeddings_Cost(response, self._Token_Cost, self._Model)
                if Verbose:
                    print(f'Embeddings Cost {cost} and tokens {tokens}')
                return embeddings

#############################################################################
# Retrieve Embeddings for DF columne -- assumes the existance of a Question column 
    def Get_Embeddings_DF_Column(self, df, Embedding_LLM = 'OpenAI', Verbose=False):
        if Embedding_LLM=='OpenAI':
            df['Embedding'] = df.Question.apply(lambda x: self.OpenAI_Get_Embedding(x))
        return 0

#############################################################################
# Retrieve Embeddings for DF column -- assumes the existance of a Question column 
    def Get_Embeddings_DF(self,Verbose=False):
        self.Get_VDS_Embeddings_DF_Column(self._VDS_DF, Embedding_LLM = 'OpenAI', Verbose=Verbose)
        return 0

#############################################################################
# Calculate Embeddings for DF assumes      
class VDS(OpenAI_Embeddings):
    def Load_VDS_DF(self, sep='|', Sheetname = 'VDS', Verbose=False, Debug=False):
        Suffix = self._DBFilename[-3:]
        if Suffix == 'txt':
            try:
                df = pd.read_csv(self._DBFilename, sep=sep )

                # convert embeddings from CSV str type back to list type
                df['Embedding'] = df['Embedding'].apply(literal_eval)
                self._VDS_DF = df
                if Verbose:
                    print(f'Load_VDS_DF imported {df.shape[0]} rows from {self._DBFilename}')
                return 0
            except:
                print(f'Load_VDS_DF Error failed to import file {self._DBFilename}')
                return -1
        elif Suffix == 'lsx':                  
            try:
                df = pd.read_excel(self._DBFilename, sheet_name=Sheetname)
                self._VDS_DF = df
                if Verbose:
                    print(f'Load_VDS_DF imported {df.shape[0]} rows from {self._DBFilename}')
                return 0
            except:
                print(f'Load_VDS_DF Error failed to import file {self._DBFilename}')
                return -1
        else:
            print(f'Load_VDS_DF: Unsupported Filetype {self._DBFilename}')
        return 0
        # for version 1, import dataframe
        


#############################################################################
#   
    def Store_VDS_DF(self, Format = 'txt', Delimator = '|', Increment_Filename=True, Verbose=False):
        # strip the suffix, e.g. .csv, .xlsx, assume filename is given as prefix-i.suffix
        if Increment_Filename:
            Filename_list = self._DBFilename.split("-")
            Prefix = Filename_list[0]
            Suffix_list = Filename_list[1].split(".")
            i = int(Suffix_list[0])
            # increment counter, i
            i = str(i + 1)
            # Filename = prefix + i (suffix is added below)
            Filename = Prefix + "-" + i
        else:
            Suffix = self._DBFilename[-4:].replace(".","")
            if Suffix == 'csv':
                if Delimator is None:
                    Delimator = ","  
                Filename = self._DBFilename[0:len(self._DBFilename)-4]
            if Suffix == 'txt':
                if Delimator is None:
                    Delimator= "|"  
                Filename = self._DBFilename[0:len(self._DBFilename)-4]
            elif Suffix == 'xslx':              
                Filename = self._DBFilename[0:len(self._DBFilename)-5]
            else:
                print(f'Store_VDS_DF: Unsupported Filetype {Suffix}')
                return 0
            
        # Convert embedding vector to n columns in dataframe before saving
        if Format == 'xlsx':
            Filename = Filename + ".xlsx"
            try:
                self._VDS_DF.to_excel(Filename,sheet_name='VDS',index=False, header=True)
                if Verbose:
                    print(f'Store_VDS_DF() wrote {self._VDS_DF.shape[0]} rows to file {Filename}')
                return 0
            except:
                print(f'Store_VDS_DF Error failed to write to {Filename}')
                return -1
        elif Format == 'csv':
            Filename = Filename + ".csv"
            self._VDS_DF.to_csv(Filename,header=True, index=False, sep = Delimator) 

        elif Format == 'txt':
            Filename = Filename + ".txt"
            self._VDS_DF.to_csv(Filename,header=True, index=False, sep = Delimator) 

        else:
            print(f"Store_VDS_DF: File format {Format} is not supported")
                                  
# Insert row into Embeddings DF
    def Insert_VDS(self, Question, Query, Metadata, Embedding=[], Verbose=False):
        #remove CR
        Query = Query.replace("\n", "")
        # tmp dataframe
        df_tmp = pd.DataFrame({'Question':Question,'Query':Query, 'Metadata':Metadata, 'Embedding':[Embedding]})
        # append to self._VDS_DF
        self._VDS_DF = pd.concat([self._VDS_DF,df_tmp], ignore_index=True)
        if Verbose:
            print(self._VDS_DF[['Question','Query']])
       # df = df.reset_index()
        # write DF to file
        self.Store_VDS_DF(Format='txt',Delimator="|",Increment_Filename=True, Verbose=False)

        return 0
    
    def Retrieve_Embeddings_DF_Column(self, Verbose=False):
        # Assumption Vector Datastore is a pandas dataframe
        if Verbose:
            print("Retrieve_Embeddings_DF_Column: Get_VDS_Embeddings_DF")
        rtn = self.Get_Embeddings_DF_Column(df=self._VDS_DF, Embedding_LLM = 'OpenAI', Verbose=True)
       # rtn = self.Get_Embeddings_DF
        if Verbose:
            print(f'Retrieve_Embeddings_DF_Column: after embedding call {self._VDS_DF.head(2)}')        

        if Verbose:
            print("Retrieve_Embeddings_DF_Column: Store_VDS_DF")
        rtn = self.Store_VDS_DF(Format='Pipe',Verbose=True)

        return 0
    
    """ Assume Question is the embedding representation of the data"""
    def Search_VDS(self, Question, Similarity_Func = 'Cosine', Top_n=1, Debug=False) \
         -> tuple[list[int], list[str], list[str], list[float]]:
        if Similarity_Func == 'Cosine':
            f = lambda x,y: 1 - spatial.distance.cosine(x, y)

        # Calculate similarity measure for reach question
        Similar_Questions = [
            ( i, r['Question'], r['Query'],f(Question,r["Embedding"]))
                for i, r in self._VDS_DF.iterrows()
            ]
        
        # need a default value to insert if no queries match

        # filter top N most similar questions
        # x[3] -> f(Question,r["Embedding"])    
        Similar_Questions.sort(key=lambda x: x[3], reverse=True)
        if Debug:
            print(f'VDS {self._VDS_DF}')
            print(f'Similar_Questions {Similar_Questions}')
        Idx, Questions, Queries, Simlarirty = zip(*Similar_Questions)
        return Idx[:Top_n], Questions[:Top_n], Queries[:Top_n], Simlarirty[:Top_n]




################
## search embeddings for similar queries
    # def cosine_similarity(self, x:list(),y:list()) -> float:
    #     return 1 - spatial.distance.cosine(x, y)
    
    # def Search_by_Similarity_df(self, query: str, df: pd.DataFrame, similarity_func=self.cosine_similarity(x,y),
    #     top_n: int = 100 ) -> tuple[list[str], list[float]]:
    #     """Returns a list of strings and relatednesses, sorted from most related to least."""
   
    #     strings_and_relatednesses = [
    #         (row["text"], similarity_func(query_embedding, row["embedding"]))
    #         for i, row in df.iterrows()
    #     ]
    #     strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    #     strings, relatednesses = zip(*strings_and_relatednesses)
    #     return strings[:top_n], relatednesses[:top_n]
    

