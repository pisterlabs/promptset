import pandas as pd
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from config import openai_key
import pickle
import os
import numpy as np
from sklearn.decomposition import PCA


class SearchDataGenerationPipeline():
    def __init__(self, output_dir, latest_semester):
        self.output_dir = output_dir
        self.latest_semester = latest_semester
        openai.api_key = openai_key
        self.embedding_count = 0
        self.prev_embeddings_store = {}
        self.save_prev_embeddings_store_every = 500

    # generate embeddings using the openai api
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding_from_openai(self, text: str, model="text-embedding-ada-002"):
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


    def get_embedding_from_prev_embeddings_store(self, text):
        return self.prev_embeddings_store.get(text, None)

    
    def read_prev_embeddings_store(self):
        with open("prev_embedding_store.pkl", "rb") as f:
            prev_embeddings_store = pickle.load(f)
            self.prev_embeddings_store = prev_embeddings_store

    
    def write_prev_embeddings_store(self):
        with open("prev_embedding_store.pkl", "wb") as f:
            pickle.dump(self.prev_embeddings_store, f)


    def get_embeddings(self, text):
        # try to get it from prev_embedding_store.pkl (if we've already seen this text)
        # if not found, use the openai api, and save it to prev_embedding_store.pkl
        if (embedding := self.prev_embeddings_store.get(text, None)) is None:
            embedding = self.get_embedding_from_openai(text)
            print(f"{self.embedding_count} OpenAI - got embedding for {text}")
            self.prev_embeddings_store[text] = embedding
        else:
            print(f"{self.embedding_count} pkl - got embedding for {text}")

        self.embedding_count += 1
        if self.embedding_count % self.save_prev_embeddings_store_every == 0:
            self.write_prev_embeddings_store()

        return embedding


    def generate_embeddings(self, df):
        df['embeddings'] = df['embedding_text'].apply(self.get_embeddings)
        return df


    def drop_useless_columns(self, df):
        relevant_columns = ['acad_org','description', 'catalog_nbr', 'class_nbr', 'class_section', 'subject', 'subject_descr', 'descr', 'units', 'acad_career_descr', 'strm', 'topic']
        return df[relevant_columns]


    def add_topic_to_descr(self, row):
        if not pd.isna(row['topic']):
            return row['descr'] + ' - ' + row['topic']
        else:
            return row['descr']

    def preprocess_classes_with_topics(self, df):
        topic_class_map = {}
        df['descr'] = df.apply(lambda row: self.add_topic_to_descr(row), axis=1)
        df = df[~((df['topic'].notna()) & (df['strm'] != self.latest_semester))]
        df = df[~((df['topic'].notna()) & (df['units'] == '0'))] # filter out discussion sections with zero units of special topic classes
        
        # Identify rows with the same 'catalog_nbr' and 'subject' but different 'topic' (implying multiple special topic offerings)
        duplicate_rows = df[df.duplicated(subset=['catalog_nbr', 'subject'], keep=False) & df['topic'].notna()]
        
        # Create a dictionary to keep track of the counts of each (subject, catalog_nbr) combo
        subject_catalog_count = {}

        # keep count of how many sessions are present for a given special topic class
        for index, row in duplicate_rows.iterrows():
            key = (str(row['subject']), str(row['catalog_nbr']))
            if key not in subject_catalog_count:
                subject_catalog_count[key] = 1
            else:
                subject_catalog_count[key] += 1
                
        # Modify duplicate rows to have unique 'catalog_nbr'
        for index, row in duplicate_rows.iterrows():
            key = (str(row['subject']), str(row['catalog_nbr']))
            # only modify the catalog_nbr if there are multiple sections that we need to differentiate from
            if subject_catalog_count[key] > 1:
                new_catalog_nbr = f"{row['catalog_nbr']}.{row['class_section']}"
                key = (str(row['subject']), str(row['catalog_nbr']))
                new_val = (str(row['subject']), str(new_catalog_nbr))
                
                # add it to topic_class_map
                arr = topic_class_map.get(key, [])
                arr.append(new_val)
                topic_class_map[key] = arr

                df.at[index, 'catalog_nbr'] = new_catalog_nbr
        
        with open(os.path.join(self.output_dir, 'topic_class_map.pkl'), 'wb') as f:
            pickle.dump(topic_class_map, f)
        return df, topic_class_map
        

    # Function to keep one row per group based on 'units' or string value
    def keep_one_row(self, group):
        try:
            # Attempt to convert 'units' to integers
            group['units'] = group['units'].astype(float)
        except ValueError:
            # If conversion to int fails, 'units' remains unchanged (string)
            pass

        if any(isinstance(val, str) for val in group['units']):
            return group[group['units'].apply(lambda x: isinstance(x, str))]
        else:
            return group[group['units'] == group['units'].max()]


    def filter_duplicated_classes(self, df):
        df = df.sort_values(by='strm', ascending=False)
        result_df = df.groupby(['catalog_nbr', 'subject'], group_keys=False).apply(self.keep_one_row).reset_index(drop=True)
        result_df = result_df.drop_duplicates(subset=['catalog_nbr', 'subject'])
        return result_df


    def generate_embedding_text_for_classes_with_topics(self, row):
        return f"{row['subject']} {row['catalog_nbr']} {row['descr']} . {row['topic']}. {row['description']}"


    def generate_embedding_text_for_classes_without_topics(self, row):
        return row['subject_descr'] + ' - ' + row['descr'] + '. ' + row['description']

    
    def generate_embedding_text(self, df):
        df['embedding_text'] = df.apply(lambda row: self.generate_embedding_text_for_classes_with_topics(row) if not pd.isna(row['topic']) else self.generate_embedding_text_for_classes_without_topics(row), axis=1)
        return df


    # end of pipeline stuff
    def generate_pca_data(self):
        with open(os.path.join(self.output_dir, 'embedding_matrix_32.pkl'), 'rb') as f:
            embedding_matrix = pickle.load(f)
        
        pca = PCA(n_components=3)
        pca.fit(embedding_matrix)
        transformed_coordinates = pca_data = pca.transform(embedding_matrix)

        # save pca
        with open(os.path.join(self.output_dir, 'pca.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        # save transformed coordinates
        with open(os.path.join(self.output_dir, 'pca_transformed_coords.pkl'), 'wb') as f:
            pickle.dump(transformed_coordinates, f)


    def generate_index_to_data_dict(self, df):
        # index to data dict
        index_to_data_dict = {}
        for index, row in df.iterrows():
            index_to_data_dict[index] = {
                'level': row['acad_career_descr'],
                'catalog_number': row['catalog_nbr'],
                'class_number': row['class_nbr'],
                'subject': row['subject_descr'],
                'name': row['descr'],
                'credits': row['units'],
                'description': row['description'],
                'mnemonic': row['subject'],
                'group': row['acad_org'],
                'strm': row['strm']}

        with open(os.path.join(self.output_dir, 'index_to_data_dict.pkl'), 'wb') as f:
            pickle.dump(index_to_data_dict, f)

        return index_to_data_dict


    def generate_data_to_index_dict(self):
        # data to index dict
        data_to_index = {}

        with open(os.path.join(self.output_dir, 'index_to_data_dict.pkl'), 'rb') as f:
            index_to_data_dict = pickle.load(f)
        
        for index in index_to_data_dict.keys():
            data = index_to_data_dict[index]
            mnemonic_number_tuple = (data['mnemonic'], data['catalog_number'])
            data_to_index[mnemonic_number_tuple] = index

        with open(os.path.join(self.output_dir, 'data_to_index_dict.pkl'), 'wb') as f:
            pickle.dump(data_to_index, f)

        
    def generate_filter_indices(self, df):
        acad_level_to_arr_map = {
            "Undergraduate": set(),
            "Graduate": set(),
            "Law": set(),
            "Graduate Business": set(),
            "Medical School": set(),
            "Non-Credit": set()}

        latest_sem_indices = set()
        for index, row in df.iterrows():
            acad_level_to_arr_map[row["acad_career_descr"]].add(index)
            if row['strm'] == self.latest_semester:
                latest_sem_indices.add(index)
        
        for key in acad_level_to_arr_map:
            filename = os.path.join(self.output_dir, key + "_indices.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(acad_level_to_arr_map[key], f)

        with open(os.path.join(self.output_dir, 'latest_sem_indices.pkl'), 'wb') as f:
            pickle.dump(latest_sem_indices, f)


    def generate_embedding_matrix(self, df):
        embedding_vector_strings = df['embeddings']
        embedding_matrix = np.array([np.array(eval(embedding)) for embedding in embedding_vector_strings])
        # print(f"Generated embedding matrix of shape: {embeddings.shape}")

        # save the embedding matrix
        with open(os.path.join(self.output_dir, 'embedding_matrix_64.pkl'), 'wb') as f:
            pickle.dump(embedding_matrix, f)

        embedding_matrix = embedding_matrix.astype(np.float32)  # cast to float32 to save memory
        with open(os.path.join(self.output_dir, 'embedding_matrix_32.pkl'), 'wb') as f:
            pickle.dump(embedding_matrix, f)
        return embedding_matrix
    
    def cast_to_str(self, val):
        if isinstance(val, str):
            return val
        else:
            return str(val)


    def run(self, df, output_dir, latest_semester):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Preprocessing data...")
        df = self.drop_useless_columns(df)
        # cast the catalog_nbr to a string
        df['catalog_nbr'] = df['catalog_nbr'].apply(self.cast_to_str)
        df, _ = self.preprocess_classes_with_topics(df)
        df = self.filter_duplicated_classes(df)
        self.read_prev_embeddings_store()   # load previous embedding dict
        df = self.generate_embedding_text(df)

        print("Generating embeddings...")
        df = self.generate_embeddings(df)

        self.write_prev_embeddings_store()  # save previous embedding dict
        df.to_csv(os.path.join(self.output_dir, 'data.csv'), index=False)

        # save the dataframe and load it back up to standardize embedding format
        df.to_csv(os.path.join(self.output_dir, 'data.csv'), index=False)
        df = pd.read_csv(os.path.join(self.output_dir, 'data.csv'))

        print("Generating output files")
        # end of pipeline stuff
        self.generate_embedding_matrix(df)
        self.generate_pca_data()
        self.generate_index_to_data_dict(df)
        self.generate_data_to_index_dict()
        self.generate_filter_indices(df)



if __name__ == "__main__":
    output_dir = "data_pipeline_output/"
    latest_semester = 1242
    df = pd.read_csv("prev_semester_data_with_descriptions.csv")

    pipeline = SearchDataGenerationPipeline(output_dir, latest_semester)
    pipeline.run(df, output_dir, latest_semester)