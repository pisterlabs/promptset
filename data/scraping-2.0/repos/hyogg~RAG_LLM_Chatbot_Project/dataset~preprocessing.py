from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizerFast
from flatten_json import flatten

import tiktoken
import pandas as pd
import numpy as np
import json
import csv
import glob

class BertTokenSplitter :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl", encoding_name: str = "klue/bert-base") :
        '''
            Args:
                jsonl_input_path: Path to the JSONL file containing the original crawling data
                jsonl_output_path: Path to save the file
                encoding_name: BERT model name used for embedding
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path
        self.encoding_name = encoding_name

    def load_data(self):
        data_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(line)
        return data_list

    def bert_split(self, data):
        '''
            Desc:
                If the description in the original crawl file (.jsonl) sentence exceeds 380 tokens (based on BERT),
                split the data.
            Args:
                data : Crawling file
                jsonl_output_path
        '''
        encoding_name = self.encoding_name
        tokenizer = BertTokenizerFast.from_pretrained(encoding_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(chunk_size=380, chunk_overlap=50, tokenizer=tokenizer, separators=["한편", "이 밖에도", ". "], keep_separator=False)
        
        # TODO: The key must be 'description' -> Raises KeyError if not present
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file :
            for line in data:
                try :
                    text_dict = json.loads(line)
                    text = text_dict['description']
                    # print(text)
                except KeyError:
                    raise KeyError("작품 설명의 key값을 'description'으로 설정하세요.")

                split_data = splitter.split_text(text)
                for single_split_data in split_data :
                    # print(f'split... -> {single_split_data}')
                    new_line = text_dict
                    new_line['description'] = single_split_data
                    print(f'new_line: {new_line}')
                    json.dump(new_line, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
    
    def bert_len_check(self, text: str = None):
        len_list = []
        over_list = []
        if text is None:
            with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
                for line in file:
                    tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)
                    tokens = tokenizer(line, return_tensors='pt')['input_ids']
                    line = json.loads(line)
                    if tokens.size(1) > 450 :
                        len_list.append({line['title']:tokens.size(1)})
                    print(f'"{line["title"]}" 하는 중.. Token Length : {tokens.size(1)}')
            print(len_list)
        else :
            tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)
            tokens = tokenizer(text, return_tensors='pt')['input_ids']
            print(f'해당 Text의 Token Length : {tokens.size(1)}')
        return tokens.size(1)

    def split_process(self):
        print('Running bert_split')
        data = self.load_data()
        # print(data[1])
        self.bert_split(data)


class GptTokenSplitter :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl", encoding_name: str = "cl100k_base") :
        '''
            Args:
                jsonl_input_path: Path to the JSONL file containing the original crawling data
                jsonl_output_path: Path to save the file
                encoding_name: 
                    - gpt-3.5-turbo, gpt-4, text-embedding-ada-002 : "cl100k_base" (default)
                    - Codex models, text-davinci-002, text-davinci-003 : "p50k_base"
                    - GPT-3 models like "davinci" : "r50k_base" or "gpt2"
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path
        self.encoding_name = encoding_name

    def load_data(self):
        data_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(line)
        return data_list

    def tiktoken_len(self, text):
        '''
            Desc:
                Check the length of received text using tiktokens.
            Args:
                1. text
                2. encoding_name :
                    - cl100k_base : gpt-3.5-turbo, gpt-4, text-embedding-ada-002 모델 사용시
                    - p50k_base : text-davinci-002, text-davinci-003 모델 사용시
                    - r50k_base (or gpt2) : GPT-3 models like "davinci"
        '''
        tokenizer = tiktoken.get_encoding(self.encoding_name)
        # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # 위 코드로 안될 시 모델 이름 넣어서 이걸로 해보기
        tokens = tokenizer.encode(text)
        return len(tokens)

    def tiktoken_split(self, data):
        '''
            Desc:
                If the description in the original crawl file (.jsonl) sentence exceeds 2000 tokens (based on cl100k_base),
                split the data.
            Args:
                data : Crawling file
        '''
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)

        # TODO: The key must be 'description' -> Raises KeyError if not present
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file :
            for line in data:
                try :
                    text_dict = json.loads(line)
                    text = text_dict['description']
                except KeyError:
                    raise KeyError("작품 설명의 key값을 'description'으로 설정하세요.")
                token_len = self.tiktoken_len(text)

                # If the token length exceeds 2000, split the document
                if token_len > 2000 :
                    split_data = splitter.split_text(text)
                    for single_split_data in split_data :
                        # print(f'split... -> {single_split_data}')
                        new_line = text_dict
                        new_line['description'] = single_split_data
                        json.dump(new_line, jsonl_file, ensure_ascii=False)
                        jsonl_file.write('\n')
                else :
                    json.dump(text_dict, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
    
    def tiktoken_len_check(self):
        '''
        To check the length of a JSONL file using tiktokens
        '''
        len_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokenizer = tiktoken.get_encoding(self.encoding_name)
                tokens = tokenizer.encode(line)
                line = json.loads(line)
                len_list.append({len(tokens):line['title']})
        print(len_list)
        return len(tokens)

    def split_process(self):
        print('Running tiktoken_split')
        data = self.load_data()
        self.tiktoken_split(data)


class ConvertFile:
    ''' 
        Convert file extension
    '''
    def __init__(self):
        pass

    def jsonl_to_csv(self, input_path: str = "./files/museum_passage.jsonl", output_path: str = './files/dataset.csv', headers: list[str] = ['title', 'link', 'era', 'info', 'description']) :
        # Input format -> JSONL file without question data
        try :
            with open (output_path, 'w', newline='', encoding='utf-8-sig') as c :
                csvwriter = csv.writer(c)
                csvwriter.writerow(headers)

                with open(input_path, 'r', encoding='utf-8') as f :
                    for line in f :
                        print('Processing line:', line)
                        data = json.loads(line)
                        row = flatten(data)
                        csvwriter.writerow([row[header] for header in headers])
            print('Conversion completed successfully.')
            input('Press Enter to exit...')
        except Exception as e :
            print(f'Convert failed... -> {e}')
    
    # Input format -> CSV file
    def csv_to_jsonl(self, input_path: str = "./files/split_df_modified.csv", output_path: str = './files/converted_jsonl.jsonl') :
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        print(f'CSV > JSONL Convert Success!')

class PreprocessingPassage:
    '''
        Passages are generated by this class
        for utilization in crafting QA datasets
        through the facilitation of GPT API.
    '''
    def __init__(self):
        pass
    
    def extract_col(self, jsonl_input_path: str = './files/passage_bert_split.jsonl', csv_output_path: str = None, jsonl_output_path: str = None):
        '''
            Extracts 'era', 'size', 'property', and 'collection number' from the columns of the input JSONL file,
            insert into 'description' column, and saves them as CSV and JSONL files.

            Returns :
                df = pd.DataFrame([{'title':'...', 'description':'국적/시대:era, 크기:size, 문화재구분:property, 소장품번호:num'},
                                    'title':'...', 'description':'..original first desc..'}])
        '''
        data_list = []
        jsonl_input_path = './files/passage_bert_split.jsonl'
        with open(jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                data_list.append({
                    'title': json_data['title'],
                    'era': json_data['era'],
                    'info': json_data['info'],
                    'description': json_data['description']
                })
        df = pd.DataFrame(data_list)

        # info column split
        info_df = df['info'].str.split('\\n', expand=True)
        info_df.columns = ['size', 'property', 'num']

        # replace if 'num' is empty and 'property' is not empty
        mask = (info_df['num'].isna()) & (info_df['property'].notna())
        info_df.loc[mask, 'num'] = info_df['property']
        info_df.loc[mask, 'property'] = None

        # Merge with the source dataframe
        df = pd.concat([df, info_df], axis=1)
        df.fillna('정보없음', inplace=True)

        # ['era', 'size', 'property', 'number'] only needs to be present once per title
        info_df = df.groupby('title').first().reset_index()
        info_df['new_desc'] = info_df.apply(lambda row: f"{row.title} -> 국적/시대:{row.era}, 크기:{row.size}, 문화재구분:{row.property}, 소장품번호:{row.num}", axis=1)
        info_df['description'] = info_df['new_desc']
        info_df.drop(columns='new_desc', axis=1, inplace=True)

        # Append info_df to the bottom row of the source df
        df = pd.concat([df, info_df])
        df = df.reset_index(drop=True)

        # Delete everything except title, desc
        df.drop(columns=['era', 'info', 'size', 'num', 'property'], axis=1, inplace=True)

        # Process output to CSV
        if csv_output_path:
            df.to_csv('./files/title_desc_passage.csv', index=False, encoding='utf-8-sig')

        # Process output to JSONL
        if jsonl_output_path:
            df.to_json('./files/title_desc_passage.jsonl',lines=True, orient='records', force_ascii=False)

        return df

    # Note: extract_info_col() method (above) parses all of era, size, info_num, and info_property.
    #       This code below parses into only two categories: size and info.
    def extract_era_info(self, jsonl_input_path: str = './files/passage_bert_split.jsonl', csv_output_path: str = None, jsonl_output_path: str = None):
        '''
            ! The order of fields must be [title, era, info, description]
            Add "era" and "info" values to the "description" fields and delete the corresponding fields.
                
            Returns :
                df = pd.DataFrame([{'title':'...', 'description':'국적/시대: era, 정보: info'},
                                    'title':'...', 'description':'..original first desc..'}])
        '''
        if self.input_path :
            json_input_path = self.input_path
            result_list = []
            title_set = set()  # To track the already added combined_text, create a title set
            try :
                with open(json_input_path, 'r', encoding='utf-8') as file:            
                    for line in file:
                        text_dict = json.loads(line)
                        keys_in_order = list(text_dict.keys())
                        title_key, era_key, info_key, description_key = keys_in_order[:4]
                        try :
                            title = text_dict[title_key]
                            era = text_dict[era_key]
                            info = text_dict[info_key]
                            description = text_dict[description_key]
                        except KeyError as e:
                            raise KeyError(f"KeyError: {e}. 필수 키가 없습니다.")
                        text_dict = {'title':title, 'description':description}
                        # Create a new field in "description" for combined_text
                        combined_text = f"국적/시대:{era}, 정보:{info}"
                        # Check if "era" and "info" are already in the set, only add combined_text to "description" if not added
                        if title not in title_set:
                            text_dict_with_combined = text_dict.copy()
                            text_dict_with_combined['description'] = combined_text
                            result_list.append(text_dict_with_combined)
                            # Also include the original description
                            result_list.append(text_dict)
                            title_set.add(title)
                        else : 
                            result_list.append(text_dict)
                df = pd.DataFrame(result_list)

                # Process output to CSV
                if csv_output_path:
                    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

                # Process output to JSONL
                if jsonl_output_path:
                    df.to_json(jsonl_output_path,lines=True, orient='records', force_ascii=False)

            except Exception as ee :
                print('오류 -> {ee}')


class PreprocessingQuestion:
    '''
        Preprocessing class designed to split and preprocess QA datasets obtained through GPT API.
    '''
    def __init__(self):
        pass

    def concat_from_files(self, file_pattern: str = './files/df_*.csv', output_csv_path: str = './files/concat_df.csv'):
        '''
            Desc:
                Loads multiple DataFrames from the specified file paths and concatenates them.

            Args:
                - file_pattern: Pattern specifying the files to be loaded.
                - output_csv_path: PATH to save the CSV file

        '''

        # Use glob to find files matching the pattern
        files = glob.glob(file_pattern)
        files = sorted(files)

        dfs = []
        for file_path in files:
            try :
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as ee :
                print(ee)
        concat_df = pd.concat(dfs, ignore_index=True)
        concat_df.to_csv(output_csv_path, encoding='utf-8-sig')


    def format_and_split(self, input_csv_path: str = "./files/concat_df.csv", output_csv_path: str = "./files/split_df.csv"):
        '''
            Desc:
                Load the concatenated DataFrame from a CSV file, assign ctx_id and tit_id,
                and separate the columns for questions and answers.
            
            Args:
                input_csv_path: Path to the CSV file containing the concatenated DataFrame.
                output_csv_path: Path to save the CSV file with split DataFrame.
            
        '''
        df = pd.read_csv('./files/concat_df.csv')
        df['ctx_id'] = df.index
        df['tit_id'] = df.groupby('title').ngroup()
        df = df[['tit_id', 'ctx_id', 'title', 'context', 'question']]
        q_expand = df['question'].str.split('질문:', expand=True)
        new_df = pd.concat([df, q_expand], axis=1)
        new_df.drop(columns=['question', 0], inplace=True)
        
        new_list = []
        for index, row in new_df.iterrows() :
            for col_name in new_df.columns[4:]:
                if pd.notna(row[col_name]):
                    new_row = {'tit_id':row['tit_id'], 'ctx_id':row['ctx_id'], 'title':row['title'], 'context':row['context'], 'qa':row[col_name]}
                    new_list.append(new_row)
        new_list_df = pd.DataFrame(new_list)
        
        a_expand = new_list_df['qa'].str.split('답변:|답변 :|답:|답 :', regex=True, expand=True)
        replace_dict = {' : ': '', ': ':'', ' :':'', ':': '', ' \n ': '', ' \n':'', '\n ':'', '\n':''}
        a_expand = a_expand.apply(lambda x: x.replace(replace_dict, regex=True))
        
        df = pd.concat([new_list_df, a_expand], axis=1)
        df.drop(columns='qa', inplace=True)
        df.rename(columns={0:'question', 1:'answer'}, inplace=True)

        # Remove leading and trailing ',', '.', and ' '
        df['question'] = df['question'].str.replace('^[,.\s]+', '', regex=True)
        df['answer'] = df['answer'].str.replace('^[,.\s]+', '', regex=True)
        df['context'] = df['context'].str.replace('^[,.\s]+', '', regex=True)

        df['question'] = df['question'].str.replace('[,.\s]+$', '', regex=True)
        df['answer'] = df['answer'].str.replace('[,.\s]+$', '', regex=True)
        df['context'] = df['context'].str.replace('[,.\s]+$', '', regex=True)

        # Remove rows where the answer is Na
        df.dropna(subset='answer', inplace=True)
        df.to_csv('./files/split_df.csv', encoding='utf-8-sig', index=False)

    def train_test_split(
            self,
            input_path : str = "./files/dataset_ver1_1205.csv",
            train_output_path : str = "./files/train.csv", 
            test_output_path : str = "./files/test.csv",
            test_size=0.2,
            random_state=10
        ):

        df = pd.read_csv(input_path, encoding='utf-8-sig')
        # Calculate the number of test and train data
        test_num = int(df.shape[0] * test_size)
        train_num = df.shape[0] - test_num

        # Extract test data with ctx_id having more than 2 (to preserve passages)
        test_except_ctx = df['ctx_id'].value_counts()[df['ctx_id'].value_counts() < 2].index.tolist()
        test_except_index = df[df['ctx_id'].isin(test_except_ctx)].index.tolist()
        test_indices = [i for i in range(df.shape[0]) if i not in test_except_index]
        
        # Random shuffle and data split
        np.random.seed(random_state)
        np.random.shuffle(test_indices)
        test_split_df = df.iloc[test_indices[:test_num]]
        train_split_df = df.iloc[~df.index.isin(test_split_df.index)]

        # Check for duplicate ctx_id in train and test data
        testidx = test['ctx_id'].value_counts().index.tolist()
        trainidx = test['ctx_id'].value_counts().index.tolist()
        all_include_check = all(item in trainidx for item in testidx)
        if all_include_check :
            print('Successfully extracted Test data with Contexts having more than 2 questions.')
        else :
            print('Contexts in the Test dataset are not present in the Train dataset. Please create the dataset again.')

        train_split_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
        test_split_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')