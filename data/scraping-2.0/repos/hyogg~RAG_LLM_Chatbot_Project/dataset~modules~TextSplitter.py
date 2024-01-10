from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import BertTokenizerFast
import tiktoken
import json

def load_jsonl_data(file_path):
    """
        Args:
            file_path (str): JSONL 파일 경로

        Returns:
            list: JSONL 파일에서 로드된 데이터 목록

        Raises:
            FileNotFoundError: 지정된 파일 경로가 존재하지 않을 때 발생합니다.
            IOError: 파일 읽기 중 오류가 발생했을 때 발생합니다.
    """
    try:
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line))
        return data_list
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")
    except IOError as e:
        raise IOError(f"{file_path} 파일 읽기 중 오류가 발생했습니다: {e}")

class BertTokenSplitter :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl", encoding_name: str = "klue/bert-base") :
        '''
        Args:
            jsonl_input_path: 원본 데이터가 있는 JSONL 파일의 경로
            jsonl_output_path: 분할된 데이터를 저장할 JSONL 파일의 경로
            encoding_name: BERT 모델 이름
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path
        self.encoding_name = encoding_name

    def bert_split(self, data):
        """
        데이터를 로드하고, BERT 토크나이저를 사용하여 텍스트를 분할한 후 결과를 JSONL 파일로 저장합니다.
        """
        data = load_jsonl_data(self.jsonl_input_path)
        tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)
        splitter = CharacterTextSplitter.from_huggingface_tokenizer(chunk_size=380, chunk_overlap=50, tokenizer=tokenizer, separators=["한편", "이 밖에도", ". "], keep_separator=False)
        
        # The key must be 'description' -> Raises KeyError if not present
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file:
            for text_dict in data:
                try:
                    text = text_dict['description']
                except KeyError:
                    raise KeyError("JSON 데이터에서 'description' 키를 찾을 수 없습니다.")

                split_data = splitter.split_text(text)
                for single_split_data in split_data:
                    new_line = text_dict.copy()
                    new_line['description'] = single_split_data
                    json.dump(new_line, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')               
    
    def bert_len_check(self, text: str = None):
        """
        제공된 텍스트의 BERT 토큰 길이를 검사합니다.
        Args:
            text (str, optional): 검사할 텍스트. None이면 jsonl_input_path에서 데이터를 로드하여 검사합니다.
        Returns:
            int: 토큰 길이
        """
        tokenizer = BertTokenizerFast.from_pretrained(self.encoding_name)

        if text is None:
            data = load_jsonl_data(self.jsonl_input_path)
            for text_dict in data:
                tokens = tokenizer(text_dict['description'], return_tensors='pt')['input_ids']
                print(f'"{text_dict["title"]}" 토큰 길이: {tokens.size(1)}')
        else :
            tokens = tokenizer(text, return_tensors='pt')['input_ids']
            print(f'해당 텍스트의 토큰 길이: {tokens.size(1)}')

        return tokens.size(1)

class GptTokenSplitter :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl", encoding_name: str = "cl100k_base") :
        '''
            Args:
                jsonl_input_path: 원본 데이터가 있는 JSONL 크롤링 파일의 경로
                jsonl_output_path: 분할된 데이터를 저장할 JSONL 파일의 경로
                encoding_name: 토큰화에 사용할 모델 이름
                    - gpt-3.5-turbo, gpt-4, text-embedding-ada-002 : "cl100k_base" (default)
                    - Codex models, text-davinci-002, text-davinci-003 : "p50k_base"
                    - GPT-3 models like "davinci" : "r50k_base" or "gpt2"
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path
        self.encoding_name = encoding_name

    def tiktoken_split(self, data):
        '''
        데이터를 로드하고, GPT 기반 토크나이저를 사용하여 텍스트를 분할한 후 결과를 JSONL 파일로 저장
        '''
        data = load_jsonl_data(self.jsonl_input_path)
        tokenizer = tiktoken.get_encoding(self.encoding_name)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)

        # The key must be 'description' -> Raises KeyError if not present
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file :
            for text_dict in data:
                try:
                    text = text_dict['description']
                except KeyError:
                    raise KeyError("JSON 데이터에서 'description' 키를 찾을 수 없습니다.")
                token_len = len(tokenizer.encode(text))

                if token_len > 2000:
                    split_data = splitter.split_text(text)
                    for single_split_data in split_data:
                        new_line = text_dict.copy()
                        new_line['description'] = single_split_data
                        json.dump(new_line, jsonl_file, ensure_ascii=False)
                        jsonl_file.write('\n')
                else:
                    json.dump(text_dict, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
    
    def tiktoken_len_check(self):
        '''
        JSONL 파일에 있는 모든 텍스트의 GPT 토큰 길이를 검사합니다.
        '''
        data = load_jsonl_data(self.jsonl_input_path)
        tokenizer = tiktoken.get_encoding(self.encoding_name)
        for text_dict in data:
            tokens = tokenizer.encode(text_dict['description'])
            print(f'"{text_dict["title"]}" 토큰 길이: {len(tokens)}')
        return len(tokens)