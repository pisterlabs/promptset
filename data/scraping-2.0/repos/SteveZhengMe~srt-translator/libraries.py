import datetime
import errno
import os
import time
import openai
import deepl
import srt

class TranslatorBase:
    def is_available(self):
        # Default to True
        return True
    
    def get_usage(self):
        # unlimited usage
        return (-1,-1)


class DeepLUtil(TranslatorBase):
    def __init__(self, conf:dict):
        self.conf = conf
        self.translator = deepl.Translator(self.conf["deepl_key"])
        self.limit = self.translator.get_usage().character.limit-1000
        self.current_count = self.translator.get_usage().character.count
        self.last_count_check = self.current_count
    
    def is_available(self) -> bool:
        # sync the usage every 10000 characters
        if (self.current_count - self.last_count_check > 10000) or (self.current_count - self.last_count_check > 1000 and self.limit - self.current_count < 10000):
            self.current_count = self.translator.get_usage().character.count
            self.last_count_check = self.current_count
            print(f"{'.'*40} (DeepL usage: {self.current_count}/{self.limit})")
        
        return self.current_count < self.limit
    
    def translate(self, batch:list) -> list:
        result = self.translator.translate_text(
            batch, 
            target_lang=self.conf["target_language"][0:2]
        )
        self.current_count += len("".join(batch))
        return [item.text for item in result]
    
    def get_usage(self):
        return self.limit - self.current_count, self.limit
        

class OpenAIUtil(TranslatorBase):
    def __init__(self, conf:dict):
        self.conf = conf
        openai.api_key = self.conf["openai_key"]
    
    def translate(self, batch:list) -> list:
        text = f"<p>{'</p><p>'.join(batch)}</p>"
        if f"openai_user_prompt_{self.conf['target_language']}" in self.conf:
            text = self.conf[f"openai_user_prompt_{self.conf['target_language']}"] + " " + text
        else:
            text = self.conf["openai_user_prompt_default"] + " " + text
        
        # try 5 times if openai is not available
        for i in range(5):
            try:
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": f"{self.conf['openai_system_prompt']} {self.conf['target_language']}."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0 
                )
                
                translated = chat_completion.choices[0].message.content.strip()
                #remove the first <p> and the last </p> and split to a list
                translated_list = translated[3:-4].split("</p><p>")
                
                # give or take items to make the length of the list equal to the length of the batch
                if len(translated_list) > len(batch):
                    translated_list = translated_list[:len(batch)]
                elif len(translated_list) < len(batch):
                    translated_list.extend(["*"]*(len(batch)-len(translated_list)))
                
                return translated_list
            except Exception as e:
                try_again_in = 2*(i+1)
                print(f"OpenAI is not available, try again in {try_again_in} seconds, {5-i} times left")
                time.sleep(2*(i+1))
                continue
        
        

class SRTTranslator:
    def __init__(self, srt_file:str, conf:dict):
        self.conf = conf
        if os.path.isfile(srt_file):
            # add language information to the target file name
            self.target_file = srt_file[:-4] + "_" + self.conf["target_language"] + ".srt"
            with open(srt_file, encoding="utf-8") as file:
                self.subtitles = self.__validate_subtitles(file.read())
        else:
            # raise file not found exception
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), srt_file)
    
    def translate(self, translator_list:list, buffer_size=20):
        translated_list = []
        batch = []
        total_indexes_count = len(self.subtitles)
        index_start = 0
        print("-"*10 + f" Total {total_indexes_count} indexes " + "-"*10)
        for subtitle in self.subtitles:
            content = self.__before_translate(subtitle.content)
            batch.append(content)
            if subtitle.index % buffer_size == 0:
                is_translated = False
                for translator in translator_list:
                    if translator.is_available():
                        print("-"*10 + f" Index: {index_start} ~ {subtitle.index} " + "-"*10)
                        try:
                            translated_list.extend(translator.translate(batch))
                            batch = []
                            index_start = subtitle.index+1
                            is_translated = True
                            break
                        except deepl.exceptions.QuotaExceededException as e:
                            continue
                if not is_translated:
                    raise Exception("All translators are not available")
        
        if len(batch) > 0:
            is_translated = False
            for translator in translator_list:
                if translator.is_available():
                    print("-"*10 + f" Last batch: {index_start} ~ {total_indexes_count} " + "-"*10)
                    translated_list.extend(translator.translate(batch))
                    is_translated = True
                    break
            if not is_translated:
                raise Exception("All translators are not available")
        
        # replace the content with the translated text in the subtitles
        if len(self.subtitles) == len(translated_list):
            for i in range(len(self.subtitles)):
                self.subtitles[i].content = self.__after_translate(translated_list[i])
        
        return self
    
    def __before_translate(self, text:str) -> str:
        result = text.strip()
        # replace the "{\an8}" in the text
        result = result.replace("{\\an8}", "")
        
        result = result.replace("\r\n","<br>")
        result = result.replace("\n","<br>")
        
        # if the nerber of [ is not equal to the number of ], delete [
        if result.count("[") > result.count("]"):
            result = result.replace("[","")
        
        return result
    
    def __after_translate(self, text:str) -> str:
        result = text.strip()
        
        result = result.replace("<br>","\n")
        return result
    
    def __validate_subtitles(self, content:str) -> list:
        subtitles = list(srt.parse(content))
        result_list = []
        deleted_index = -1
        # if the next subtitle's start time is not 200 milliseconds after the previous subtitle's end time, merge them
        for i in range(len(subtitles) - 1):
            if i > deleted_index:
                if subtitles[i+1].start - subtitles[i].end < datetime.timedelta(milliseconds=200):
                    subtitles[i].end = subtitles[i+1].end
                    subtitles[i].content += "\n" + subtitles[i+1].content
                    deleted_index = i+1
                result_list.append(subtitles[i])
        
        # add the last subtitle if the deleted_index is not the last one
        if deleted_index != len(subtitles) - 1:
            result_list.append(subtitles[-1])
        
        removed_indexes_count = len(subtitles) - len(result_list)
        if removed_indexes_count > 0:
            print(f"Remove {removed_indexes_count} subtitles, because they are too close to the previous one")
        
        return list(srt.sort_and_reindex(result_list))
        
    def save(self):
        # write to the target file
        with open(self.target_file, "w", encoding="utf-8") as target:
            target.write(srt.compose(self.subtitles))
        return self.target_file
