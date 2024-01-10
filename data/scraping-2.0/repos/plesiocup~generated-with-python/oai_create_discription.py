#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 0.28.1 or lower.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://oai-generated.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

class Oai_create_discription:

    def get_discription_response(self, element):
        # message_content = 'タイトルが' + element + 'の映画が魅力的に思えるような説明文を想像で200字程度で出力して' # 聞き方は要検討
        # message_text = [{"role":"system","content": message_content}]

        message_text = [{"role":"system","content": "映画のタイトルからあらすじを想像して出力してほしい。文字数は200字程度。です、ますを使わず断定的な文で"},
                        {"role": "user", "content": "タイトル： 機動戦士ガンダム　逆襲のシャア"},
                        {"role": "assistant", "content": "アムロとシャアの13年に及ぶ宿命の対決に決着がつく、シリーズ最終章。宇宙世紀0093、シャアがネオ・ジオン総帥に就任、地球に宣戦を布告した。ブライト率いる連邦軍の独立部隊、ロンド・ベルに所属していたアムロは、最新型のνガンダムに搭乗して出撃する。アムロはシャアの野望を阻止することができるのか？"},
                        {"role": "user", "content": "タイトル： ハリー・ポッターとアズカバンの囚人 (吹替版)"},
                        {"role": "assistant", "content": "13歳になったハリーを待ち受けるのは、不吉な死の予言さえ告げられる中、ハリーが直面する両親の死の真相。今まで見えなかったものが見え始め、わからなかったことがわかり始める第3章。"},
                        {"role": "user", "content": "タイトル： シン・ゴジラ"},
                        {"role": "assistant", "content": "突如現れた巨大不明生物「ゴジラ」が東京湾を襲い、政府は自衛隊を川崎に派遣して対峙。緊急事態に直面し、人智を超えるゴジラに対抗する策を模索する中、巨大生物との壮絶な戦いが始まる。"},
                        {"role": "user", "content": "タイトル： " + element}]

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k",
                messages = message_text,
                temperature = 0.6,
                max_tokens = 500,
                top_p = 0.95,
                frequency_penalty = 0,
                presence_penalty = 0,
                stop = None
            )
            
            response_len = len(response['choices'][0]['message']['content'])
            # print(response_len)

            if response_len > 250:
                    message_content = response['choices'][0]['message']['content'] + '\n上の文章を250文字以内に要約して' # 聞き方は要検討
                    message_text = [{"role":"system","content": message_content}]
                    # print("要約しなくちゃ！！！！")

                    try:
                        response = openai.ChatCompletion.create(
                            engine="gpt-35-turbo-16k",
                            messages = message_text,
                            temperature = 0.6,
                            max_tokens = 500,
                            top_p = 0.95,
                            frequency_penalty = 0,
                            presence_penalty = 0,
                            stop = None
                        )

                        return response['choices'][0]['message']['content']
                    
                    except Exception as e:
                        print(f"Error: {e}")
                        return "error!"
                    
            else:
                return response['choices'][0]['message']['content']
        
        except Exception as e:
            print(f"Error: {e}")
            return "error!"
        
    def get_prompt_response(self, title, category):
        message_text = [{"role":"system","content": f"Imagine the story of a movie with this title {title} and this category {category}. Write a prompt to output the most moving scene in that story in dalle"}]

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k",
                messages = message_text,
                temperature = 0.6,
                max_tokens = 500,
                top_p = 0.95,
                frequency_penalty = 0,
                presence_penalty = 0,
                stop = None
            )
            
            response_len = len(response['choices'][0]['message']['content'])
            # print(response_len)

            if response_len > 250:
                    message_content = response['choices'][0]['message']['content'] + '\n上の文章を250文字以内に要約して' # 聞き方は要検討
                    message_text = [{"role":"system","content": message_content}]
                    # print("要約しなくちゃ！！！！")

                    try:
                        response = openai.ChatCompletion.create(
                            engine="gpt-35-turbo-16k",
                            messages = message_text,
                            temperature = 0.9,
                            max_tokens = 500,
                            top_p = 0.95,
                            frequency_penalty = 0,
                            presence_penalty = 0,
                            stop = None
                        )

                        return response['choices'][0]['message']['content']
                    
                    except Exception as e:
                        print(f"Error: {e}")
                        return "error!"
                    
            else:
                return response['choices'][0]['message']['content']
        
        except Exception as e:
            print(f"Error: {e}")
            return "error!"
        