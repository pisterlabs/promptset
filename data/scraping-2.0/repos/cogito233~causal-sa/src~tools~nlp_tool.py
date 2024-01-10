from __future__ import unicode_literals, print_function
# from efficiency.log import show_var


class NLP:
    def __init__(self, disable=['ner', 'parser', 'tagger', "lemmatizer"]):
        import spacy

        self.nlp = spacy.load('en_core_web_sm', disable=disable)
        try:
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        except:
            self.nlp.add_pipe('sentencizer')

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)

    @staticmethod
    def sent_bleu(ref_list, hyp):
        from nltk.translate import bleu
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method4
        refs = [ref.split() for ref in ref_list]
        hyp = hyp.split()
        return bleu(refs, hyp, smoothing_function=smoothie)


class Translator:
    def __init__(self, cache_file='.cache_trans_lookup.csv', load_translator=True):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        if load_translator:
            from googletrans import Translator
            self.translator = Translator()

    def load_cache(self):
        from collections import defaultdict
        cache = defaultdict(dict)
        from efficiency.log import fread
        data = fread(self.cache_file, verbose=False)
        data = [i for i in data if (i['input'] != i['output']) and i['input'].strip() and i['output'].strip()]
        for i in data:
            cache[(i['src_lang'], i['tgt_lang'])][i['input']] = i['output']
            cache[(i['tgt_lang'], i['src_lang'])][i['output']] = i['input']

        return cache

    def save_cache(self, text, translated_text, src_lang, tgt_lang):
        if text not in self.cache[(src_lang, tgt_lang)]:
            datum = [{
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'input': text,
                'output': translated_text,
            }]
            self.cache[(src_lang, tgt_lang)][text] = translated_text
            self.cache[(tgt_lang, src_lang)][translated_text] = text

            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(datum, self.cache_file, mode='a')

    def raw_translate(self, text, src_lang, tgt_lang):
        def successful_pass():
            translated_text = self.translator.translate(text, src=src_lang, dest=tgt_lang).text

            if translated_text != text and translated_text.strip():
                return translated_text

        if_success = successful_pass()
        if if_success:
            return if_success
        else:
            import time
            time.sleep(10)
            if_success = successful_pass()
            if if_success:
                return if_success
            else:
                print(
                    "[Error] Translation failed. You have very likely reached the limit of the `googletrans' "
                    "library. Try to wait for a while or change your IP address. Alternatively, you can also edit the "
                    "source code here to change it to Google cloud API by setting up your credentials.")
                print(src_lang, tgt_lang, text)
                import pdb;
                pdb.set_trace()

    def translate(self, text, src_lang='en', tgt_lang='de', verbose=True, enable_cache=True):
        if src_lang == tgt_lang:
            return text

        this_cache = self.cache[(src_lang, tgt_lang)]
        if enable_cache and (text in this_cache):
            translated_text = this_cache[text]
        else:
            translated_text = self.raw_translate(text, src_lang, tgt_lang)
            self.save_cache(text, translated_text, src_lang, tgt_lang)

        #if verbose:
        #    from efficiency.log import show_var
        #    show_var(['text', 'translated_text', ])
        return translated_text

    @staticmethod
    def print_countries(langs):
        langs = sorted(langs)

        import pycountry
        rows = [
            {'lang': lang,
             'country': pycountry.languages.get(alpha_2=lang).name if pycountry.languages.get(alpha_2=lang) else None
             }
            for lang in langs
        ]
        import pandas as pd
        df = pd.DataFrame(rows, index=None)
        print(df)
        import pdb;
        pdb.set_trace()
        return df

    @staticmethod
    def get_language_list(list_from=['googletrans', 'pycountry', 'langcodes'][0]):
        langs = []
        if list_from == 'googletrans':
            from googletrans import LANGUAGES
            translateable_langs = []
            for language_code, language_name in LANGUAGES.items():
                translateable_langs.append(language_code)
            translateable_langs = sorted(translateable_langs)
            langs = translateable_langs
        elif list_from == 'pycountry':
            import pycountry
            code2family = [{"alpha_3": i.alpha_3, "name": i.name} for i in pycountry.language_families]

            pyc_langs = []
            for language in pycountry.languages:
                name = language.name.replace(' (macrolanguage)', '').strip()
                name = name.split('(', 1)[0].strip()
                item = {'name': name}
                try:
                    item['lang'] = language.alpha_2
                except:
                    try:
                        item['lang'] = language.alpha_3
                    except:
                        pass
                pyc_langs.append(item)
            additional_dict = [
                {'lang': 'zh-cn', 'name': 'Chinese (Simplified)'},
                {'lang': 'zh-tw', 'name': 'Chinese (Traditional)'},
                {'lang': 'iw', 'name': 'Modern Hebrew'},
                {'lang': 'jw', 'name': 'Javanese'},
                {'lang': 'me', 'name': 'Montenegrin'},
            ]
            pyc_langs.extend(additional_dict)
            from langcodes import Language
            for item in pyc_langs:
                language_code = item['lang']
                # lang = Language.get(language_code)
                # item['family'] = lang.family_name()
            import pandas as pd
            pyc_langs = pd.DataFrame(pyc_langs)
            langs = pyc_langs

        return langs


class Chatbot:
    model_version2engine = {
        'gpt4': "gpt-4",
        'gpt3.5': "gpt-3.5-turbo",
        'gpt3': "text-davinci-003",

        'gpt3.043': "text-davinci-003",
        'gpt3.042': "text-davinci-002",
        'gpt3.041': "text-davinci-001",
        'gpt3.04': "davinci",
        'gpt3.03': "curie",
        'gpt3.02': "babbage",
        'gpt3.01': "ada",
    }
    engine2pricing = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4-32k": 0.12,
        "gpt-4": 0.06,
        "text-davinci-003": 0.0200,
        "text-davinci-002": 0.0200,
        "text-davinci-001": 0.0200,
        "davinci": 0.0200,
        "curie": 0.0020,
        "babbage": 0.0005,
        "ada": 0.0004,
    }

    def __init__(self, model_version='gpt3.5', max_tokens=100, output_file=None, output_folder='./',
                 system_prompt="You are a helpful assistant.", cache_files=[],
                 openai_key_alias='OPENAI_API_KEY', openai_org_alias='OPENAI_ORG_ID', ):
        import os
        import openai
        api_key = os.environ[openai_key_alias]
        openai.api_key = api_key
        if openai_org_alias != 'OPENAI_ORG_ID':
            organization_id = os.environ[openai_org_alias]
            openai.organization = organization_id

        self.model_version = model_version
        self.engine = self.model_version2engine.get(model_version, model_version)
        self.max_tokens = max_tokens
        self.set_system_prompt(system_prompt)
        self.output_file = f'{output_folder}/.cache_{model_version}.csv' if output_file is None else output_file
        self.gpt_files = cache_files + [self.output_file]

        self.openai = openai
        self.num_tokens = []
        self.cache = self.load_cache()
        # self.list_all_models()
        self.clear_dialog_history()

    def set_system_prompt(self, system_prompt="You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.system_is_default = system_prompt == "You are a helpful assistant."

    def clear_dialog_history(self):
        self.dialog_history = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": "Who won the world series in 2020?"},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        ]

    def dialog_history_to_str(self, pure_completion_mode=False, ):
        dialog_history = [turn for turn in self.dialog_history
                          if not ((turn['role'] == 'system') and self.system_is_default)]

        if not self.if_newer_engine:
            # dialog_history = [turn for turn in dialog_history if not (turn['role'] == 'system')]
            for turn_i, turn in enumerate(dialog_history):
                if turn['role'] == 'system':
                    system_prompt = turn['content']
                    dialog_history.pop(turn_i)
                    next_turn = dialog_history[turn_i]
                    if next_turn['role'] == 'user':
                        dialog_history[turn_i]['content'] = '\n'.join([system_prompt, next_turn['content']])
                        break
            self.dialog_history = dialog_history
        if pure_completion_mode:
            dialog_history_text = turn['content']
        else:
            dialog_history_text = []
            for turn in dialog_history:
                if turn['role'] == 'system':
                    prefix = 'S'
                elif turn['role'] == 'user':
                    prefix = 'Q'
                elif turn['role'] == 'assistant':
                    prefix = 'A'
                else:
                    continue
                this_text = f"{prefix}: {turn['content']}"
                if prefix == 'A':
                    this_text += '\n'
                dialog_history_text.append(this_text)

            dialog_history_text = '\n'.join(dialog_history_text)

            if not self.if_newer_engine:
                if turn['role'] != 'assistant':
                    dialog_history_text += '\nA:'
        return dialog_history_text

    def list_all_models(self):
        model_list = self.openai.Model.list()['data']
        model_ids = [x['id'] for x in model_list]
        model_ids.sort()
        print(model_ids)
        import pdb;
        pdb.set_trace()

    @property
    def _total_cost(self):
        default_price = self.engine2pricing['davinci']
        return sum(self.num_tokens) // 1000 * self.engine2pricing.get(self.engine, default_price)

    def print_cost(self):
        print(f"[Info] Spent ${self._total_cost} for {sum(self.num_tokens)} tokens.")

    def save_cache(self, question, response_text):
        if (not (question in self.cache)) and response_text:
            self.cache[question] = response_text
            datum = [{
                'pred': response_text,
                'query': question,
            }]
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(datum, self.output_file, mode='a')

    def load_cache(self):
        cache = {}
        from efficiency.log import fread
        for file in self.gpt_files:
            data = fread(file, verbose=False)
            cache.update({i[f'query{q_i}']: i[f'pred{q_i}'] for i in data
                          for q_i in list(range(10)) + ['']
                          if f'query{q_i}' in i})
        cache = {k: v for k, v in cache.items() if v}  # there are cases where the response is empty
        return cache

    def ask(self, *args, **kwargs):
        def repeat():
            sec = 100
            print(f'[Info] openai.error.RateLimitError. Wait for {sec} seconds')
            self.print_cost()
            '''
            Default rate limits for gpt-4/gpt-4-0314 are 40k TPM and 200 RPM. Default rate limits for gpt-4-32k/gpt-4-32k-0314 are 80k PRM and 400 RPM. 
            https://platform.openai.com/docs/guides/rate-limits/overview
            '''

            import time
            time.sleep(sec)
            return self.ask(*args, **kwargs)

        import openai
        try:
            return self.raw_query(*args, **kwargs)
        except openai.error.InvalidRequestError:
            import pdb;
            pdb.set_trace()
            if len(self.dialog_history) > 10:
                import pdb;
                pdb.set_trace()
            for turn_i, turn in enumerate(self.dialog_history):
                if turn['role'] == 'assistant':
                    turn['content'] = turn['content'][:1000]

        except openai.error.RateLimitError:
            return repeat()
        except openai.error.APIError:
            return repeat()
        # except:
        #     repeat()

    def raw_query(self, question, system_prompt='You are a helpful assistant.',
                  turn_off_cache=False, valid_ways=['cache', 'api_call'],
                  continued_questions=False,
                  max_tokens=None, stop_sign="\nQ: ",
                  model_version=[None, 'gpt3', 'gpt3.5', 'gpt4'][0],
                  engine=[None, "text-davinci-003", "gpt-3.5-turbo", "gpt-4-32k-0314", "gpt-4-0314", "gpt-4"][0],
                  enable_pdb=False, verbose=True, only_response=True,
                  ):
        self.set_system_prompt(system_prompt)
        if model_version is not None:
            if model_version != self.model_version:
                turn_off_cache = True

        enable_api = 'api_call' in valid_ways
        if model_version is not None:
            engine = self.model_version2engine.get(model_version, model_version)
        elif engine is not None:
            engine = engine
        else:
            engine = self.engine
        self.engine = engine  # to be called in print_cost()

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        verbose = True if enable_pdb else verbose

        if_newer_engine = engine.startswith('gpt-3.5') or engine.startswith('gpt-4')
        self.if_newer_engine = if_newer_engine

        if not continued_questions:
            self.clear_dialog_history()

        self.dialog_history.append({"role": "user", "content": question}, )

        prompt = self.dialog_history_to_str()
        cache_input = prompt

        if enable_pdb:
            print(cache_input)
            import pdb;
            pdb.set_trace()
        if (cache_input in self.cache) & (not turn_off_cache):
            response_text = self.cache[cache_input]
            if not if_newer_engine:
                response_text = str(response_text).split(stop_sign, 1)[0]
            response_text = response_text.strip()
            if verbose: print(f'[Info] Using cache for "{cache_input[:10]}..."')
        elif enable_api:
            openai = self.openai
            if if_newer_engine:
                response = openai.ChatCompletion.create(
                    model=engine,
                    temperature=0,
                    max_tokens=max_tokens,
                    messages=self.dialog_history,
                )
                response_text = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=engine,
                    # prompt=[question],
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    stop=stop_sign,
                )
                response_text = response['choices'][0]['text']
            self.num_tokens.append(response['usage']["total_tokens"])
            response_text = response_text.strip()
            if verbose: self.print_cost()
        else:
            response_text = ''

        self.dialog_history.append({"role": "assistant", "content": response_text}, )

        if verbose:
            print()
            print(self.dialog_history_to_str())

        if enable_pdb:
            import pdb;
            pdb.set_trace()

        if enable_api:
            if not turn_off_cache:
                self.save_cache(cache_input, response_text)

        if only_response:
            return response_text
        return response_text, output


def main():
    raw_text = 'Hello, world. Here are two people with M.A. degrees from UT Austin. This is Mr. Mike.'
    nlp = NLP()
    sentences = nlp.sent_tokenize(raw_text)
    words = nlp.word_tokenize(sentences[0], lower=True)
    #show_var(['sentences', 'words'])

    chat = Chatbot()
    query = 'What is the best way to learn Machine Learning?'
    response = chat.ask(query)


if __name__ == '__main__':
    main()