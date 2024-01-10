'''
Schema and implementation of data loaders, wrapping the xml loading process and various cleaning procedures.
'''

from abc import ABC, abstractmethod
from type_utils import ProcessedData, UnprocessedData, MatchedData, Match, CleanedAndLabeledData, Label
import os
import re
from lxml import etree
from nltk.corpus import stopwords
from typing import List, Callable
import hashlib
from nltk.stem.snowball import SnowballStemmer
import pickle
import json
from tqdm import tqdm
import dotenv
import openai

class AbstractDataLoader(ABC):
    '''
    Defines XML search process to be called by implementing classes
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.name = data_path

    @abstractmethod
    def load_and_preprocess_data(self) -> ProcessedData | UnprocessedData: # TODO: handle this better, maybe with a different class
        """Load and preprocess text data"""
        pass

    def _process_files_to_data(self, clean_files: List[str], processing_function: Callable[[str], List[str]]) -> ProcessedData:
        data: ProcessedData = { 'good': [], 'bad': [] }
        
        for a in clean_files:
            
            print(a)

            a = os.path.join(self.data_path, a)
            
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(a, parser)
            
            for node in tree.xpath('//snippet'):
                if node.get('quality') != None and node.get('quality') == 'good':
                    data['good'].append(processing_function(node.text))
                elif node.get('confirmed') != None and node.get('confirmed') == 'yes':
                    data['good'].append(processing_function(node.text))
                elif node.get('classifier_result') != None and node.get('classifier_result') == 'good':
                    pass
                else:
                    data['bad'].append(processing_function(node.text))

        return data
    
    def _is_number(self, w) -> bool:

        result = False
        
        try:
            n = int(w)
            result = True
        except ValueError:
            pass
        
        return result

    def _is_t_valid(self, t: str) -> bool:
        
        terms_to_find= [r'Notre-\s*Dame', 'Cité', r'Saint-\s*Louis', 'Arènes', 
            r'Palais\s*de\s*Justice|Palais-\s*de-\s*Justice',
            'Morgue', r'Sainte-\s*Chapelle', 'Conciergerie', r'[Qq]uai\s*de\s*l\'Horloge', r'Pont-\s*Neuf', r'Cluny|Thermes',
            r'Saint-\s*Germain-\s*des-\s*Prés', 'Nesle', r'[Ss]aint-\s*Sulpice', r'[Pp]alais\s*du\s* Luxembourg', 
            r'[Jj]ardin\s*du\s*Luxembourg', 'Observatoire', r'Panthéon|Sainte-\s*Geneviève', r'[Eéeé]glise\s*Saint-\s*Étienne',
            'Odéon', r'[Jj]ardin\s*des\s*Plantes', 'Gobelins', 'Auxerrois', 'Louvre', r'Carrousel|Doyenné', 'Tuileries', 
            r'Palais-\s*Royal', r'Comédie-\s*Française', 'Bourse', 'Innocents', 'Halles', r'Saint-\s*Eustache', 'Temple',
            r'[Tt]our\s*Saint-\s*Jacques', r'H[oôóòö]tel\s*de\s*Ville|Gr[eêéèë]ve', 'Rivoli', 
            r'Bastille|[Cc]olonne\s*de\s*Juillet', 'Tournelles', r'[Bb]oulevar[dt]\s*de\s*la\s*Madeleine', 'Capucines',
            r'[Bb]oulevard\s*des\s*Italiens', r'[Bb]oulevar[dt]\s*Montmartre', r'[Bb]oulevar[dt]\s*Poissonnière',
            r'[Bb]oulevar[dt]\s*Bonne-\s*Nouvelle', r'[Bb]oulevar[dt]\s*Saint-\s*Denis', r'[Bb]oulevar[dt]\s*Saint-\s*Martin',
            r'[Bb]oulevar[dt]\s*du\s*Temple|[Bb]oulevard\s*du\s*crime', r'[Bb]oulevar[dt]\s*des\s*Filles', 'Beaumarchais',
            r'[Pp]orte\s*Saint-Denis', r'[Cc]afé\s*Tortoni', r'[Cc]afé\s*Anglais', r'Maison-\s*Dorée', 
            r'Notre-\s*Dame-\s*de-\s*Lorette', r'Opéra-\s*[Cc]omique', 'Panorama', 'Opéra', r'[Aa]venue\s*\s*de\s*l’Op[eé]ra',
            r'[Rr]ue\s*de\s*la\s*Paix|[Rr]ue\s*\de\s* Napoléon', 'Vivienne', r'[Rr]ue\s*Saint-\s*Jacques', 
            r'[Rr]ue\s*Saint-\s*Denis', r'[Ff]aubourg\s*Saint-\s*Honoré', r'[Rr]ue\s*du\s*[Ff]aubourg\s*Saint-\s*Antoine',
            r'[Ff]aubourg\s*Saint-\s*Antoine', r'[Pp]lace\s*des\s*Vosges|Place\s*Royale', r'Champs-\s*Elysées', 
            r'Concorde|[Pp]lace\s*Louis[.\s*]XV|obélisque', r'[EÉ]toile|Triomphe', 'Vend[oôóòö]me', r'[Ll]a\s*Madeleine', 
            'Caire', r'des\s*Miracles', r'Quinze-\s*Vingts', r'cimeti[eêéèë]re\s*du\s*P[eêéèë]re-\s*Lachaise,' 
            r'[Bb]utte\s*Montmartre', r'Montfaucon|[Vv]oierie', 'Chaumont', r'[Cc]h[aâáàä]teau\s*de\s*Vincennes', 'Invalides',
            r'[Eéeé]cole\s*Militaire|Champ-\s*de-\s*Mars', 'Grenelle']   
        
        result = False
        
        if t > '':
            
            was_matched = False
            for term in terms_to_find:
            
                for match in re.finditer(term, t, flags=re.IGNORECASE):
                    was_matched = True
                    
                if was_matched == True:
                    break
            
            if was_matched == False and self._is_number(t) == False:
                result = True
            
        return result
    
class NoSWLoader(AbstractDataLoader):
    '''
    Loads data without stopwords removed
    '''
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.name = data_path + 'NoSW'

    def load_and_preprocess_data(self) -> ProcessedData:
        # hash self.data_path
        hash_name = hashlib.md5(self.name.encode('utf-8')).hexdigest()
        
        if os.path.exists(f'{hash_name}.pkl'):
            print('loading data...')
            return pickle.load(open(f'{hash_name}.pkl', 'rb'))
        
        else:
            clean_files = sorted(os.listdir(self.data_path))
            data = self._process_files_to_data(clean_files, self._tokenize_text)
            pickle.dump(data, open(f'{hash_name}.pkl', 'wb'))
            print('saved data to', f'{hash_name}.pkl')
            return data
    
    def _tokenize_text(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^\s0123456789abcdefghijklmnopqrstuvwxyzàâäæçèéêëîïñôùûüÿœ̀œ]',
                        ' ',
                        text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return [t for t in clean_text.lower().split(' ') if self._is_t_valid(t)]

class MatchLoader(AbstractDataLoader):
    '''
    Loads data in the Match format (see data_types.py), which is used to track the monument itself through the process.
    '''
    def load_and_preprocess_data(self) -> MatchedData:
        clean_files = sorted(os.listdir(self.data_path))
        data: MatchedData = {'good': [], 'bad': []}

        for a in clean_files:
            
            # print(a)

            a = os.path.join(self.data_path, a)
            
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(a, parser)
            root = tree.getroot()
            for child in root:
                snippet = child.find('snippet')
                monument = child.find('matching_term')
                match: Match = {'monument': monument.text, 'snippet': snippet.text}

                if snippet.get('quality') != None and snippet.get('quality') == 'good':
                    data['good'].append(match)
                elif snippet.get('confirmed') != None and snippet.get('confirmed') == 'yes':
                    data['good'].append(match)
                elif snippet.get('classifier_result') != None and snippet.get('classifier_result') == 'good':
                    pass
                else:
                    data['bad'].append(match)

                    
        print()
        print('# good', len(data['good']))
        print('# bad', len(data['bad']))

        return data

class OriginalDataLoader(AbstractDataLoader):
    '''
    Implementation of the original strategy used in the notebooks
    '''
    def load_and_preprocess_data(self) -> ProcessedData:
        """Load and preprocess text data"""

        # hash self.data_path
        hash_name = hashlib.md5(self.name.encode('utf-8')).hexdigest()
        
        if os.path.exists(f'{hash_name}.pkl'):
            return pickle.load(open(f'{hash_name}.pkl', 'rb'))
        
        else:

            clean_files = sorted(os.listdir(self.data_path))


            sw = list(set(stopwords.words('french') + ['ici', 'là', 'elles', 'trop', 'tous', 'selon', 'presque', 'tant', 
                                                'fois', 'quant', 'ainsi', 'cette', 'doit', 'tout', 'bien', 'toute', 
                                                'si', 'autre', 'sans', 'comment', 'rien', 'là', 'peu', 'mêmes', 'si', 
                                                'plutôt', 'ceux', 'faire', 'moins', 'être', 'faudra', 
                                                'deux', 'a', 'paris', 'plus', 'où', 'saint', 'cette']))
            
            data: ProcessedData = { 'good': [], 'bad': [] }
            
            for a in clean_files:
                
                print(a)

                a = os.path.join(self.data_path, a)
                
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(a, parser)
                
                for node in tree.xpath('//snippet'):
                    if node.get('quality') != None and node.get('quality') == 'good':
                        data['good'].append(self._tokenize_text(node.text, sw))
                    elif node.get('confirmed') != None and node.get('confirmed') == 'yes':
                        data['good'].append(self._tokenize_text(node.text, sw))
                    elif node.get('classifier_result') != None and node.get('classifier_result') == 'good':
                        pass
                    else:
                        data['bad'].append(self._tokenize_text(node.text, sw))
                
            print()
            print('# good', len(data['good']))
            print('# bad', len(data['bad']))

            pickle.dump(data, open(f'{hash_name}.pkl', 'wb'))
            print('saved data to', f'{hash_name}.pkl')

            return data

    

    def _tokenize_text(self, text: str, sw: List[str]) -> List[str]:
        
        clean_text = re.sub(r'[^\s0123456789abcdefghijklmnopqrstuvwxyzàâäæçèéêëîïñôùûüÿœ̀]',
                        ' ',
                        text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return [t for t in clean_text.lower().split(' ') if self._is_t_valid(t) and t not in sw]

class StemDataLoader(OriginalDataLoader):
    '''
    Loads data with an additional stemming process. In short, attemps to reduce words to their root.
    '''
    def __init__(self, data_path):
        super().__init__(data_path=data_path)
        self.name = data_path + '_stemlemma'

    def _tokenize_text(self, text: str, sw: list) -> List[str]:
        """Tokenize text, remove stopwords, and stem the words"""
        stemmer = SnowballStemmer(language='french')
        return [stemmer.stem(t) for t in super()._tokenize_text(text, sw)]
    
class GPTCleanedLoader(OriginalDataLoader):
    '''
    Uses GPT 3.5-turbo to clean the data and remove OCR errors.
    '''
    def __init__(self, data_path, cleaned_path, verbose=False, inference=False):
        super().__init__(data_path=data_path)
        self.cleaned_path = cleaned_path
        self.name = data_path + '_gptcleaned'
        self.verbose = verbose # turns on print messages, and the original / cleaned messages are printed
        self.inference = inference

        self.prompt = "Votre travail consiste à transformer cette numérisation OCR défectueuse en une numérisation rectifiée, en corrigeant l'espacement, le formatage et la grammaire et la syntaxe appropriées en français si nécessaire. Renvoie uniquement le texte corrigé, sans nouvelle ligne supplémentaire ni espace blanc divers. N'ajoutez PAS de contenu supplémentaire à votre réponse, sinon vous serez pénalisé." 
        #"Your job is to turn this faulty OCR scan into a rectified one, correcting spacing, formatting, and French-language proper grammar and syntax as necessary. Return only the corrected text, without additional new lines or misc whitespace"

        api_key = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

        if api_key is None:
            raise Exception("OPENAI_API_KEY not found in .env. This may be because you have not created a .env file.")
        
        openai.api_key = api_key
        
    def load_and_preprocess_data(self) -> UnprocessedData:

        # attempt to load from local
        hash_name = hashlib.md5(self.name.encode('utf-8')).hexdigest()
        print(hash_name)
        
        if os.path.exists(f'{hash_name}.pkl'):
            if self.verbose: 
                print('loading full data from store...')
            return pickle.load(open(f'{hash_name}.pkl', 'rb'))
        
        # else, load manually
        clean_files = sorted(os.listdir(self.data_path))
        data: UnprocessedData = { 'good': [], 'bad': [] }
       

        for file_name in tqdm(clean_files):
            full_path = os.path.join(self.data_path, file_name)
            filedata: List[CleanedAndLabeledData] = []



            # get cleaned file, which is the file name with a json extension instead of xml
            cleaned_file = os.path.join(self.cleaned_path, file_name).replace('.xml', '.json')


            fully_cleaned = True
            if os.path.exists(cleaned_file):
                if self.verbose:
                    print('loading from json')
                with open(cleaned_file, 'r') as f:
                    filedata = json.load(f)

                
                for d in filedata:
                    if d['cleaned'] == -1:
                        fully_cleaned = False
                        break
            else:
                fully_cleaned = False

            if not fully_cleaned:
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(full_path, parser)
                
                idx = 0
                for node in tqdm(tree.xpath('//snippet')):
                    label = None
                    text = node.text
                    if node.get('quality') != None and node.get('quality') == 'good':
                        label = Label.good
                    elif node.get('confirmed') != None and node.get('confirmed') == 'yes':
                        label = Label.good
                    elif node.get('classifier_result') != None and node.get('classifier_result') == 'good':
                        if self.inference:
                            label = Label.good
                        else:
                            pass
                    else:
                        label = Label.bad

                    if label != None:
                        # two cases: either we have attempted a clean at every snippet, or we have none
                        if len(filedata) > idx: # all attempted
                            if filedata[idx]['cleaned'] == -1: # but current failed
                                cleaned = self._call_gpt(text, self.verbose)
                            
                                if self.verbose:
                                    print('cleaned res', cleaned)

                                filedata[idx] = { # type: ignore
                                    'label': label, # type: ignore
                                    'text': text,
                                    'cleaned': cleaned
                                }

                                if idx % 10 == 0: # save every 10
                                    # write filedata back to json
                                    with open(cleaned_file, 'w') as f:
                                        json.dump(filedata, f)
                            else:
                                if self.verbose: 
                                    print('skipping...')
                        else: # not previously attempted, standard procedure
                            cleaned = self._call_gpt(text, self.verbose)
                        
                            if self.verbose:
                                print('cleaned res', cleaned)
                            # if cleaned == -1:
                            #     raise Exception('GPT call failed')
                            filedata.append({
                                'label': label, # type: ignore
                                'text': text,
                                'cleaned': cleaned
                            })

                        idx += 1

                if self.verbose:
                    print('saving to json...')
                with open(cleaned_file, 'w') as f:
                    json.dump(filedata, f)


            for snippet in filedata:
                if snippet['cleaned'] == -1:
                    snippet['cleaned'] = snippet['text']
                if snippet['label'] == Label.good:
                    data['good'].append(snippet['cleaned'])
                else:
                    data['bad'].append(snippet['cleaned'])
            
            # save data
        print()
        print('# good', len(data['good']))
        print('# bad', len(data['bad']))

        pickle.dump(data, open(f'{hash_name}.pkl', 'wb'))
        print('saved data to', f'{hash_name}.pkl')

        return data
        
    def _tokenize_text(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^\s0123456789abcdefghijklmnopqrstuvwxyzàâäæçèéêëîïñôùûüÿœ̀]',
                        ' ',
                        text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return [t for t in clean_text.lower().split(' ') if self._is_t_valid(t)]
    
    def _call_gpt(self, text: str, verbose=False) -> str | int:
        """Call GPT to clean the text"""

        model = "gpt-3.5-turbo"
        msgs = [
            {
                "role": "system",
                "content": self.prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]

        if verbose:
            print('incoming msg', msgs)

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=msgs,
                temperature=0.2,
            )

        except Exception as e:
            print('error', e)
            return -1


        return response.choices[0]['message']['content'] # type: ignore