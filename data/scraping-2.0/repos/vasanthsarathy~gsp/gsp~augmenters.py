from abc import ABC, abstractmethod
import ast
from googletrans import Translator
import random
import os
import click

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

from gtts import gTTS 
import speech_recognition as sr
from pydub import AudioSegment
import librosa
import librosa.display as librosa_display
import matplotlib.pyplot as plt
import nlpaug.augmenter.audio as naa
from nlpaug.util.audio.visualizer import AudioVisualizer
import soundfile as sf

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI,ChatAnthropic
from langchain.chains import LLMChain

import translators as ts 

class Augmenter(ABC):
    """
    Abstract class for all augmenter chains
    """
    def __init__(self,llm=None):
        self.llm = llm
        self.current_utterance = ""

    def post_processor(self, output_list_str):
        # converts a string list of outputs to a list of strings
        if isinstance(output_list_str, str):
            cleaned = output_list_str.replace("\n","")
            try:
                output_list = ast.literal_eval(cleaned)
                return output_list
            except:
                print("Exception!")
                print(cleaned)
        return output_list_str

    @abstractmethod
    def run(self):
        """
        To be implemented 
        """


class SpeechAugmenter(Augmenter,ABC):
    def __init__(self):
        super().__init__()

    def tts(self, utterance, mp3_filepath, accent="us"):
        """
        Converts to wave file 
        """
        var = gTTS(text = utterance,lang = 'en',tld=accent) 
        var.save(mp3_filepath) 
        return True

    def conv_mp3_wav(self, mp3_filepath, wav_filepath):
        """
        converts mp3 file to wave
        """
        # convert mp3 file to wav  
        src=(mp3_filepath)
        sound = AudioSegment.from_mp3(src)
        sound.export(wav_filepath, format="wav")
        return wav_filepath
    

    def stt(self, wav_filepath):
        # use the audio file as the audio source                                        
        r = sr.Recognizer()
        file_audio = sr.AudioFile(wav_filepath)
        with file_audio as source:
            audio_text = r.record(source)
        try:
            output = r.recognize_google(audio_text)
        except:
            output = r.recognize_sphinx(audio_text)
        return output


class AccentSpeechAugmenter(SpeechAugmenter):
    def __init__(self):
        super().__init__()
        self.accents = {"australian": "com.au",
                   "british": "co.uk",
                   "american": "us",
                   "canadian": "ca",
                   "indian": "co.in",
                   "irish": "ie",
                   "south_african": "co.za"}
        self.type = "speech:accent"

    def run(self, utterance, accent):
        if utterance:
            self.tts(utterance, "out.mp3", accent=self.accents[accent])
            filepath = self.conv_mp3_wav("out.mp3", "out.wav")
            result = self.stt("out.wav")
            meta = {'augmenter': f"{self.type}:{accent}"}
            variations = [result]
        else:
            variations = []
        return {'utterance': utterance, 
                'variations': variations, 
                'metadata': meta}

                   
class AudioSpeechAugmenter(SpeechAugmenter):
    def __init__(self):
        super().__init__()
        self.type = "speech:audio"

    def run(self, utterance, augmenter="crop"):
        self.tts(utterance, "out.mp3")
        filepath = self.conv_mp3_wav("out.mp3", "out.wav")
        augm = self._augment_audio("out.wav","out.wav", augmenter=augmenter)
        result = self.stt("out.wav")
        meta = {'augmenter': f"{self.type}:{augmenter}"}
        return {'utterance': utterance, 
                'variations': [result], 
                'metadata': meta}

    def _augment_audio(self, input_wav_file, output_wav_file, augmenter):
        data, sr = librosa.load(input_wav_file)

        augmenters = {"crop": naa.CropAug(sampling_rate=sr),
                      "loudness": naa.LoudnessAug(),
                      "mask": naa.MaskAug(sampling_rate=sr, mask_with_noise=False),
                      "noise": naa.NoiseAug(),
                      "pitch": naa.PitchAug(sampling_rate=sr, factor=(2,3)),
                      "shift": naa.ShiftAug(sampling_rate=sr),
                      "speed": naa.SpeedAug(),
                      "VTLP": naa.VtlpAug(sampling_rate=sr),
                      "normalize": naa.NormalizeAug(method='minmax'),
                      "polarity_inversion": naa.PolarityInverseAug()
                     }

        if not augmenter == "none":
            aug = augmenters[augmenter]
            augmented_data = aug.augment(data)
            sf.write(output_wav_file, augmented_data[0], sr)
            return True

        sf.write(output_wav_file, data, sr)
        return True

### Style Augmenters
class NoStyleAugmenter(Augmenter):
    def __init__(self,llm):
        super().__init__(llm)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = [utterance]
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out

class DirectnessStyleAugmenter(Augmenter):

    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance in below in {n} different ways to make it more indirect and polite. Feel free to make it more verbose if necessary. Return as a python list.
        Even if only one item, return as a list.
        
        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out


class CorrectionStyleAugmenter(Augmenter):

    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance in below in {n} different ways with mid-sentence corrections. The utterance is correct, introduce some inaccuracies and correct to what is in the utterance. Feel free to make it more verbose if necessary. Return as a python list.
        Even if only one item, return as a list.

        Example:
        utterance: put the donut left of the mug
        rewritten: put the bag, actually i mean the donut, to the left of the mug

        Example:
        utterance: put the donut left of the mug
        rewritten: put the donut to the left of the fridge, wait.. no., I mean left of the mug.

        Example:
        utterance: put the donut left of the mug
        rewritten: put the donut to the right of the mug, actually scratch that no, put it to the left of it. 

        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out


class ASRStyleAugmenter(Augmenter):

    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance below in {n} different ways with errors that typically arise from automated speech recognition systems. This can include audio issues and limitations of speech-to-text transcription systems. 

        There are typically three types of errors that occur in speech recognition. First, Substitution; where a word in the reference word sequence is transcribed as a different word. Second, Deletion; where a word in the reference is completely
        missed in the automatic transcription. And finally, Insertion; where a word appears in the automatic transcription that has no correspondent in the reference word sequence. Ensure that the rewrites capture these types of errors.

        Feel free to make it more verbose if necessary. Return as a python list. Even if only one item, return as a list.

        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out


class FamiliarityStyleAugmenter(Augmenter):
    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance below in {n} different ways to make it sound like it is said by someone who speaks English as a second or third language, lacking language familiarity, fluency, having alternative word choices, having different degrees of politeness, and deference. Return as a python list.
        Even if only one item, return as a list.
        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out


class FormalityStyleAugmenter(Augmenter):
    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance below in {n} different ways to make it sound less formal and much more casual, almost like something someone might say on twitter or reddit. Return as a python list.
        Even if only one item, return as a list.
        
        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out
        
class DisfluencyStyleAugmenter(Augmenter):
    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance below in {n} different ways to introduce speech disfluencies. A speech disfluency is any of various breaks, irregularities, or non-lexical vocables which occur within the flow of otherwise fluent speech. These include "false starts", i.e. words and sentences that are cut off mid-utterance; phrases that are restarted or repeated, and repeated syllables; "fillers", i.e. grunts, and non-lexical or semiarticulate utterances such as huh, uh, erm, um, and hmm, and, in English, well, so, I mean, and like; and "repaired" utterances, i.e. instances of speakers correcting their own slips of the tongue or mispronunciations (before anyone else gets a chance to). Huh is claimed to be a universal syllable. Return as a python list.
        Even if only one item, return as a list.
        
        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out

class WordChoiceStyleAugmenter(Augmenter):
    def __init__(self,llm):
        super().__init__(llm)
        self.template = """
        Rewrite the utterance in {n} different ways to change the verbs and nouns with synonyms or other ways of describing the same action or object. Return as a python list.
        Even if only one item, return as a list.
        
        utterance: \n{utterance}\n
        rewritten:
        """
        self.prompt = PromptTemplate(input_variables=["utterance","n"],template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.type = "style"

    def run(self, utterance, n):
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        results = self.post_processor(self.chain.run(utterance=utterance, n=n).lower())
        out = {'utterance': utterance,
               'variations': results,
               'metadata': meta}
        return out

### Text augmenters 

class TranslationAugmenter(Augmenter):
    def __init__(self, ctx):
        super().__init__()
        #self.translator = Translator()
        if ctx['preaccelerate']:
            click.secho(">> Preaccelerating translation engine.")
            _ = ts.preaccelerate_and_speedtest() 
            click.secho("Preacceleration completed.")
        self.type = "translate"

    def run(self, utterance, language):
        self.current_utterance = utterance
        # result = self.translator.translate(utterance, src='en', dest=language)
        try:
            result = ts.translate_text(utterance, translator="alibaba", from_language="en", to_language=language)
        except:
            print("Utterance: ", utterance)
            print("Language: ", language)
            result = ""
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}", "language": language}
        
        out = {'utterance': utterance,
               'variations': [result],
               'metadata': meta}
        return out



class BackTranslationTextAugmenter(Augmenter):
    def __init__(self, ctx):
        super().__init__()
        if ctx['preaccelerate']:
            click.secho(">> Preaccelerating translation engine.")
            _ = ts.preaccelerate_and_speedtest() 
            click.secho("Preacceleration completed.")
        self.type = "text"

    def run(self, utterance, n):
        self.current_utterance = utterance
        outs = []
        langs = []

        counter = 0
        from gsp.languages import languages
        random.shuffle(languages)
        for lang in languages:
            #try:
            # connect
            #result = self.translator.translate(utterance, src='en', dest=lang[0])
            #back = self.translator.translate(result.text, src=lang[0], dest='en')
            result = ts.translate_text(utterance, from_language="en", to_language=lang[0])
            back = ts.translate_text(utterance, from_language=lang[0], to_language="en")
            counter += 1
            langs.append(lang)
            outs.append(back)
            #except:
             #   pass

            if counter == n:
                break
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}", "languages": langs}
        
        out = {'utterance': utterance,
               'variations': outs,
               'metadata': meta}
        return out
        
class SynonymTextAugmenter(Augmenter):
    def __init__(self, ctx):
        super().__init__()
        os.environ["MODEL_DIR"] = "models/"
        model_dir ="models/"
        self.type = "text"

    def run(self, utterance, n):
        aug = naw.SynonymAug(aug_src='wordnet')
        variations = aug.augment(utterance)
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        out = {'utterance': utterance,
               'variations':variations,
               'metadata': meta}
        return out


class SpanCropTextAugmenter(Augmenter):
    def __init__(self, ctx):
        super().__init__()
        os.environ["MODEL_DIR"] = "models/"
        model_dir ="models/"
        self.type = "text"

    def run(self, utterance, n):
        aug = naw.RandomWordAug(action='crop')
        variations = aug.augment(utterance)
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        out = {'utterance': utterance,
               'variations':variations,
               'metadata': meta}
        return out

class ContextualWordEmbeddingTextAugmenter(Augmenter):
    def __init__(self, ctx):
        super().__init__()
        os.environ["MODEL_DIR"] = "models/"
        model_dir ="models/"
        self.type = "text"

    def run(self, utterance, n):
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        variations = []
        variations = aug.augment(utterance)
        meta = {'augmenter': f"{self.type}:{self.__class__.__name__}"}
        out = {'utterance': utterance,
               'variations':variations,
               'metadata': meta}
        return out

