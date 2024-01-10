import json
import click
import gsp.io as io
from gsp.augmenters import *
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI,ChatAnthropic
from langchain.chains import LLMChain
import itertools
import pandas as pd
from tqdm import tqdm

def prepare(ctx):
    """
    This pipeline converts a json dataset to a format suitable for finetuning an LLM 
    """
    data = io.load_jsonl(ctx['input'])
    data_for_finetuning = []
    for obj in data:
        if "augmented_utterance" in obj:
            # use augmented utterance if available
            if not obj["augmented_utterance"] == "":
                item = {'input': obj['augmented_utterance'],'output': obj['semantics']}
        elif "styled_utterance" in obj:
            # see the stylize() function for the key-values in the stylized jsons 
            item = {'input': obj['styled_utterance'],'output': obj['semantics']}
        else:
            # if the data has not been stylized then use old keyvalues
            item = {'input': obj['utteranceText'], 'output': obj['desiredSemantics']}

        item['text'] = f"### Input: {item['input']}\n\n ### Output: {item['output']}"
        data_for_finetuning.append(item)

    # removing duplicates
    memo = set()
    pruned_data = []
    K = "input"
    for sub in data_for_finetuning:
        
        # testing for already present value
        if sub[K] not in memo:
            pruned_data.append(sub)
            
            # adding in memo if new value
            memo.add(sub[K])

    #df = pd.DataFrame(pruned_data)
    #df.to_csv(f"data/output/data_{ctx['type']}.csv", index=False)
            
    stream = io.record_objs(pruned_data, f"finetuning_{ctx['type']}", ctx)
    for i in stream:
        pass
    print("Ready for finetuning")
    return True


def stylize(ctx):
    if ctx['verbose']:
        print("Config: ",json.dumps(ctx, indent=2))

    # Load up the data 
    stream = io.load_original_data(ctx)
    stream = io.record_objs(stream, "original", ctx)

    # Stylize the data and record in a new file 
    stylized_stream = stylize_utterance(stream, ctx)
    stylized_stream = io.record_objs(stylized_stream, "stylized", ctx)

    i = 0;
    for f in stylized_stream:
        i += 1

    click.secho(f"Stylizing completed. Total of {i} utterances available")
    return True


def augment(ctx):
    if ctx['verbose']:
        print("Config: ", json.dumps(ctx, indent=2))

    # Load the data
    stylized_stream = io.load_styled_data(ctx)

    # Split this into three streams
    accent_stream, audio_stream, text_stream, translate_stream = itertools.tee(stylized_stream, 4)

    # accent augmentation
    if ctx['accent']:
        accent_stream = accent_augment(accent_stream, ctx)
        accent_stream = io.record_objs(accent_stream, "accent", ctx)
    else:
        for f in accent_stream:
            pass

    # audio augmentation
    if ctx['wav']:
        audio_stream = audio_augment(audio_stream, ctx)
        audio_stream = io.record_objs(audio_stream, "audio", ctx)
    else:
        for f in audio_stream:
            pass
    
    # Text augmentation
    if ctx['text']:
        text_stream = text_augment(text_stream, ctx)
        text_stream = io.record_objs(text_stream, "text", ctx)
    else:
        for f in text_stream:
            pass

    # Translate augmentation
    if ctx["language"]:
        translate_stream = translate_augment(translate_stream, ctx)
        translate_stream = io.record_objs(translate_stream, "translated", ctx)
    else:
        for f in translate_stream:
            pass

    final_stream = itertools.chain(accent_stream,audio_stream,text_stream,translate_stream)
    final_stream = io.record_objs(final_stream, "FINAL", ctx)

    counter = 0
    for f in final_stream:
        counter += 1

    click.secho(f"Augmentation completed. Total of {counter} utterances available")

    return True


def accent_augment(stream, ctx):
    click.secho(">> Augmenting with accents")
    counter = 0
    accents =["indian", "american", "irish", "australian", "none"] 
    for obj in stream:
        click.secho(f">> Utterance: [{counter}]")
        utterance = obj['styled_utterance']
        for accent in tqdm(accents, desc="Accents", position=1, leave=False):
            if not accent == "none":
                aug = AccentSpeechAugmenter()
                augmented = aug.run(utterance, accent=accent)
                obj['augmented_utterance'] = augmented['variations'][0]
                obj['augmentation_info']=augmented['metadata']
                yield obj
            else:
                obj['augmented_utterance'] = ""
                obj['augmentation_info']= {}   
                yield obj
        counter += 1

def translate_augment(stream, ctx):
    click.secho(">> Augmenting with translations")
    languages = ["de","none","ja","hu","fr","zh"]
    counter = 0
    for obj in stream:
        click.secho(f">> Utterance: [{counter}]")
        utterance = obj['styled_utterance']
        for language in tqdm(languages, desc="Languages", position=1, leave=False):
            if not language == "none":
                aug = TranslationAugmenter(ctx)
                augmented = aug.run(utterance, language=language)
                obj['augmented_utterance'] = augmented['variations'][0]
                obj['augmentation_info']=augmented['metadata']
                yield obj
            else:
                obj['augmented_utterance'] = ""
                obj['augmentation_info']= {}   
                yield obj
        counter += 1



def audio_augment(stream, ctx):
    click.secho(">> Augmenting with audio")
    wavs = ["crop", "mask", "noise", "pitch", "speed", "normalize", "polarity_inversion", "none"]
    counter = 0
    for obj in stream:
        click.secho(f">> Utterance: [{counter}]")
        utterance = obj['styled_utterance']
        for augmenter in tqdm(wavs, desc="Audio", position=1, leave=False):
            if not augmenter == "none":
                aug = AudioSpeechAugmenter()
                augmented = aug.run(utterance, augmenter=augmenter)
                obj['augmented_utterance'] = augmented['variations'][0]
                obj['augmentation_info']=augmented['metadata']
                yield obj
            else:
                obj['augmented_utterance'] = ""
                obj['augmentation_info']= {} 
                yield obj
        counter += 1

def text_augment(stream, ctx):
    click.secho(">> Augmenting with text")
    texts = ["back_translation", "synonym", "span_crop", "contextual_embedding", "none"]
    text_augmentations = {"back_translation":  BackTranslationTextAugmenter,
                          "synonym":  SynonymTextAugmenter,
                          "span_crop":  SpanCropTextAugmenter,
                          "contextual_embedding":  ContextualWordEmbeddingTextAugmenter,
                          "none": "none"}
    counter = 0 
    for obj in stream:
        click.secho(f">> Utterance: [{counter}]")
        utterance = obj['styled_utterance']
        for augmenter in tqdm(texts, desc="Text", position=1, leave=False):
            if not augmenter == "none":
                aug = text_augmentations[augmenter](ctx)
                augmented = aug.run(utterance,1)
                obj['augmented_utterance'] = augmented['variations'][0]
                obj['augmentation_info']=augmented['metadata']
                yield obj
            else:
                obj['augmented_utterance'] = ""
                obj['augmentation_info']= {}   
                yield obj
        counter += 1


def stylize_one(ctx):
    llm = ChatOpenAI(model_name=ctx['model'])
    styles = {"none":  NoStyleAugmenter,
                "directness": DirectnessStyleAugmenter,
                "formality": FormalityStyleAugmenter,
                "disfluency": DisfluencyStyleAugmenter,
                "familiarity": FamiliarityStyleAugmenter,
                "word_choice": WordChoiceStyleAugmenter,
                "asr": ASRStyleAugmenter,
                "correction": CorrectionStyleAugmenter}
    utterances = []
    for name,style in tqdm(styles.items(), desc="Styles"):
        stylizer = style(llm)
        styled_utterances = stylizer.run(ctx['utterance'], ctx['num_per_style'])
        utterances.append(styled_utterances)
    return utterances

               


def stylize_utterance(stream, ctx):
    click.secho(">> Data loaded. Stylizing ...")
    llm = ChatOpenAI(model_name=ctx['model'])
    styles = {"none":  NoStyleAugmenter,
                "directness": DirectnessStyleAugmenter,
                "formality": FormalityStyleAugmenter,
                "disfluency": DisfluencyStyleAugmenter,
                "familiarity": FamiliarityStyleAugmenter,
                "word_choice": WordChoiceStyleAugmenter,
                "asr": ASRStyleAugmenter,
                "correction": CorrectionStyleAugmenter}

    counter = 0
    for obj in stream:
        click.secho(f">> Utterance: [{counter}]")
        # Do style extensions 
        for style in tqdm(ctx['style'], desc="Styles", position=0, leave=False):
            stylizer = styles[style](llm)
            styled_utterances = stylizer.run(obj['utteranceText'], ctx['num_variations'])
            for styled_utterance in tqdm(styled_utterances['variations'], desc="Variations", position=1, leave=False):
                new_item = {'base_utterance': obj['utteranceText'],
                            'styled_utterance': styled_utterance,
                            'stylizer': styled_utterances['metadata']['augmenter'],
                            'semantics': obj['desiredSemantics'],
                            'robot_repertoire': obj['promptInfo']}
                yield new_item
        counter += 1

# TO BE DEPRECATED
def run(ctx):
    if ctx['verbose']:
        print("Config: ",json.dumps(ctx, indent=2))

    # Load up the data 
    click.secho(">> Loading data")
    stream = io.load_data(ctx)
    stream = io.record_objs(stream, "original", ctx)

    
    # Stylize the data and record in a new file 
    click.secho(">> Stylizing data")
    stylized_stream = stylize(stream, ctx)
    stylized_stream = io.record_objs(stylized_stream, "stylized", ctx)
   
    # Split this into three streams
    accent_stream, audio_stream, text_stream = itertools.tee(stylized_stream, 3)

    # accent augmentation
    click.secho(">> Augmenting with accents")
    accent_stream = accent_augment(accent_stream, ctx)
    accent_stream = io.record_objs(accent_stream, "accent", ctx)

    # audio augmentation
    click.secho(">> Augmenting with audio")
    audio_stream = audio_augment(audio_stream, ctx)
    audio_stream = io.record_objs(audio_stream, "audio", ctx)
    
    # Text augmentation
    click.secho(">> Augmenting with text")
    text_stream = text_augment(text_stream, ctx)
    text_stream = io.record_objs(text_stream, "text", ctx)


    final_stream = itertools.chain(accent_stream,audio_stream,text_stream)
    final_stream = io.record_objs(final_stream, "FINAL", ctx)

    for f in final_stream:
        pass

    return True


