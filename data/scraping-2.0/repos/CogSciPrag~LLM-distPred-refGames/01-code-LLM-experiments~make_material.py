import openai
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse
import openai
import random
from pprint import pprint

color    = ["blue"   , "green" , "red"     , "orange" ]
shape    = ["circle" , "square", "triangle", "hexagon"]
texture  = ["stripes", "dots"  , "spades"  , "stars"  ]
features = ["color"  , "shape" , "texture" ]

def my_shuffle(a):
    return(random.sample(a, len(a)))

def stringify_object(o):
    return(o["color"] + " " + o["shape"] + " with " + o["texture"])

def make_context_production(objects, words, target):

    context = f"""Your task is to play a conversation game. There are three objects that you and your friend can see. You have to choose a single word to identify one of the three objects for your friend.

The three objects are:

{objects}

Your task is to make your friend pick out the following target object:

{target}

Which of the following words would you choose:

{words}

Your answer:

I would choose the word """

    return(context)

def make_context_interpretation(objects, words, trigger):

    context = f"""Your task is to play a conversation game. There are three objects that you and your friend can see. Your friend wants to communicate about one of these objects. Your friend selects a single word. Your task is to guess which object your friend is trying to refer to.

The three objects are:

{objects}

Your friend can choose from the following list of words:

{words}

Your friend chose the word:

{trigger}

Which object do you think your friend is trying to refer to?

Your answer:

My friend wants to refer to """

    return(context)


def sample_vignette():
    color_shuffled    = random.sample(color   , len(color))
    shape_shuffled    = random.sample(shape   , len(shape))
    texture_shuffled  = random.sample(texture , len(texture))
    features_shuffled = random.sample(features, len(features))

    nuisance_feature = features_shuffled[-1]

    # utterances

    utterances = [eval( features_shuffled[0] + "_shuffled" )[0],
                  eval( features_shuffled[0] + "_shuffled" )[1],
                  eval( features_shuffled[1] + "_shuffled" )[0],
                  eval( features_shuffled[1] + "_shuffled" )[1]]
    random.shuffle(utterances)
    utterances_string = '\n'.join(utterances)
    # print("utterances:\n", utterances_string)

    objects = [{features_shuffled[0] : eval( features_shuffled[0] + "_shuffled" )[0],
                features_shuffled[1] : eval( features_shuffled[1] + "_shuffled" )[0],
                features_shuffled[2] : eval( features_shuffled[2] + "_shuffled" )[0] },
               {features_shuffled[0] : eval( features_shuffled[0] + "_shuffled" )[1],
                features_shuffled[1] : eval( features_shuffled[1] + "_shuffled" )[0],
                features_shuffled[2] : eval( features_shuffled[2] + "_shuffled" )[0] },
               {features_shuffled[0] : eval( features_shuffled[0] + "_shuffled" )[0],
                features_shuffled[1] : eval( features_shuffled[1] + "_shuffled" )[1],
                features_shuffled[2] : eval( features_shuffled[2] + "_shuffled" )[0] }
               ]
    # objects_shuffled = my_shuffle(objects)
    object_indices = my_shuffle([0,1,2])
    objects_shuffled = [objects[x] for x in object_indices]

    # objects

    objects_indef = ["a " + stringify_object(o) for o in objects_shuffled]
    objects_def   = ["the" + stringify_object(o) for o in objects_shuffled]

    objects_indef_string = '\n'.join(objects_indef)
    objects_def_string   = '\n'.join(objects_def)
    # print(objects_indef_string)

    # production info

    trigger_object       = stringify_object(objects[1])
    trigger_object_index = object_indices.index(1)

    production_target      = eval(features_shuffled[0] + "_shuffled")[1]
    production_competitor  = eval(features_shuffled[1] + "_shuffled")[0]
    production_distractor1 = eval(features_shuffled[0] + "_shuffled")[0]
    production_distractor2 = eval(features_shuffled[1] + "_shuffled")[1]

    production_index_target      = utterances.index(production_target)
    production_index_competitor  = utterances.index(production_competitor)
    production_index_distractor1 = utterances.index(production_distractor1)
    production_index_distractor2 = utterances.index(production_distractor2)

    # interpretation info

    trigger_feature = features_shuffled[0]
    trigger_word    = eval(features_shuffled[0] + "_shuffled")[0]

    interpretation_target     = stringify_object(objects[0])
    interpretation_competitor = stringify_object(objects[2])
    interpretation_distractor = stringify_object(objects[1])

    interpretation_index_target     = object_indices.index(0)
    interpretation_index_competitor = object_indices.index(2)
    interpretation_index_distractor = object_indices.index(1)

    vignette = {
        "objects" : objects_indef_string,
        "utterances" : utterances_string,
        "trigger_feature" : trigger_feature,
        "nuisance_feature" : nuisance_feature,
        "production_target" : production_target,
        "production_competitor" : production_competitor,
        "production_distractor1" : production_distractor1,
        "production_distractor2" : production_distractor2,
        "production_index_target" : production_index_target,
        "production_index_competitor"  : production_index_competitor,
        "production_index_distractor1" : production_index_distractor1,
        "production_index_distractor2" : production_index_distractor2,
        "trigger_object" : trigger_object,
        "trigger_feature" : trigger_feature,
        "trigger_word" : trigger_word,
        "interpretation_target" : interpretation_target,
        "interpretation_competitor" : interpretation_competitor,
        "interpretation_distractor" : interpretation_distractor,
        "interpretation_index_target"     : interpretation_index_target,
        "interpretation_index_competitor" : interpretation_index_competitor,
        "interpretation_index_distractor" : interpretation_index_distractor
    }

    vignette['context_production'] = make_context_production(
        vignette['objects'],
        vignette['utterances'],
        'the ' + vignette['trigger_object']
    )

    vignette['context_interpretation'] = make_context_interpretation(
        vignette['objects'],
        vignette['utterances'],
        vignette['trigger_word']
    )

    return (vignette)

#vignette = sample_vignette()
#pprint(vignette)
