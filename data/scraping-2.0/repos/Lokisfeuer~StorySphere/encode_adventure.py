# !pip install openai==0.28  # TODO: Upgrade your code to most recent version.
# from rnn import train_model
import json
import numpy as np
import openai
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

PRE_ENC_LENGTH = 1050
PRE_RNN_HIDDEN = 2000

TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
MODEL = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

# openai.api_key = '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'

# These two are commented out because they contain boolean lists that need to be written out.
'''
    'mot': {
        'texts': ['zu_was_beschreibung'],
        'bools': ['wie', 'positive_faktoren', 'negative_faktoren'],  # TODO: Expand the factors!
        'scalars': [],
        'single_ids': [],
        'list_ids': ['wer', 'zu_was_fuer_objekten', 'von_wem']
    },
    
    'bea': {
        'texts': ['aussehen'],
        'bools': ['art'],  # TODO: Ausschreiben
        'scalars': ['difficulty'],
        'single_ids': ['wo'],
        'list_ids': []
    },
'''

text_features_to_prompts = {
    'name': 'Give me the name of a fictional character',
    'backstory': 'Give me the backstory of a fictional character',
    'was': 'Give me a short description of what could happen at a fictional scene in a Theatre I am writing',
    'warum': 'Give me conditions for a scene in my self-written theatre to occur like who needs to be on stage',
}

all_features = {
    'sci': {
        'texts': ['name', 'backstory'],
        'bools': ['charakterbogen', 'plaene_fuer_den_charakter', 'hat_eine_backstory'],
        'scalars': [],
        'single_ids': [],
        'list_ids': ['startszene', 'events', 'gruppen', 'backstory_sonstiges']
    },
    'eus': {
        'texts': ['was', 'warum'],
        'bools': ['untersuchen', 'soziale_interaktion', 'fight', 'start'],
        'scalars': ['schwierigkeitsgrad', 'wahrscheinlichkeit'],
        'single_ids': [],
        'list_ids': ['wer', 'wo', 'Gegenstände', 'Geheimnisse', 'personen', 'wer_muss_da_sein', 'wo_kann_das_sein',
                     'motivationen']
    },
    'npc': {
        'texts': ['name', 'backstory'],
        'bools': ['charakterbogen', 'plaene', 'hat_eine_backstory'],
        'scalars': [],
        'single_ids': [],
        'list_ids': ['events_und_szenen', 'gruppen', 'backstory_sonstiges']
    },
    'geh': {
        'texts': ['was'],
        'bools': [],
        'scalars': ['positivitaet'],
        'single_ids': [],
        'list_ids': ['wer_weiss_davon', 'wen_und_was_betrifft_das']
    },
    'gru': {
        'texts': ['grund_des_zusammenhalts'],
        'bools': [],
        'scalars': [],
        'single_ids': ['moegliche_motivation_von_aussen', 'geburtsort_der_gruppe'],
        'list_ids': []
    },
    'geg': {
        'texts': ['was'],
        'bools': [],
        'scalars': ['wert'],
        'single_ids': [],
        'list_ids': ['wessen', 'wo']
    }
}


'''
This class is the central structure for an adventure. It's supposed to be convertible to virtually any other 
possible representation of an adventure. To save this as JSON works already. Currently I am working on a computer 
readable representation of an adventure (in a high-dimensional vector field). Also I have in mind a full text 
representation, maybe a representation that uses a lot of graphics, a representation that would work as a computer 
game like the AI-RPG project, the adventure as a board game and so on.
'''
class Adventure:
    def __init__(self, name):
        self.name = name
        self.sci = ObjectClass('sci',
                               name=str,
                               charakterbogen=bool,
                               plaene_fuer_den_charakter=bool,
                               startszene=(list, str),  # list of events and scenes (where start-scene is true)
                               events=(list, str),  # list of events and scenes
                               gruppen=(list, str),  # list of groups
                               hat_eine_backstory=bool,
                               backstory=str,
                               backstory_sonstiges=(list, str)
                               )
        self.mot = ObjectClass('mot',
                               wer=(list, str),  # list of Persons (PCs and NPCs) and groups
                               zu_was_beschreibung=str,
                               zu_was_fuer_objekten=(list, str),
                               wie=bool,  # always True
                               positive_faktoren=(list, bool),
                               negative_faktoren=(list, bool),
                               # TODO: beide vollständig ausschreiben. Listen sind reserviert für unklar lange Listen.
                               # both factors are exactly 10 bools, each hardcoded to the emotions from the Notizbuch.
                               von_wem=(list, str)  # list of Persons ??
                               )
        self.eus = ObjectClass('eus',
                               wer=list,  # this seems wrong!
                               wo=(list, str),
                               was=str,
                               untersuchen=bool,
                               Gegenstände=(list, str),  # list of Gegenstände
                               Geheimnisse=(list, str),  # list of secrets
                               soziale_interaktion=bool,  # is it a scene of social interaction?
                               personen=(list, str),  # list of persons whose relation to the players might change
                               fight=bool,  # is it a fight scene?
                               schwierigkeitsgrad=float,
                               warum=str,
                               wer_muss_da_sein=(list, str),  # list of persons
                               wo_kann_das_sein=(list, str),  # list of locations
                               start=bool,
                               wahrscheinlichkeit=float,
                               motivationen=(list, str)
                               )
        # TODO: Orte
        self.npc = ObjectClass('npc',
                               name=str,
                               charakterbogen=bool,  # hat einen Charakterbogen?
                               plaene=bool,  # es gibt Zukunftspläne für diesen NPC
                               events_und_szenen=(list, str),  # list of events
                               gruppen=(list, str),  # list of groups
                               hat_eine_backstory=bool,
                               backstory=str,
                               backstory_sonstiges=(list, str)
                               )
        self.geh = ObjectClass('geh',
                               was=str,
                               wer_weiss_davon=(list, str),  # list of Personen
                               wen_und_was_betrifft_das=(list, str),  # list of persons, Gegenstände und Orten
                               positivitaet=float  # how positive is this secret to the players.
                               )
        self.gru = ObjectClass('gru',
                               grund_des_zusammenhalts=str,
                               moegliche_motivation_von_aussen=str,  # ??, ids are strings
                               geburtsort_der_gruppe=str  # roomID, Geburtsort der Gruppe
                               )
        self.bea = ObjectClass('bea',
                               art=(list, bool),  # TODO Ausschreiben!
                               difficulty=float,  # how big of a challenge does this beast pose.
                               wo=str,  # roomIDs
                               aussehen=str
                               )
        self.geg = ObjectClass('geg',
                               wessen=(list, str),  # list of Persons
                               wert=float,
                               was=str,
                               wo=(list, str)  # list of locations
                               )

    def save(self, path='adventure.json'):
        to_save = {}
        for i in [self.sci, self.mot, self.eus, self.npc, self.geh, self.gru, self.bea, self.geg]:
            to_save.update(i.to_save())
        with open(path, 'w+') as f:
            f.write(json.dumps(to_save, indent=4))

    def load(self, path='adventure.json'):
        with open(path, 'r') as f:
            data = json.load(f)
        for i in [self.sci, self.mot, self.eus, self.npc, self.geh, self.gru, self.bea, self.geg]:
            i.all_objects = data[i.name]
            i.id_counter = len(data[i.name])

    def to_list(self):
        to_save = {}
        for i in [self.sci, self.mot, self.eus, self.npc, self.geh, self.gru, self.bea, self.geg]:
            to_save.update(i.to_save())
        return json.dumps(to_save, indent=4)

    def to_text(self):
        return 'Adventure to text doesn\'t really work yet.'


# This class is more or less an add-on to the adventure class.
class ObjectClass:
    def __init__(self, class_name, **features):
        self.name = class_name
        self.features = features
        self.id_counter = 0
        self.all_objects = []

    def add(self, **features_values):
        for i, val in features_values.items():
            if i not in list(self.features.keys()):
                raise ValueError
            else:
                if isinstance(self.features[i], tuple):
                    if not isinstance(val, list):
                        raise ValueError
                    if not isinstance(val[0], self.features[i][1]):
                        raise ValueError
                elif not isinstance(val, self.features[i]):
                    raise ValueError
        object_id = f'id_{self.name[0:3]}_{self.id_counter}'
        features_values.update({'ID': object_id})
        self.id_counter += 1
        self.all_objects.append(features_values)
        return object_id

    def to_save(self):
        return {self.name: self.all_objects}


# This is not up-to-date. It generates a demo-adventure about Max Mustermann.
def demo_adventure():
    adv = Adventure('demo')
    # Max once met a monster which he now meets again in the very first scene.
    # Max wants revenge and intends to kick the monster with his boots.
    # John also exists. He knows that Max once met the monster.
    # John and Max are a group.
    adv.sci.add(
        name='Max',
        charakterbogen=False,
        plaene_fuer_den_charakter=True,
        startszene=['id_Eve_1'],  # list of events and scenes (where start-scene is true)
        # events=[],  # list of events and scenes
        gruppen=['id_Gru_1'],  # list of groups
        hat_eine_backstory=True,
        backstory='This is Max awesome backstory. Max was born in Musterhausen. He was once attacked by a monster.',
        backstory_sonstiges=['id_Bea_1']
    )
    adv.mot.add(
        wer=['id_Spi_1'],  # list of Persons (PCs and NPCs) and groups
        zu_was_beschreibung='Max will sich am Monster rächen indem er es mit seinen Stiefeln tritt.',
        zu_was_fuer_objekten=['id_Geg_1'],
        wie=True,  # always True
        positive_faktoren=[False, False, False, False, False, False, False, False, False, False],
        # exactly 10 bools, each hardcoded to the emotions from the notizbuch
        negative_faktoren=[True, False, True, False, False, False, False, False, True, False],
        # von_wem=(list, str)  # he hasn't been motivated by anyone on the outside.
    )
    adv.eus.add(
        wer=['id_Spi_1', 'id_Bea_1'],
        wo=['id_Ort_1_leidergibtesnochkeineorte'],
        was='Max meets the monster that once attacked him again.',
        untersuchen=False,
        Gegenstände=['id_Geg_1'],  # list of Gegenstände
        Geheimnisse=['id_Geh_1'],  # list of secrets
        soziale_interaktion=False,  # is it a scene of social interaction?
        # personen=(list, str),  # since its no social interaction the SC can't change any social relations.
        fight=True,  # is it a fight scene?
        schwierigkeitsgrad=0.8,
        warum='Max und Monster sind am gleichen Ort.?!',
        # wer_muss_da_sein=(list, str),  # list of persons  # muss nicht unbedingt was hin.
        # wo_kann_das_sein=(list, str),  # list of locations  # dito
        start=True,
        wahrscheinlichkeit=1.,
        motivationen=['id_Mot_1']
    )
    # TODO: Orte
    adv.npc.add(
        name='John',
        charakterbogen=False,  # hat einen Charakterbogen?
        plaene=False,  # es gibt Zukunftspläne für diesen NPC
        events_und_szenen=['id_Eve_1'],  # list of events
        gruppen=['id_Gru_1'],  # list of groups
        hat_eine_backstory=True,
        backstory='John is the one who originally sold Max his boots.',
        backstory_sonstiges=['id_Spi_1', 'id_Geg_1']
    )
    adv.geh.add(
        was='Max once was attacked by the monster in his childhood.',
        wer_weiss_davon=['id_Spi_1', 'id_NPC_1'],  # list of Personen
        wen_und_was_betrifft_das=['id_Spi_1', 'id_Bea_1'],  # list of persons, Gegenstände und Orten
        positivitaet=0.2  # how positive is this secret to the players.
    )
    adv.gru.add(
        grund_des_zusammenhalts='John and Max are very good friends.',
        # moegliche_motivation_von_aussen=str,  # There is no motivation from the outside
        # geburtsort_der_gruppe=str  # roomID, Geburtsort der Gruppe
    )
    adv.bea.add(
        # art=(list, bool),
        difficulty=0.8,  # how big of a challenge does this beast pose.
        wo='id_Ort_1_leidergibtesnochkeineorte',  # roomIDs
        aussehen='This beast is a big Monster that seem really quite threatening.'
    )
    adv.geg.add(
        wessen=['id_Spi_1'],  # list of Persons
        wert=2.,
        was='anti-monster-Boots',
        # wo=[]  # Wo Max halt ist.
    )
    # adv.save('demo_adventure.json')
    return adv


# returns a high-dimensional (1024) vector representation of the passed in sentence.
def roberta(sentence):
    # from https://huggingface.co/sentence-transformers/all-roberta-large-v1

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Load model from HuggingFace Hub
    # I made this global variables because they take years to load so best just do it once.

    # Tokenize sentences
    encoded_input = TOKENIZER(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = MODEL(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # print("Sentence embeddings:")
    # print(sentence_embeddings)
    return sentence_embeddings.tolist()[0]


def rnn_pres(list_of_ids, id_to_pre):
    pass  # this function is supposed to return the output of the RNN encoder when fed by the pre_encoding of the
    # objects of list_of_ids
    return list(range(PRE_RNN_HIDDEN))  # this has the same length


# this function takes an object (by id) and returns an encoding which is either a pre_encoding (ignoring ids) or,
# if id_to_pre is not None the full encoding.
def enc_obj(obj_class, id, id_to_pre=None):
    features = all_features[obj_class.name]
    for i in ['texts', 'bools', 'scalars', 'single_ids', 'list_ids']:
        if i not in features.keys():
            features.update({i: []})
    f_v = obj_class.all_objects[int(id[7:]) - 1]  # =features_values
    enc = []

    # deal with actual texts ; 1024 Values all together
    text = 'This is text.'
    for n in features['texts']:
        if n in f_v.keys():
            text = f'{text}\n{n}: {f_v[n]}'
    text_embedding = roberta(text)
    for i in text_embedding:
        enc.append(i)

    # deal with booleans; 2 Values each
    for n in features['bools']:
        if n in f_v.keys():  # 2 values.
            enc.append(1.)
            if f_v[n]:
                enc.append(1.)
            else:
                enc.append(0.)
        else:
            enc.append(0.)
            enc.append(0.)

    # deal with scalars; 2 Values each
    for n in features['scalars']:
        if n in f_v.keys():
            enc.append(1.)
            enc.append(float(f_v[n]))
        else:
            enc.append(0.)
            enc.append(0.)

    # check length
    expected_length = {'sci': 1030, 'eus': 1036, 'npc': 1030, 'geh': 1026, 'gru': 1024, 'bea': 1028,
                       'geg': 1026}  # TODO: add mot
    if len(enc) != expected_length[obj_class.name]:
        raise ValueError
    # fill up with zeros then return if done.
    for i in range(PRE_ENC_LENGTH - len(enc)):
        enc.append(0)
    if id_to_pre is None:
        return enc

    # deal with single ids; PRE_ENC_LENGTH values each
    for n in features['single_ids']:
        if n in f_v:
            enc.append(1.)
            for i in id_to_pre[f_v[n]]:
                enc.append(i)
        else:
            enc.append(0.)
            for i in range(PRE_ENC_LENGTH):
                enc.append(0.)

    # deal with list of ids; PRE_RNN_HIDDEN values each (=per list)
    for n in features['list_ids']:
        if n in f_v:
            enc.append(1.)
            eve = rnn_pres(f_v[n], id_to_pre)
            for i in eve:
                enc.append(i)
        else:
            enc.append(0.)
            for i in range(PRE_RNN_HIDDEN):
                enc.append(0.)

    return enc


# This function writes an adventure with every mathematically possible object.
def generate_adventure_objs():
    adv = Adventure(name='AllObjects')
    all_options = {}

    for cla in all_features.keys():
        opt = {}
        for b in all_features[cla]['bools']:
            opt.update({b: [False, True]})

        for s in all_features[cla]['scalars']:
            opt.update({s: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

        for t in all_features[cla]['texts']:
            opt.update({t: [t]})
        all_options.update({cla: opt})

    # generate objs:

    def iter_(idcs_, maxs_):
        if idcs_ == []:
            return None, None
        if 0 in maxs_:
            raise ValueError
        idcs_[-1] += 1
        x = 0
        for i in range(len(idcs_)):
            idx = idcs_[-(i + 1)]
            max = maxs_[-(i + 1)]
            x += 1
            if idx == max:
                idcs_[-x] = 0
                if x == len(idcs_):
                    return None, None
                idcs_[-(x + 1)] += 1
        return idcs_, maxs_

    def create_obj(opt, idcs, cla, adv):
        # TODO: Debug: Why is this not called or doesn't work?
        name_to_feat = {'sci': adv.sci, 'mot': adv.mot, 'eus': adv.eus, 'npc': adv.npc, 'geh': adv.geh, 'gru': adv.gru,
                        'bea': adv.bea, 'geg': adv.geg}
        parameter = {}
        for o, idx in zip(opt.items(), idcs):
            if not isinstance(o[1][idx], str):
                parameter.update({o[0]: o[1][idx]})
            else:
                prompt = f'Give me a very short fascinating story consisting of up to five sentences:\n\n'
                # response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=2.,
                #                                    max_tokens=200)
                # response = response['choices'][0]['text']
                # parameter.update({o[0]: response})
                parameter.update({o[0]: prompt})
        # TODO: texts!
        name_to_feat[cla].add(**parameter)

    print('start writing')
    for cla in all_features.keys():
        print(cla)
        opt = all_options[cla]
        idcs = [0 for _ in opt.keys()]
        maxs = [len(i) for i in opt.values()]
        while idcs is not None:
            create_obj(opt, idcs, cla, adv)
            idcs, maxs = iter_(idcs, maxs)

    return adv


# This function generates a handful of objects and prints the result of enc_obj for each.
def test():
    adv = demo_adventure()
    # adv = Adventure(name='demo')
    # adv.load('demo_adventure.json')
    adv.sci.add(
        name='Alfred',
        charakterbogen=True,
        # plaene_fuer_den_charakter=True,
        # startszene=['id_Eve_1'],  # list of events and scenes (where start-scene is true)
        # events=[],  # list of events and scenes
        gruppen=['id_Gru_1'],  # list of groups
        hat_eine_backstory=True,
        backstory='This is Max awesome backstory. Max was born in Musterhausen. He was once attacked by a monster.',
        backstory_sonstiges=['id_sci_1']
    )
    adv.sci.add(
        name='Berta',
        charakterbogen=False,
        plaene_fuer_den_charakter=True,
        startszene=['id_Eve_1'],  # list of events and scenes (where start-scene is true)
        events=['id_Eve_1'],  # list of events and scenes
        gruppen=['id_Gru_1'],  # list of groups
        hat_eine_backstory=True,
        backstory='This is Max awesome backstory. Max was born in Musterhausen. He was once attacked by a monster.',
        backstory_sonstiges=['id_sci_1']
    )
    adv.sci.add()
    print(enc_obj(adv.sci, id='id_spi_1'))
    # print(pre_encode_object(adv.mot, id='id_mot_1'))
    print(enc_obj(adv.eus, id='id_eus_1'))
    print(enc_obj(adv.npc, id='id_npc_1'))
    print(enc_obj(adv.geh, id='id_geh_1'))
    print(enc_obj(adv.gru, id='id_gru_1'))
    print(enc_obj(adv.bea, id='id_bea_1'))
    print(enc_obj(adv.geg, id='id_geg_1'))
    print('Spielercharaktere:')
    print(enc_obj(adv.sci, id='id_spi_1'))
    print(enc_obj(adv.sci, id='id_spi_2'))
    print(enc_obj(adv.sci, id='id_spi_3'))
    print(enc_obj(adv.sci, id='id_spi_4'))


# This function (currently) first cally generate_adventure_objs() to then get the pre-encoding for each object.
# It saves the resulting array and prints its overall length.
def main():
    adv = generate_adventure_objs()
    adv.save(path='all_objects_adv.json')
    name_to_feat = {'sci': adv.sci, 'mot': adv.mot, 'eus': adv.eus, 'npc': adv.npc, 'geh': adv.geh, 'gru': adv.gru,
                    'bea': adv.bea, 'geg': adv.geg}
    i = 0
    all = []
    print('start encoding')
    for name, cla in name_to_feat.items():
        print(name)
        for j in range(cla.id_counter):
            i += 1
            all.append(enc_obj(cla, id=f'id_{name}_{j}'))
    arr = np.array(all)
    np.savetxt('test.csv', arr, delimiter=',')
    np.save("pres.npy", arr)
    print(i)
    # Generate A LOT of adventures and their objects.
    # train RNN with train_model from RNN
    # save the resulting models
    # write function RNN to use these saved model
    # test enc_obj with optional parameter id_to_pre


if __name__ == '__main__':
    main()
