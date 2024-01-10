# export PYTHONPATH=/notebooks/pip_install/
# pip install -U transformers

# export GEVENT_SUPPORT=True for debug in flask mode 
import os
from database.arangodb import NEBULA_DB, DBBase
from typing import NamedTuple
from nebula3_experiments.prompts_utils import *
from experts.pipeline.api import PipelineConf
from nebula3_experiments.vg_eval import VGEvaluation  # in case error of "Failed to import transformers, cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_"  : then pip install requests==2.29.0
from typing import Union
import time
from sklearn.cluster import KMeans
import tqdm
import json
os.environ["TRANSFORMERS_CACHE"] = "/storage/hft_cache"
os.environ["TORCH_HOME"] = "/storage/torch_cache"
os.environ["CONFIG_NAME"] = "default" 

os.environ["ARANGO_DB"]="ipc_200"
nebula_db = NEBULA_DB()

VISUAL_CLUES_COLLECTION = 's4_visual_clues'
REID_CLUES_COLLECTION = 's4_re_id'
MOVIES_COLLECTION = "Movies"
FUSION_COLLECTION = "s4_fusion"
LLM_OUTPUT_COLLECTION = "s4_llm_output"
KEY_COLLECTION = "llm_config"
FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
# FS_SAMPLES = 5                   # Samples for few-shot gpt

from abc import ABC, abstractmethod
import openai
from itertools import compress
from huggingface_hub.inference_api import InferenceApi
import requests

import importlib


#sys.path.insert(0, "/notebooks/fast_demo/")
#sys.path.insert(0, "/notebooks/fast_demo/vidarts_advanced_main/")


from blip2_service import BLIP2Service
blip2_service = BLIP2Service("http://209.51.170.37:8087/infer")

def is_whole_word_within_caption(substring, string):
    for word in string.split():
        if substring == word:
            return True
    return False

def where_whole_word_within_caption(substring, string):
    for word in string.split():
        if substring == word:
            return True
    return False

def callback_blip2_caption_extract(url):
    # # Inputs
    texts = [""]
    # bboxes = [[53.04999923706055, 199.6999969482422, 127.80999755859375, 396.29998779296875], [100.77999877929688, 209.41000366210938, 170.8800048828125, 380.7200012207031], [632.1799926757812, 124.19999694824219, 1019.760009765625, 743.0499877929688], [1599.1800537109375, 372.8900146484375, 1624.800048828125, 401.55999755859375], [1780.93994140625, 332.8999938964844, 1864.1199951171875, 438.6600036621094], [1406.1400146484375, 331.239990234375, 1527.050048828125, 444.1300048828125], [1610.6300048828125, 349.67999267578125, 1672.9599609375, 440.4100036621094], [1542.489990234375, 348.260009765625, 1655.1400146484375, 442.3699951171875], [744.3300170898438, 520.1500244140625, 843.3699951171875, 748.6900024414062], [996.4600219726562, 220.19000244140625, 1397.4300537109375, 940.010009765625], [979.1500244140625, 620.7100219726562, 1524.0, 948.0], [18.440000534057617, 727.5999755859375, 956.8499755859375, 948.969970703125], [928.1699829101562, 759.6500244140625, 1581.5799560546875, 945.5599975585938]]
    opportunities = 10
    while (opportunities):
        outputs = blip2_service.get_url_response(url, texts)
        if not outputs:
            time.sleep(1)
            opportunities -= 1
            print("BLIP2 service may be down")
            continue
        else:
            break

    return outputs[0]


print(importlib.metadata.version('openai'))

def flatten(lst): return [x for l in lst for x in l]

# Hosted Inference API :HF HTTP request
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Model Hub is where the members of the Hugging Face community can host all of their model checkpoints 
# hf = HuggingFaceHub(repo_id="google/flan-t5-xl")
# if 0:
#     hf_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", device_map="auto", torch_dtype=torch.bfloat16)
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
#     model = HuggingFaceLLM(hf_model, tokenizer)


api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
nre = DBBase()

def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_input_type_from_db(pipeline_id, collection):
    pipeline_data = nre.get_doc_by_key({'_key': pipeline_id}, collection)
    if pipeline_data:
        if "dataset" in pipeline_data["inputs"]["videoprocessing"]:
            input_type = pipeline_data["inputs"]["videoprocessing"]["dataset"]["type"]
        else:
            input_type = pipeline_data["inputs"]["videoprocessing"]["movies"][0]["type"]
    else:
        return -1
    return input_type

class SummarizeScene():
    def __init__(self, prompting_type: str='few_shot', gpt_type: str='gpt-3.5-turbo-16k',
                    semantic_token_deduplication: bool=True,
                    min_shots_for_semantic_similar_dedup: int=40, write_res_to_db: bool=True,  
                    verbose: bool=False):
        
        self.revert_dummy_names = True
        self.prompt_building_type = 'caption_with_numbers'
        self.gpt_temperatue = 0 # 0 causing leaking of the 1-shot into the generation
        self._one_place_based_scene = True
        self.append_to_db = False
        self.write_res_to_db = write_res_to_db
        self.gpt_type = gpt_type
        self.prompting_type = prompting_type
        self.semantic_token_deduplication = semantic_token_deduplication
        self.verbose = verbose
        self.one_shot_context_ex_prefix_summary = '''Video Summary: 3 persons, in a room, Susan, Michael, & Tom. They look strange, Tom with a giant head, michael with a mask, one of them is giant. The three people appear very tense as they look around frantically. '''
        self.one_shot_context_ex_prefix_caption = '''Caption 1: Susan standing in front of a tv screen with a camera. 
                            Caption 2: Michael with a body in the middle of the screen. 
                            Caption 3: Tom standing next to a giant human in the shape of a human. 
                            Caption 4: Susan standing in front of a camera and looking at the camera. 
                            Caption 5: Susan in a blue dress and a man in a suit. 
                            Caption 6: Susan in a room with a bike in the background. 
                            Caption 7: Tom sitting on a chair in the middle of a room. 
                            Caption 8: Michael with a mask on his face and Michael with a mask on his. 
                            Caption 9: Tom in a suit and tie with a giant head. 
                            Caption 10: Michael in a suit with the capt that says, i'm. 
                            Caption 11: Tom in a spider suit sitting on a train car. 
                            Caption 12: Tom in a blue shirt and Tom in a black shirt. 
                            Caption 13: Susan standing in front of a bookcase with a bookcase in the background. {}.'''.format(self.one_shot_context_ex_prefix_summary)

        self.one_shot_context_ex_prefix_caption = self.one_shot_context_ex_prefix_caption.replace('\n ',' ').replace("  ", "")
# Follow the LLM used for fusion
        self.settings = PipelineConf()
        self.llm = self.settings["llm_fusion"]
        if 'chat' in self.llm:
            self.gpt_type = 'gpt-3.5-turbo-16k'
        else:
            self.gpt_type = self.llm

        self.one_shot_context_ex_prefix_then = '''Susan standing in front of a tv screen with a camera and then Michael with a body in the middle of the screen and then Tom standing next to a giant human in the shape of a human and then Susan standing in front of a camera and looking at the camera and then Susan in a blue dress and a man in a suit and then Susan in a room with a bike in the background and then Tom sitting on a chair in the middle of a room and then Michael with a mask on his face and Michael with a mask on his and then Tom in a suit and tie with a giant head and then Michael in a suit with the capt that says,'i'm ' and then Tom in a spider suit sitting on a train car and then Tom in a blue shirt and Tom in a black shirt and then Susan standing in front of a bookcase with a bookcase in the background. {}.''' \
        .format(self.one_shot_context_ex_prefix_summary)

        # self.places = 'indoor'
        self.top_k_per_mdf = 1
        cluster_based_place = True
        if self.verbose:
            print("promting_type", self.prompting_type)


        if self.gpt_type == 'HF_':
            InferenceApi(repo_id="gpt-j-6b-shakespeare", token=api_token)
        elif self.gpt_type == 'chat_gpt_3.5' or self.gpt_type == 'gpt-4' or self.gpt_type == 'gpt-3.5-turbo-16k':
            self.chatgpt = ChatGptLLM()
        # elif self.gpt_type == 'text-davinci-003':
        #     context_win = 4096
        if self.semantic_token_deduplication:
            self.evaluator = VGEvaluation()
        else:
            evaluator = None

        self.min_shots_for_semantic_similar_dedup = min_shots_for_semantic_similar_dedup
        self.collection_name = 's4_scene_summary'

        with open(os.path.join('multi_modal','place365_ontology.json'), "r") as f:
            self.places_fixed_ontology = json.load(f)

        return

    def _most_probable_place(self, mdf_no, movie_id, top_k_places=2):

        
        ranking_method = 'avg_retrive_score'
        all_scene = list()
        places_w_score = dict()
        rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION) # + scene_elements
        scene_elements = rc_movie_id['scene_elements']
        for ix, frame_num in enumerate(mdf_no):
            # TODO per SE clustering 
            mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
            obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
            
            for x in range(len(obj['global_scenes']['blip'])):
                place = obj['global_scenes']['blip'][x][0]
                
                if not(place in self.places_fixed_ontology):
                    print("Place omitted out of list", place)
                    continue
                if ranking_method == 'avg_retrive_score':
                    place_sim_score = obj['global_scenes']['blip'][x][1]
                elif ranking_method == 'avg_retrive_rank':
                    place_sim_score = x + 1
                else:
                    raise

                if isinstance(place_sim_score, str):
                    place_sim_score = eval(place_sim_score)

                if place in places_w_score:
                    places_w_score[place].append(place_sim_score)
                else:
                    places_w_score[place] = [place_sim_score]
            # print(place, places_w_score[place])

        place_avg_score = dict()

        if ranking_method == 'avg_retrive_score':
            max_avg_dist = -1 # Max score
        elif ranking_method == 'avg_retrive_rank':
            max_avg_dist = -1000 # Min rank
        else:
            raise

        max_avg_dist_place = []
        for k, v in places_w_score.items():

            if ranking_method == 'avg_retrive_score':
               score_of_interest = np.sum(np.array(v))/len(mdf_no) # Max score
            elif ranking_method == 'avg_retrive_rank':
                score_of_interest = -(np.sum(np.array(v))/len(mdf_no)) /len(v)
                # score_of_interest = len(v)

            else:
                raise

            # print("key val ,freq, score_of_interest", k, -np.sum(np.array(v))/len(mdf_no), len(v), score_of_interest)
            
            if score_of_interest > max_avg_dist:
                if ranking_method == 'avg_retrive_score':
                    max_avg_dist = score_of_interest
                elif ranking_method == 'avg_retrive_rank':
                    max_avg_dist = score_of_interest
                else:
                    raise

            place_avg_score[k] = score_of_interest

        print("Best", max(place_avg_score, key=place_avg_score.get))
        top_k_places_avg_score = sorted(place_avg_score, key=place_avg_score.get)[-top_k_places:][::-1]
        print("Top {} places {}".format(top_k_places, top_k_places_avg_score))
            

        return top_k_places_avg_score
# Semantic de-dupllication
    def _places_semantic_dedup(self, mdf_no: list[int], movie_id: str):
        # Place voting
        all_scene = list()
        place_per_scene_elements = dict()
        rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION) # + scene_elements
        scene_elements = rc_movie_id['scene_elements']

        for ix, frame_num in enumerate(mdf_no):
            # TODO per SE clustering 
            mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
            obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
            scene = [obj['global_scenes']['blip'][x][0] for x in range(len(obj['global_scenes']['blip']))][:self.top_k_per_mdf]
            # print([obj['global_scenes']['blip'][x] for x in range(len(obj['global_scenes']['blip']))][:1])
            all_scene.append(scene)
            scene_boundary = [x for x in scene_elements if (frame_num >= x[0] and frame_num < x[1])][0]
            if str(scene_boundary) in place_per_scene_elements:
                place_per_scene_elements[str(scene_boundary)].extend(scene)
            else:
                place_per_scene_elements[str(scene_boundary)] = scene

        all_scene = flatten(all_scene)
        uniq_places, cnt = np.unique(all_scene, return_counts=True)
        # n_scenes_by_length = max(1+int(len(mdf_no)/50), uniq_places.shape[0]) #actually it doesn;t do a thing
        # scene_top_k_frequent = uniq_places[np.argmax(cnt)] # take most frequent place
        scene_top_k_frequent = uniq_places[np.argsort(cnt)[::-1]]#[:n_scenes_by_length] 
        if 0:
            semantic_similar_places_max_set = self._semantic_similar_places_max_set_cover(tokens=all_scene)
        
        if self.semantic_token_deduplication and len(place_per_scene_elements) > self.min_shots_for_semantic_similar_dedup:
            scene_top_k_frequent = self._merge_semantic_similar_tokens(tokens=all_scene)

        if self.verbose and (self.semantic_token_deduplication and len(place_per_scene_elements) < self.min_shots_for_semantic_similar_dedup):
            print('Too short clip/scene to filter places by semantic simillarity')


        if 0: # unittest
            for i in np.arange(9,11,1):
                locals()['all_centroids_places_' + str(i)], locals()['sum_square_within_dist_' + str(i)], _ = self._cluster_based_place_inference(kmeans_n_cluster=i)
                
        return scene_top_k_frequent

# pre_defined_mdf_in_frame_no : given specific frames to process over that matches MDF file name 
    def summarize_scene_forward(self, movie_id: str, frame_boundary: list[list]= [], 
                                caption_type='vlm', append_to_db: bool=False):

        print("summarize_scene_forward : movie_id {} frame_boundary {} caption_type {}".format(movie_id, frame_boundary, caption_type))
        
        self.append_to_db = append_to_db
        if caption_type != 'vlm' and caption_type != 'dense_caption' and caption_type != 'blip2':
            print("Unknown caption type option given : {} but should be (vlm/dense_caption)".format(caption_type))
            return -1

        # if caption_callback and caption_type != 'vlm':
        #     print("You gave callback function for VLM but not defining vlm asan option !!!!!")

        if frame_boundary != []:
            if not (any(isinstance(el, list) for el in frame_boundary)):
                print("Frame boundary structure error : should be [[fstart1, fstop1][fstart2, fstop2]]")
            all_summ = list()
            all_mdf_no = list()
            for scn_frame in range(len(frame_boundary)):
                if len(frame_boundary[scn_frame]) == 2:
                    summ, mdf_no = self._summarize_scene_forward_scene(movie_id, frame_boundary[scn_frame], 
                                        caption_type=caption_type)
                    all_summ.append(summ)
                    all_mdf_no.append(mdf_no)
                else:
                    print("Warning frame start/stop is missing need to supply 2 elements", frame_boundary[scn_frame])
            if self.write_res_to_db:
                self._insert_json_to_db(movie_id, all_summ, all_mdf_no)
            
            return all_summ
        else:
            summ, mdf_no = self._summarize_scene_forward_scene(movie_id, caption_type=caption_type)
            if self.write_res_to_db:
                self._insert_json_to_db(movie_id, summ, mdf_no)
            
            return summ
        

    def _summarize_scene_forward_scene(self, movie_id: str, frame_boundary: list[int]= [], caption_type:str= 'vlm'):

        caption_type = caption_type.lower()
        all_caption = list()
        all_reid_caption = list()
        all_global_tokens = list()
        all_obj_LLM_OUTPUT_COLLECTION_cand = list()
        all_obj_LLM_OUTPUT_COLLECTION_cand_re_id = list()
        try:
            rc_movie_id = nebula_db.get_doc_by_key({'_id': movie_id}, MOVIES_COLLECTION) # + scene_elements
            input_type = get_input_type_from_db(rc_movie_id['pipeline_id'], "pipelines")
            if input_type != -1: # if pipeline Id hasn;'t found result is -1
                if input_type == 'image':
                    print("Movie_id {} type image is not supported but only videos".format(movie_id))
                    return "Movie_id {} type image is not supported but only videos".format(movie_id), -1
            # scene_elements = rc_movie_id['scene_elements']
            movie_name = os.path.basename(rc_movie_id['url_path'])
            self.movie_name = movie_name
            rc_reid = nebula_db.get_doc_by_key({'movie_id': movie_id}, REID_CLUES_COLLECTION)
            rc_reid_fusion = nebula_db.get_doc_by_key2({'movie_id': movie_id}, FUSION_COLLECTION)
        except Exception as e:
            print(e)        
            return e, -1

        mdf_no = sorted(flatten(rc_movie_id['mdfs']))
        if frame_boundary != []:
            if np.where(np.array(mdf_no) == frame_boundary[0])[0].size == 0:
                print('MDF of the given frame boundaries are inccorrect or not found with respect to that movie', frame_boundary)
                return 'MDF of the given frame boundaries are inccorrect or not found with respect to that movie {}'.format(frame_boundary), -1
            mdf_no = mdf_no[np.where(np.array(mdf_no) == frame_boundary[0])[0][0] :1 + np.where(np.array(mdf_no) == frame_boundary[1])[0][0]]

# Actors name 
        celeb_id_name_dict = dict()
        if rc_reid_fusion:
            print("Found actors names in DB")            
            celeb_id_name  = [{int(rec['rois'][0]['face_id']): rec['rois'][0]['reid_name']} for rec in rc_reid_fusion if (int(rec['frame_num']) >=mdf_no[0] and int(rec['frame_num']) <=mdf_no[-1])]
            for f in celeb_id_name:
                if len(list(f.values())[0]) > 0: # sometimes dict has key and value is ''
                    celeb_id_name_dict.update(f)   # Uniqeness actor name dict         

        print("Celeb list", celeb_id_name_dict)
        # all_ids = list()
        all_ids_dummy_and_celeb_names = list()

        if self._one_place_based_scene:
            scene_top_k_frequent = self._most_probable_place(mdf_no, movie_id=movie_id)
        else:
            scene_top_k_frequent = self._places_semantic_dedup(mdf_no, movie_id=movie_id)
        self.scene_top_k_frequent = scene_top_k_frequent
# is indoor 

        # is_indoor = any([True if x in  scene_top_k_frequent else False for x in ['lab', 'room', 'store', 'indoor', 'office', 'motel', 'home', 'house', 'bar', 'kitchen']])    #https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
        # if is_indoor: # @@HK TODO TOP-gun has faces w/o outdoor hence MDF based on faces only is not a good option hence any() =>all()
        #     reid = True
        dummy_name_2_gender = dict()
        gender_count_for_dummy_names = dict()

        for ix, frame_num in enumerate(tqdm.tqdm(mdf_no)):
                
            mid = MovieImageId(movie_id=movie_id, frame_num=frame_num)
            obj = nebula_db.get_movie_frame_from_collection(mid, VISUAL_CLUES_COLLECTION)
            
            if caption_type == 'vlm':            
                caption = obj['global_caption']['blip']
            elif caption_type == 'dense_caption':
                caption = nebula_db.get_movie_frame_from_collection(mid,LLM_OUTPUT_COLLECTION)['candidate']
                all_obj_LLM_OUTPUT_COLLECTION_cand.append(caption)
            elif caption_type == 'blip2':
                caption = callback_blip2_caption_extract(obj['url'])
            else:
                raise 
                
            # scene = obj['global_scenes']['blip'][0][0]
            # all_scene.append(scene)
            all_global_tokens.extend([x[0] for x in obj['global_objects']['blip']])
            # mdf_re_id_dict = rc_reid['frames'][ix]
            
            caption_re_id = self._merge_reid_into_caption(rc_reid=rc_reid, frame_num=frame_num, 
                                                            caption=caption, all_ids_dummy_and_celeb_names=all_ids_dummy_and_celeb_names,
                                                            celeb_id_name_dict=celeb_id_name_dict,
                                                            dummy_name_2_gender=dummy_name_2_gender,
                                                            gender_count_for_dummy_names=gender_count_for_dummy_names)

            if caption_re_id:
                all_reid_caption.append(caption_re_id)
            else:
                all_reid_caption.append(caption)

        # if all_reid_caption:
        seq_caption = ' and then '.join(all_reid_caption)
        seq_caption_w_caption = ''.join([' Caption ' + str(ix+1) + ': ' + x  for ix, x in enumerate(all_reid_caption)])
        # n_uniq_ids = np.unique(all_ids).shape[0]
        id_str_summ = ''
        if celeb_id_name_dict:
            re_id_celeb_names = [v for k,v in celeb_id_name_dict.items()]
            unrecognized_id = np.array(list(set(np.unique(flatten(all_ids_dummy_and_celeb_names))) - set(re_id_celeb_names)))
            recognized_id_str = ' and '.join(re_id_celeb_names)
            if len(re_id_celeb_names) > 0:
                celeb_id_str = "including {}".format(recognized_id_str)
            else:
                celeb_id_str = ''
                print("how come ???")
            if unrecognized_id.size>0:
                id_str_summ = "The scene shows {} main characters, {} and {} unrecognized one/s".format(unrecognized_id.shape[0] + len(re_id_celeb_names), celeb_id_str, unrecognized_id.shape[0] )
            else:
                id_str_summ = "The scene shows {} main characters, {}".format(len(re_id_celeb_names), celeb_id_str)
            # all_ids_dummy_and_celeb_names.extend([re_id_celeb_names])
        else:
            unrecognized_id = np.unique(flatten(all_ids_dummy_and_celeb_names)) # no celeb and no unrecognized i.e no main characters 
            if unrecognized_id.size>0:
                id_str_summ = "The scene shows {} main unrecognized characters".format(unrecognized_id.shape[0] )

        all_actor_names = np.unique(flatten(all_ids_dummy_and_celeb_names))
        n_uniq_ids = len(all_actor_names)


        if self.prompting_type == 'zeroshot':
            prompt = '''Summarize the video scene given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : {} Summary :'''.format(scene_top_k_frequent, n_uniq_ids, seq_caption_w_caption)
            # prompt = "Give a concise summary of the following video scene captions separated by the word 'then':{} Summary :".format(seq_caption)
        elif self.prompting_type == 'few_shot':
            prolog_refine = ', by 2-3 sentences, '
            prolog_refine = ', by 3-4 sentences, '
            prompt_prefix_caption = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption_w_caption, scene_top_k_frequent, n_uniq_ids, all_actor_names=all_actor_names, 
                                                    in_context_examples=self.one_shot_context_ex_prefix_caption, few_shot_seperator = '''###''',
                                                    prolog_refine=prolog_refine, uniq_id_prior_put_in_caption_end=True)
            prompt_prefix_then = get_few_shot_prompt_paragraph_based_to_tuple_4K(seq_caption, scene_top_k_frequent, n_uniq_ids, all_actor_names=all_actor_names, 
                                                        in_context_examples=self.one_shot_context_ex_prefix_then, few_shot_seperator = '''###''',
                                                        prolog_refine=prolog_refine, uniq_id_prior_put_in_caption_end=True)
            
            self.prompt_prefix_caption = prompt_prefix_caption
            self.prompt_prefix_then = prompt_prefix_then
            # https://github.com/NEBULA3PR0JECT/nebula3_llm_task/blob/8254fb4bb1f81ae87ece51f91cf76d5a778ed6f1/llm_orchestration.py#LL545C31-L548C34
        else:
            raise
        
        if self.prompt_building_type == 'caption_and_then_caption':
            prompt_final = self.prompt_prefix_then
        elif self.prompt_building_type == 'caption_with_numbers':
            prompt_final = self.prompt_prefix_caption
        else:
            raise ValueError("Not valid option : prompt_building_type")

        gen_summ = self._generate_summry(prompt_final)
        if self.revert_dummy_names and bool(dummy_name_2_gender):
            for k,v in dummy_name_2_gender.items():
                gen_summ = gen_summ.replace(k, v)

        if n_uniq_ids >0 and self.revert_dummy_names:
            if 'The video features' in gen_summ:
                start_char_summ_ix = gen_summ.find('The video features')
                endof_char_summ = gen_summ.find(",", start_char_summ_ix)
                gen_summ = gen_summ[endof_char_summ+2:]
            gen_summ  = '''{}. {}'''.format(id_str_summ, gen_summ)  # main character. {}'''.format(n_uniq_ids, rc[0]) 
        
        return gen_summ, mdf_no         

    def _dummy_names_to_gender_create(self, dummy_name_2_gender:dict, ids_dummy_names:str, 
                                        gender:str, gender_count_for_dummy_names:dict):
        
        if ids_dummy_names not in dummy_name_2_gender:
            # [v if gender in v for k,v in dummy_name_2_gender] # already has a lady or a man
            if gender in gender_count_for_dummy_names: # this gender has already been counted 
                gender_count_for_dummy_names[gender] = gender_count_for_dummy_names[gender] + 1
                dummy_name_2_gender[ids_dummy_names] = gender + '-' + str(gender_count_for_dummy_names[gender])
            else:
                dummy_name_2_gender[ids_dummy_names] = gender + '-1'
                gender_count_for_dummy_names[gender] = 1


    def _merge_reid_into_caption(self, rc_reid:dict, frame_num: int, caption: str, 
                                    all_ids_dummy_and_celeb_names, celeb_id_name_dict: dict, 
                                    dummy_name_2_gender: dict, gender_count_for_dummy_names:dict):
        
        man_names = list(np.unique(['James', 'Allan', 'Ron', 'George' ,'Nicolas', 'John', 'daniel', 'Henry', 'Jack', 'Leo', 'Oliver']))
        woman_names = list(np.unique(['Jane', 'Jennifer', 'Eileen', 'Sandra', 'Emma', 'Charlotte', 'Mia']))

        mdf_re_id_dict = [x  for x in rc_reid['frames'] if x['frame_num']==frame_num]

        ided_name_are_in_caption = []
        if celeb_id_name_dict:
            ided_name_are_in_caption = [v in caption for k,v in celeb_id_name_dict.items() if len(v)!=0][0]
# TODO for BLIP2 based caption that adds wrong celeb name ided_name_are_in_caption will be empty and it will leak in !!!
        if (mdf_re_id_dict) and not(ided_name_are_in_caption): #and is_indoor: #places == 'indoor':  # conditioned on man in the scene if places==indoor
            reid = True
            assert(mdf_re_id_dict[0]['frame_num'] == frame_num)
            for id_rec in mdf_re_id_dict: # match many2many girl lady, woman to IDs at first
                if 'face_no_id' in id_rec:
                    pass # TBD
                    
                if 're-id' in id_rec:
                    ids_n = id_rec['re-id']
                    caption_re_id = None
                    if ids_n: # in case face but no Re_id, skip
    #TODO @@HK a woaman in 1st scene goes to Id where same ID can appears later under" persons" 
    # Movies/-6576299517238034659 'a man in a car looking at Susan in the back seat" However there only 2 IDs "a man in a car looking at a woman in the back seat" no woman!! ''two men in a red car, one is driving and the other is driving'' but only 1 ID is recognized so ? 
                        # all_ids.extend([ids['id'] for ids in ids_n])
                        # Gender exclusive
                        male_str = ['man', 'person', 'boy', 'human', 'someone']  #TODO add 'someone' to man/woman if they have celeb name remove the "a boy" add driver  : complicated caption : "Two men are sitting in a car. The driver looks directly at the camera while the man in the passenger seat talks to him. The window to the driver's right is partially rolled down."
                        female_str = ['woman', 'lady' , 'girl', 'she']
                        male_female_str = list()
                        male_female_str.extend(male_str)
                        male_female_str.extend(female_str)

                        many_person_str = ['men', 'women', 'persons', 'people']

                        is_male = list(compress(male_str, [caption.find(x)>0 for x in male_str]))
                        is_female = list(compress(female_str, [caption.find(x)>0 for x in female_str]))
                        ids_dummy_names = []
                        # if len(ids_n) > 1 or 1:
                        if 'men ' in caption.lower():
                            ids_dummy_names = [celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]
                            ids_phrase = ', including ' + ' and '.join(ids_dummy_names) + ','
                            caption_re_id = caption.lower().replace('men', 'men' + ids_phrase) 
                        elif 'women ' in caption.lower():
                            ids_dummy_names = [celeb_id_name_dict.get(ids['id'], woman_names[ids['id']]) for ids in ids_n]
                            ids_phrase = ', including ' + ' and '.join(ids_dummy_names) + ','
                            caption_re_id = caption.lower().replace('women', 'women' + ids_phrase) 
                            # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('women', 'women' + ids_phrase)
                        elif 'person ' in caption.lower():
                            ids_dummy_names = [celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]
                            ids_phrase = ', including ' + ' and '.join(ids_dummy_names) + ','
                            caption_re_id = caption.lower().replace('person', 'person' + ids_phrase)
                            # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', 'person'  +ids_phrase)
                        elif 'people ' in caption.lower():  # @@HK to test effect  on Top gun re-id_frame0032 (a group of people sitting in an airplane with a man in the middle of the : Id within the people and one is with man need to refine)
                            ids_dummy_names = [celeb_id_name_dict.get(ids['id'], man_names[ids['id']]) for ids in ids_n]
                            ids_phrase = ', including ' + ' and '.join(ids_dummy_names) + ','
                            caption_re_id = caption.lower().replace('people', 'people with' + ids_phrase)
                        if ids_dummy_names:
                            all_ids_dummy_and_celeb_names.append(ids_dummy_names)
                            caption = caption_re_id # Save for the following single man addresing avoiding overwriden the IDs replaced here
                        
                        

                            # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('people', 'people'  +ids_phrase)
# @@TODO : 1st singular IDs man/woman placing than plural and should be mutual exclusive , hence manage a list when you can take out items following placing singular IDS
# The people will by Id[0] + people and Id[1] instead of man =>gender classification needed, or ID with celeb name can cover up the whole issue
                        # ids = id_rec['re-id'][0]  # TODO take the relavant Gender based ID out of the IDs in the MDF
                        for ids in id_rec['re-id']:
                            ref_per_id_found = False
                    # elif len(ids_n) == 1:
                        # ids = id_rec['re-id'][0]
                            for female_key in male_str: 
                                if is_whole_word_within_caption(female_key, caption.lower()) or is_whole_word_within_caption(str(female_key+"'s"), caption.lower()) or is_whole_word_within_caption(str("a " + female_key), caption.lower()): # avoid treating the sign "Burnman Associates
                                    ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
                                    if str("a " + female_key)  in caption.lower(): # you can't tell for sure if 2 man in caption relate to same man : like two man are driving hence replace man per re_id num
                                        caption_re_id = caption.lower().replace("a " + female_key, ids_dummy_names, 1) #caption.lower().replace('a man', ids_dummy_names, 1).replace('man', ids_dummy_names, 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', man_names[ids['id']], 1)# TODO the obj_LLM_OUTPUT_COLLECTION_cand can chnage the a man to the man 
                                    else:
                                        caption_re_id = caption.lower().replace(female_key, ids_dummy_names, 1)
                                        # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
                                    all_ids_dummy_and_celeb_names.append([ids_dummy_names])
                                    caption = caption_re_id # otherwise in case 2 Ids but only one "man" the "man" will be replaced twice !!
    # Only if the Id name is dummy i.e is not one of the celeb  and wasn't registered yet                                
                                    if not(ids_dummy_names in celeb_id_name_dict.values()) and not(ids_dummy_names in dummy_name_2_gender):
                                        self._dummy_names_to_gender_create(dummy_name_2_gender, ids_dummy_names, female_key, gender_count_for_dummy_names)
                                    
                                    ref_per_id_found = True # for the 'he' reference
                                    break # Only one reference per ID if has found 

                            if is_whole_word_within_caption('he', caption.lower()) and not ref_per_id_found: # avoid treating the sign "Burnman Associates
                                ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])

                                ind_he_loc = caption.lower().split().index('he')
                                split_caption = caption.lower().split()
                                split_caption[ind_he_loc] = ids_dummy_names
                                caption_re_id = ' '.join(split_caption)
                                # caption_re_id = caption.lower().replace('he', ids_dummy_names, 1)
                                    # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
                                all_ids_dummy_and_celeb_names.append([ids_dummy_names])
                                caption = caption_re_id # otherwise in case 2 Ids but only one "man" the "man" will be replaced twice !!
# Only if the Id name is dummy i.e is not one of the celeb  and wasn't registered yet                                
                                if not(ids_dummy_names in celeb_id_name_dict.values()) and not(ids_dummy_names in dummy_name_2_gender):
                                    self._dummy_names_to_gender_create(dummy_name_2_gender, ids_dummy_names, female_key, gender_count_for_dummy_names)
# Special treatment for he as part of she and others
                            if not ref_per_id_found:
                                for female_key in female_str:
                                    if is_whole_word_within_caption(female_key, caption.lower()) or is_whole_word_within_caption(str(female_key+"'s"), caption.lower()) or is_whole_word_within_caption(str("a " + female_key), caption.lower()): # avoid treating the sign "Burnman Associates
                                        ids_dummy_names = celeb_id_name_dict.get(ids['id'], woman_names[ids['id']])
                                        if str("a " + female_key)  in caption.lower(): # you can't tell for sure if 2 man in caption relate to same man : like two man are driving hence replace man per re_id num
                                            caption_re_id = caption.lower().replace("a " + female_key, ids_dummy_names, 1) #caption.lower().replace('a man', ids_dummy_names, 1).replace('man', ids_dummy_names, 1)
                                            # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', man_names[ids['id']], 1)# TODO the obj_LLM_OUTPUT_COLLECTION_cand can chnage the a man to the man 
                                        else:
                                            caption_re_id = caption.lower().replace(female_key, ids_dummy_names, 1)
                                            # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
                                        all_ids_dummy_and_celeb_names.append([ids_dummy_names])
                                        caption = caption_re_id # otherwise in case 2 Ids but only one "man" the "man" will be replaced twice !!
        # Only if the Id name is dummy i.e is not one of the celeb  and wasn't registered yet                                
                                        if not(ids_dummy_names in celeb_id_name_dict.values()) and not(ids_dummy_names in dummy_name_2_gender):
                                            self._dummy_names_to_gender_create(dummy_name_2_gender, ids_dummy_names, female_key, gender_count_for_dummy_names)

                                        ref_per_id_found = True # for the 'he' reference
                                        break # Only one reference per ID if has found 

#                             if 'woman' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], woman_names[ids['id']])
#                                 if 'a woman' in caption : 
#                                     caption_re_id = caption.lower().replace('a woman', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a woman', woman_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.lower().replace('woman', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                                 # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('woman', woman_names[ids['id']])
#                             elif 'lady' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], woman_names[ids['id']])
#                                 if 'a lady' in caption:
#                                     caption_re_id = caption.lower().replace('a lady', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a lady', woman_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.lower().replace('lady', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('lady', woman_names[ids['id']])
#                             elif 'girl' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], woman_names[ids['id']])
#                                 if 'a girl' in caption:
#                                     caption_re_id = caption.lower().replace('a girl', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a girl', woman_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.replace('girl', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                                         # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('girl', woman_names[ids['id']])
#                             elif 'she' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], woman_names[ids['id']])
#                                 caption_re_id = caption.lower().replace('she', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                             elif is_whole_word_within_caption('man', caption.lower()) or is_whole_word_within_caption("man's", caption.lower()) or is_whole_word_within_caption("a man", caption.lower()): # avoid treating the sign "Burnman Associates" as man
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
#                                 if 'a man' in caption.lower(): # you can't tell for sure if 2 man in caption relate to same man : like two man are driving hence replace man per re_id num
#                                     caption_re_id = caption.lower().replace('a man', ids_dummy_names, 1) #caption.lower().replace('a man', ids_dummy_names, 1).replace('man', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a man', man_names[ids['id']], 1)# TODO the obj_LLM_OUTPUT_COLLECTION_cand can chnage the a man to the man 
#                                 else:
#                                     caption_re_id = caption.replace('man', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('man', man_names[ids['id']])
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id # otherwise in case 2 Ids but only one "man" the "man" will be replaced twice !!
# # Only if the Id name is dummy i.e is not one of the celeb  and wasn't registered yet                                
#                                 if not(ids_dummy_names in celeb_id_name_dict.values()) and not(ids_dummy_names in dummy_name_2_gender):
#                                     self._dummy_names_to_gender_create(dummy_name_2_gender, ids_dummy_names, 'man', gender_count_for_dummy_names)
#                             elif 'boy' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
#                                 if 'a boy' in caption:
#                                     caption_re_id = caption.lower().replace('a boy', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a boy', man_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.replace('boy', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                             elif 'person' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
#                                 if 'a person' in caption:
#                                     caption_re_id = caption.lower().replace('a person', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a person', man_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.replace('person', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('person', man_names[ids['id']])
#                             elif is_whole_word_within_caption('he', caption.lower()):
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
#                                 caption_re_id = caption.lower().replace('he', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
#                             elif 'human' in caption.lower():
#                                 ids_dummy_names = celeb_id_name_dict.get(ids['id'], man_names[ids['id']])
#                                 if 'a human' in caption:
#                                     caption_re_id = caption.lower().replace('a human', ids_dummy_names, 1)
#                                     # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('a human', man_names[ids['id']])
#                                 else:
#                                     caption_re_id = caption.lower().replace('human', ids_dummy_names, 1)
#                                 all_ids_dummy_and_celeb_names.append([ids_dummy_names])
#                                 caption = caption_re_id
                                    # llm_out_cand_re_id = obj_LLM_OUTPUT_COLLECTION_cand.lower().replace('human', man_names[ids['id']])
                            # else: # could be found under people/plural list
                            #     print('Warning Id was found but was not associated n IDS: {} !!!! Caption: {} movie name: {}'.format(len(ids_n), caption, movie_name))
                    
    
            if not(caption_re_id) and bool(ids_n): # TODO BLIP2 sometimes give caption w/o the ID , but gives will-smith erronously in Top-gun : no reid only faces w/o id 
                caption_re_id = caption.lower() + ' and character {} is seen' .format(celeb_id_name_dict.get(ids['id'], ids['id']))
            if not bool(ids_n):# Faces no ID
                return caption
            
            return caption_re_id

    def _generate_summry(self, prompt_final):
        # concise 
        if self.gpt_type == 'HF_':
            hf_uservice = False
            model_id = "google/flan-ul2"#"google/flan-t5" #"distilbert-base-uncased"
            if hf_uservice:
                model_id = "google/flan-ul2"#"google/flan-t5" #"distilbert-base-uncased"
                data = query("The goal of life is [MASK].", model_id, api_token)
                while 'error' in data.keys():
                    print(data)
            else: #Inference API
                # inference = InferenceApi(repo_id="bert-base-uncased", token=api_token)
                InferenceApi(repo_id="gpt-j-6b-shakespeare", token=api_token)
                res = inference(inputs="The goal of life is [MASK].")

        elif self.gpt_type == 'text-davinci-003':
            if len(prompt_final) >4096-120: # MosaicML MPT-7B-Instruct 2K (https://huggingface.co/mosaicml/mpt-7b-instruct, https://huggingface.co/spaces/mosaicml/mpt-7b-instruct)
                print('Context window is too long', len(prompt_final))
            opportunities = 10
            while (opportunities):
                rc = gpt_execute(prompt_final, model='text-davinci-003', n=1, max_tokens=256) 
                if rc == []:
                    time.sleep(1)
                    opportunities -= 1
                    continue
                else:
                    break
                    
        elif self.gpt_type == 'chat_gpt_3.5' or self.gpt_type == 'gpt-4' or self.gpt_type == 'gpt-3.5-turbo-16k': # TODO if prompt is short than TH than use CHAT-GPT rather than 16K
            if len(prompt_final) > 4096-256 and self.gpt_type == 'chat_gpt_3.5':
                print('Context window is too long', len(prompt_final))
                return 'Context window is too long {}'.format(len(prompt_final))
            if len(prompt_final) > 4*4096-256 and self.gpt_type == 'gpt-3.5-turbo-16k':
                print('Context window is too long', len(prompt_final))
                return 'Context window is too long {}'.format(len(prompt_final))
            if len(prompt_final) > 32*1024-256 and self.gpt_type == 'gpt-4':
                print('Context window is too long', len(prompt_final))
                return 'Context window is too long {}'.format(len(prompt_final))
            
            opportunities = 10
            while (opportunities):
                try:
                    rc = self.chatgpt.completion(prompt_final, n=1, max_tokens=256, model=self.gpt_type, temperature=self.gpt_temperatue) # TODO set temp=0 make sure not repetative!!!
                    # rc = self.chatgpt.completion(prompt_prefix_then, n=1, max_tokens=256, model=self.gpt_type) #TODO add ChatGPT 16K
                    if rc == []:
                        time.sleep(1)
                        opportunities -= 1
                        continue
                    else:
                        break
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    opportunities -= 1

        return rc[0]

    def _insert_json_to_db(self, movie_id:str, scene_summ: str, mdf_no: list):
        combined_json = {'movie_id': movie_id, 'SM_MDF': mdf_no, 'scene_summary': scene_summ}
        if self.append_to_db:
            curr_json = nebula_db.get_doc_by_key({'movie_id': movie_id}, self.collection_name)
            if curr_json:
                if len(curr_json['SM_MDF']) == 1:  # 1st time
                    sm_mdf = list()
                    sm_mdf = [curr_json['SM_MDF']]
                    sm_mdf.append(mdf_no)

                    summy = list()
                    summy = [curr_json['scene_summary']]
                    summy.append(scene_summ)

                    # sm_mdf = list()
                    # sm_mdf.append({ '0' : curr_json['SM_MDF']})
                    # sm_mdf.append({ '1' : mdf_no})

                    # summy = list()
                    # summy.append({ '0' : curr_json['scene_summary']})  
                    # summy.append({ '1' : scene_summ})  
                    # summy.append(scene_summ)

                    combined_json = {'movie_id': movie_id, 'SM_MDF': sm_mdf, 'scene_summary': summy}
                elif len(curr_json['SM_MDF']) == 2: #cont aggregation
                    curr_json['SM_MDF'].append(mdf_no)
                    curr_json['scene_summary'].append(scene_summ)
                    combined_json = curr_json
                else:
                    print("_insert_json_to_db unknown type and self.append_to_db = ", self.append_to_db)
            else:
                print("Asked to append but no record has found hence just overwrite!!! movie_id:", movie_id)

        res = nebula_db.write_doc_by_key(combined_json, self.collection_name, overwrite=True, key_list=['movie_id'])
        print("Successfully inserted to database. Collection name: {}".format(self.collection_name))

        return

    def _cluster_based_place_inference(self, kmeans_n_cluster: int =None, top_k_by_cluster: int=5):
        
        df = pd.read_csv(os.path.join("/notebooks/multi_modal", "ontology_blip2_itc_per_mdf_top_gun.csv"), index_col=False)       
        # eval(df['frame3907.jpg'].dropna().values[0])
        ontology_list_len = [len(eval(df[x].dropna().values[0])) for x in df.keys()][0]
        n_mdf = len(df)
        
        if kmeans_n_cluster is None:
            kmeans_n_cluster = 1+int(n_mdf/30)

        mdf_places_retrival_score = [eval(df[x].dropna().values[0]) for x in df.keys()]
        mdf_no = [x for x in df.keys()]
        vlm_score_embed_per_mdf = np.array([y[1] for x in mdf_places_retrival_score for y in x]).reshape((n_mdf , -1))  #[x for l in lst for x in l]
        ontology_by_csv = np.array([y[0] for x in mdf_places_retrival_score for y in x]).reshape((n_mdf , -1))[0, :]
        

        # Sanity
        if 0:
            mdf_k = 'frame0014.jpg' # GT is 
            score_14 = eval(df[mdf_k].dropna().values[0])
            blip2_itc_mdf = np.array([x[1] for x in score_14]).reshape((ontology_list_len , -1))
            blip2_itc_text = np.array([x[0] for x in score_14]).reshape((ontology_list_len , -1))
            top_k_ind_per_mdf = np.argsort(blip2_itc_mdf.reshape(-1))[::-1][:top_k_by_cluster]
            # ontology_by_csv[top_k_ind_per_mdf]
            # all(ontology_by_csv[top_k_ind_per_mdf] == ['lecture room', 'conference room', 'television room', 'auditorium', 'classroom'])
            assert(all(ontology_by_csv[top_k_ind_per_mdf] == ['lecture room', 'conference room', 'television room', 'auditorium', 'classroom']))
            print([x for x in eval(df[mdf_k].dropna().values[0]) if x[0]=="lecture room"])
        # import sklearn 
        # print('The scikit-learn version is {}.'.format(sklearn.__version__)) 
        kmeans = KMeans(n_clusters=kmeans_n_cluster, random_state=0, n_init="auto").fit(vlm_score_embed_per_mdf)
        sum_square_within_dist = -kmeans.score(vlm_score_embed_per_mdf)
        assert(kmeans.cluster_centers_.shape[1]==ontology_list_len)
    # Per cluster members in terms of MDf No.
        classify_mdf = kmeans.predict(vlm_score_embed_per_mdf)
        cluster_mdfs = [list(compress(mdf_no, (classify_mdf == x))) for x in np.unique(classify_mdf)]

        all_centroids_places = list()
        for clust in np.arange(kmeans_n_cluster):
            top_k_ind_per_cluster = np.argsort(kmeans.cluster_centers_[clust, :])[::-1][:top_k_by_cluster]
            if self.verbose:
                print(kmeans.cluster_centers_[clust, :][top_k_ind_per_cluster])
                print(ontology_by_csv[top_k_ind_per_cluster])

            all_centroids_places.append(ontology_by_csv[top_k_ind_per_cluster])

        return all_centroids_places, sum_square_within_dist, cluster_mdfs


    def _semantic_similar_places_max_set_cover(self, tokens: list, topk: int=10, greedy: bool=True) -> str:

        uniq_places, cnt = np.unique(tokens, return_counts=True)
        frequent_uniq_places = uniq_places[np.argsort(cnt)[::-1]] # sort according to frequency
        # SentenceBERT score
        similarity_places = self.evaluator.compute_triplet_scores(src=[tuple([x]) for x in frequent_uniq_places], dst = [tuple([x]) for x in frequent_uniq_places])
        dist_places = 1- similarity_places
        max_set_entity = list()
        max_set_id = list()
        if greedy:
            # np.fill_diagonal(similarity_places, 0)

            max_set_entity.append(frequent_uniq_places[0])
            max_set_id.append(0)
            while len(max_set_entity) <topk:
                # np.take(dist_places, max_set_id, axis=1)
                greedy_id = np.argmax(np.take(dist_places, max_set_id, axis=1).sum(axis=1))
                max_set_id.append(greedy_id)
                max_set_entity.append(frequent_uniq_places[greedy_id])
                
        else:
            raise
        return max_set_entity

#topk == -1 then no additional top k 

    def _merge_semantic_similar_tokens(self, tokens: list, topk:int=10, sim_th: int=0.7, verbose:bool=False) -> str:

        uniq_places, cnt = np.unique(tokens, return_counts=True)
        frequent_uniq_places = uniq_places[np.argsort(cnt)[::-1]] # sort according to frequency
        # SentenceBERT score
        similarity_places = self.evaluator.compute_triplet_scores(src=[tuple([x]) for x in frequent_uniq_places], dst = [tuple([x]) for x in frequent_uniq_places])
        
        
        np.fill_diagonal(similarity_places, 0)
        simillar_places = list()
        for ix, ele in enumerate(similarity_places):
            if any(similarity_places[ix:, ix]>sim_th): # lower diagonal simillar ones will be removed 
                # print(frequent_uniq_places, similarity_places[:ix, ix])
                removed_places = frequent_uniq_places[np.where(similarity_places[ix:, ix]>sim_th)[0]+ix]
                # print(removed_places, ix)
                if verbose:
                    print("{} Like {} ".format(removed_places, frequent_uniq_places[ix]))
                simillar_places.extend(removed_places)
        
        simillar_places = [x.strip() for x in simillar_places]
        simillar_places = np.unique(simillar_places)
        g = list(frequent_uniq_places)
        [g.remove(x) for x in simillar_places]
        
        if topk != -1:
            top_k_uniq_not_sim = g[:topk]
        else: 
            g = top_k_uniq_not_sim
        
        return top_k_uniq_not_sim # todo elevator/door exhibit as door

class LLMBase(ABC):
    @abstractmethod
    def completion(prompt_template: str, *args, n=1, **kwargs):
        pass
# pip install openai==0.27.0 --target /notebooks/pip_install/

# Movies/7417592353856606351
#subsequent captions of key-frames 1 ### 2 

def get_few_shot_prompt_paragraph_based_to_tuple_4K(query_paragraph: str, scene_top_k_frequent: str, n_uniq_ids: int, 
                                                    in_context_examples: str, **kwargs):
    few_shot_seperator = kwargs.pop('few_shot_seperator', None)
    prolog_refine = kwargs.pop('prolog_refine', '')
    uniq_id_prior_put_in_caption_end = kwargs.pop('uniq_id_prior_put_in_caption_end', None)
    all_actor_names = kwargs.pop('all_actor_names', None)
    
    if isinstance(scene_top_k_frequent, (np.ndarray, np.generic)):

        if len(scene_top_k_frequent) == 2:
            scene = 'mainly at {} and secondary at {}'.format(list(scene_top_k_frequent)[0], list(scene_top_k_frequent)[1])
        else:
            scene = ' and or '.join(list(scene_top_k_frequent))
    else:
        if len(scene_top_k_frequent) == 2:
            scene = 'mainly at {} and secondary at {}'.format(scene_top_k_frequent[0], scene_top_k_frequent[1])
        else:
            scene = ' and or '.join(scene_top_k_frequent)

    # prolog = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place. Example: Video captions '''
    
# Alternatives TODO HK@@ : Provide a summary for the following article  ; move the "that were taken place at {}" to the end of prompt ask for action
    if uniq_id_prior_put_in_caption_end:
        prolog = '''Summarize the video {}given the video captions that were taken place {}. Example of video captions and summary: '''
        prolog = prolog.format(prolog_refine, scene)
    else:
        if n_uniq_ids > 0:
            prolog = '''Summarize the video {}given the captions that were taken place {} with {} persons. '''
            if all_actor_names:
                prolog += ''' specifically {}'''.format(' and'.join(all_actor_names))
                prolog += ''' Tell what they are doing. Example of video captions and summary: '''
            prolog = prolog.format(prolog_refine, scene, n_uniq_ids)
        else:
            prolog = '''Summarize the video {}given the captions that were taken place {}. Tell what they are doing. Example of video captions and summary: '''
            prolog = prolog.format(prolog_refine, scene)

       
    if uniq_id_prior_put_in_caption_end:
        epilog = '''Video captions: {}{}. Video Summary: '''
        suffix_prior = '''The captions are noisy and sometimes include people who are not there. We know for sure that there are at least {} main characters in the scene.'''.format(n_uniq_ids)
        if any(all_actor_names):
            suffix_prior = suffix_prior + ''' Specifically {}. '''.format(' and '.join(all_actor_names))
        suffix_prior = suffix_prior + '''Tell what they are doing, thier names and the theme'''

        epilog = epilog.format(query_paragraph, suffix_prior)
    else:
        epilog = '''Video captions: {}. Video Summary: '''
        epilog = epilog.format(query_paragraph, suffix_prior, n_uniq_ids)

    if few_shot_seperator:
        in_context_examples = in_context_examples  + '\n{}\n' .format(few_shot_seperator)   #
    prompt = '{}{}{}'.format(prolog, in_context_examples, epilog).strip()

    # print(prompt)
    return prompt
"""
        prompt = '''Summarize the video given the captions that were taken place at {} with {} persons. Start by telling how many persons and what place : 
            Example:'' {} '' {} Video summary :'''.format(scene, n_uniq_ids, shot_example, seq_caption_w_caption)

"""

# pip install openai
# pip install --upgrade openai
# pip show openai
class ChatGptLLM(LLMBase):
    def completion(self, prompt_template: str, *args, n=1, model=CHAT_GPT_MODEL, **kwargs):
        prompt = prompt_template.format(*args)
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]  
        max_tokens = kwargs.pop('max_tokens', 256)
        response = openai.ChatCompletion.create(messages=messages, max_tokens=max_tokens, n=n, model=model, **kwargs)
        return [x['message']['content'].strip() for x in response['choices']]


class MovieImageId(NamedTuple):
    movie_id: str
    frame_num: int

def main():

    summarize_scene = SummarizeScene()

    result_path = "/notebooks/nebula3_playground"
    unique_run_name = str(int(time.time()))
    # http://209.51.170.37:8087/docs
    add_action = True

    results = list()
    all_movie_id = list()

    if add_action:
        all_movie_id.append('Movies/7023181708619934815')
    all_movie_id.append('Movies/-6372550222147686303')
    all_movie_id.append('Movies/-3323239468660533929') #actionclipautoautotrain00616.mp4
    all_movie_id.append('Movies/-6372550222147686303')
    all_movie_id.append('Movies/889658032723458366')
    all_movie_id.append('Movies/-6576299517238034659')
    all_movie_id.append('Movies/-5723319113316714990')
    all_movie_id.append('Movies/2219594956981209558')
    all_movie_id.append('Movies/6293447408186786707')



    csv_file_name = 'scene_summarization_' + str(unique_run_name) + '_' + str(summarize_scene.prompting_type) + '_' + str(summarize_scene.gpt_type) +'.csv'

    for movie_id in all_movie_id:

        frame_boundary = []
        # if movie_id == 'Movies/-6372550222147686303':
        #     frame_boundary = [[834, 1181]]
        if movie_id == 'Movies/-6372550222147686303':  # dummy for debug
             frame_boundary = [[834, 1181], [14,272]]
        if movie_id == 'Movies/-5723319113316714990':
            frame_boundary = [[197, 320]]
        if movie_id == 'Movies/6293447408186786707':
            frame_boundary = [[1035, 1290]]

        scn_summ = summarize_scene.summarize_scene_forward(movie_id, frame_boundary, caption_type='dense_caption')
        # scn_summ = summarize_scene.summarize_scene_forward(movie_id) # for all clip w/o frame boundaries 

        print("Movie: {} Scene summary : {}".format(movie_id, scn_summ))

        # results.append({'movie_id':movie_id, 'summary': scn_summ, 'movie_name':movie_name, 'prompt': prompt_prefix_then, 'mdf_no': mdf_no})
        results.append({'movie_id':movie_id, 'summary': scn_summ, 
                        'movie_name':summarize_scene.movie_name, 'prompt_prefix_caption' : summarize_scene.prompt_prefix_caption})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, csv_file_name), index=False)


if __name__ == '__main__':
    main()

"""
TODO : add option for more verbality conditioned on more unique(tokens) from all the MDFs in the scene

FS_GPT_MODEL = 'text-davinci-003'
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
'gpt-4-32k'
'gpt-4-32k-0314'
'gpt-4'

input_type = self.get_input_type_from_db(pipeline_id, "pipelines")
from database.arangodb import DBBase
def get_input_type_from_db(pipeline_id, collection):
    nre = DBBase()
    pipeline_data = nre.get_doc_by_key({'_key': pipeline_id}, collection)
    if pipeline_data:
        if "dataset" in pipeline_data["inputs"]["videoprocessing"]:
            input_type = pipeline_data["inputs"]["videoprocessing"]["dataset"]["type"]
        else:
            input_type = pipeline_data["inputs"]["videoprocessing"]["movies"][0]["type"]
    return input_type

"""