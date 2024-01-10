import json
import os.path
import time

from PIL import Image
import os
import openai

openai.api_key = ""

mp_imgid={}

list_res=os.listdir('cc_sbu_align/rewrite_caption')


'''with open('/annotations/captions_val2014.json','r',encoding='utf-8') as f:
    data=json.load(f)

    for img_data in data['images']:
        mp_imgid[img_data['id']]=img_data['file_name']
    #print(mp_imgid)
    for anno_data in data['annotations']:
        caption=anno_data['caption']
        id=anno_data['image_id']
        gen_text_name=mp_imgid[id].split('.')[0]+'.txt'
        if gen_text_name in list_res:
            continue

        in_context_example = "content:The image shows a man fishing on a lawn next to a river with a bridge in the background. Trees can be seen on the other side of the river, and the sky is cloudy.\nQ:Extract verbs, nouns,and adjectives from sentences.\nA:\n<man>\n<fishing>\n<lawn>\n<river>\n<bridge>\n<sky>\n<cloudy>\n\ncontent:This image shows a kitchen with stainless steel appliances, including a refrigerator, oven, and dishwasher. The countertops are made of black granite, and there is a white backsplash behind the stove. The floor is made of beige tiles, and the walls are painted white. There is a door that leads to the outside.\nQ:Extract verbs, nouns,and adjectives from sentences.\nA:\n<kitchen>\n<stainless steel>\n<appliances>\n<refrigerator>\n<oven>\n<dishwasher>\n<countertops>\n<black>\n<granite>\n<white>\n<backsplash>\n<stove>\n<floor>\n<beige>\n<tiles>\n<walls>\n<painted>\n<door>\n<outside>\n\ncontent:"

        que_str = caption + '\n' + 'Q:Extract verbs, nouns,and adjectives from sentences.\nA:'

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": in_context_example + que_str},
            ]
        )
        #print(id)
        with open('/annotations/aug_text/{}'.format(gen_text_name), 'w', encoding='utf-8') as ft:
            ft.write(response['choices'][0]["message"]["content"])
            ft.close()

        time.sleep(2)
'''
#############################################
'''savepath='/cc_sbu_align/rewrite_caption/{}'

dir=os.listdir('/cc_sbu_align/rewrite_caption')

with open('/cc_sbu_align/filter_cap.json','r',encoding='utf-8') as f:
    data=json.load(f)
    for i,da in enumerate(data['annotations']):
        image=da['image_id']
        if image in dir:
            continue
        caption=da['caption']
        image_path=os.path.join('/cc_sbu_align/image',image+'.jpg')

        que_str=caption+'rewrite this image caption'

        response =openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": que_str},
            ],
            n = 5,
        )
        res = ''
        with open(savepath.format(image), 'w', encoding='utf-8') as fw:
            for j in range(len(response['choices'])):
                res_text = response['choices'][j]["message"]["content"]
                res += res_text + '\n'
            fw.write(res)

        time.sleep(2)'''
#########################################################
'''root='/cc_sbu_align/aug_text'
tarroot='/cc_sbu_align/aug_caption'

savepath='/cc_sbu_align/aug_caption/{}'
prompt = "content:\n<man>\n<fishing>\n<lawn>\n<river>\n<bridge>\n<sky>\n<cloudy>\nQ:The task of generation is as follows, and the generated sentences are required to contain verbs, nouns and adjectives in the given content.\nA:The image shows a man fishing on a lawn next to a river with a bridge in the background. Trees can be seen on the other side of the river, and the sky is cloudy.\n\ncontent:\n"
question="\nQ:The task of generation is as follows, and the generated sentences are required to contain verbs, nouns and adjectives in the given content.\nA:"


datadir=os.listdir(root)
tardir=os.listdir(tarroot)

for d in datadir:
    if d in tardir:
        continue
    data=os.path.join(root,d)
    print(data)
    with open(data,'r',encoding='utf-8') as f:
            cur_data=f.read()

            res_prompt=prompt+cur_data+question
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": res_prompt},
                ],
                n=5,
            )

            res=''
            with open(savepath.format(d), 'w', encoding='utf-8') as fw:
                for j in range(len(response['choices'])):
                    res_text=response['choices'][j]["message"]["content"]
                    res+=res_text+'\n'
                fw.write(res)
            time.sleep(2)'''

'''aug_root='/cc_sbu_align/aug_caption'
aug_captions=os.listdir(aug_root)

annotations=[]
res={}

for aug_ in aug_captions:
    annotation={}
    image_id=aug_.split('.')[0]
    aug_data=os.path.join(aug_root,aug_)
    with open(aug_data,'r',encoding='utf-8') as fa:
        aug_caption=fa.read().strip().split('\n')
        for aug_c in aug_caption:
            annotation['image_id']=image_id
            annotation['caption']=aug_c
            annotations.append(annotation)

res["annotations"]=annotations

with open('./filter_cap.json','w',encoding='utf-8') as fc:
    json.dump(res,fc)'''


