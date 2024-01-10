"""临时开的转发接口类"""
import base64
import collections
from dataclasses import dataclass
import enum
import os
import pickle
import string
import sys

from async_timeout import timeout
from click import prompt
from basicutils.database import Sniffer, TriggerRule

from basicutils.rpc.openai import OpenAI

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import basicutils.CONST as GLOBAL


from bs4 import BeautifulSoup
import re
import asyncio
import requests
import json
import random
import urllib
import traceback
import hashlib
import urllib
from basicutils.chain import *
from basicutils.network import *
from basicutils.task import *
from mongoengine import *
from fapi.models.Player import RefPlayerBase
from basicutils import CONST

def get_command(cmdlist: list, pos: int):
    if len(cmdlist) > pos: return cmdlist[pos]
    return None

class DiffusionGlossary(RefPlayerBase, Document):
    wordmap = DictField()

def 约稿(ent: CoreEntity):
    """#约稿 [#waifu, #召唤, #产粮, #SD]
    ai画图，txt2img，容易被封所以还是建议优先直接用网页
    """
    ses = requests.session()
    authdir = 'waifusd_auth.pickle'
    apibase = 'http://127.0.0.1:7012'
    reply = []


    if os.path.exists(authdir):
        with open(authdir, 'rb') as f:
            usr, pw = pickle.load(f)
        ses.post(apibase+'/login', data={'username':usr,'password':pw})
    
    filter_p = re.compile('<.*?>', re.MULTILINE)
    def get_resolution(src):
        mc = re.compile('([0-9]+)[x\*]([0-9]+)').search(src)
        if mc:
            w, h = mc.groups()
            return int(w), int(h)
        else:
            return 512, 512

    def filter(src):
        b = []
        for i in filter_p.sub(' ', src).replace('\n', ' ').strip().replace('{', '(').replace('}', ')'):
            if b[-1:] == [' '] and ' ' == i:
                continue
            else:
                b.append(i)
        return ''.join(b)

    def parser(src):
        pm = [
            'Negative prompt: ', 'Prompt: ', 'Steps: ', 'Sampler: ', 'CFG scale: ',
            'Seed:', 'Size:', 'Model hash: ', 'denoising strength:', 'Clip skip: ',
            'variation seed:', 'variation seed strength:', 'seed resize from:', 'ENSD: ',
            'mask blur:', 'hypernet:'
        ]
        b = [[] for i in pm]

        cur = b[1]
        p = 0

        def chk():
            nonlocal p
            nonlocal cur
            for ind, token in enumerate(pm):
                if src[p:p+len(token)] == token:
                    p += len(token)
                    cur = b[ind]
                    return False
            return True

        while p < len(src):
            if chk():
                cur.append(src[p])
                p += 1
        for p, i in enumerate(b):
            b[p] = ''.join(i).strip().removesuffix(',')
        return {j.lower().strip():i for i, j in zip(b, pm) if i}

    rawinputs = ent.chain.onlyplain()
    if rawinputs == '样例':
        with open('Assets/waifusd/cn_cheatsheet_list.pkl', 'rb') as f:
            li = pickle.load(f)
        return random.choice(li)
    if rawinputs.startswith('词汇表'):
        args = rawinputs.split(' ')
        with open('Assets/waifusd/cn_cheatsheet_dict.pkl', 'rb') as f:
            d = pickle.load(f)
        if a1 := get_command(args, 1):
            if a1 == 'search':
                if a2 := get_command(args, 2):
                    return '\n'.join([f'{k}:{v}' for k, v in d.items() if a2 in k])
                else:
                    return '请输入待查询的关键词'
            else:
                return '未知指令' + a1
        return '\n'.join(random.sample([f'{k}:{v}' for k, v in d.items()], 10))
    if rawinputs == '模板':
        return """((masterpiece)), best quality, illustration, 1 girl, beautiful,beautiful detailed sky, catgirl,beautiful detailed water, cinematic lighting, Ice Wings, (few clothes),loli,(small breasts),light aero blue hair, Cirno(Touhou), wet clothes,underwater,hold breath,bubbles,cat ears ,dramatic angle
Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, huge breasts
Steps: 75, Sampler: DDIM, CFG scale: 11, Seed: 3323485853, Size: 512x768, Model hash: e6e8e1fc, Clip skip: 2"""
    glossary = None
    if rawinputs.startswith('词库'):
        glossary = DiffusionGlossary.chk(ent.pid)
        args = rawinputs.split(' ')
        if a1 := get_command(args, 1):
            if a1 == 'set':
                if a2 := get_command(args, 2):
                    if a3 := get_command(args, 3):
                        glossary.wordmap[a2] = ' '.join(args[3:])
                        glossary.save()
                        return 'OK'
                return '错误，设置一个术语命令格式为：#约稿 词库 set 例词 sample'
            elif a1 == 'del':
                if a2 := get_command(args, 2):
                    if a2 in glossary.wordmap:
                        glossary.wordmap.pop(a2)
                        glossary.save()
                    return 'OK'
                return '错误，删除一个术语命令格式为：#约稿 词库 del 例词'
            else:
                return '错误，词库的子命令有：set del两种，不加子命令则打印当前会话的自定义术语表'
        else:

            return '\n'.join(f"{wk}: {wv}" for wk, wv in glossary.wordmap.items())

        
    txt2img_inputs = collections.namedtuple('txt2img_inputs',
        field_names=[
            "prompt", "negative_prompt", "prompt_style", "prompt_style2", "steps",
            "sampler", "restore_faces", "tiling", "batch_count", "batch_size",
            "cfg_scale", "seed", "sub_seed", "subseed_strength", "seed_resize_from_h",
            "seed_resize_from_w", "unk_1", "height", "width", "highres_fix",
            "denoising_strength","firstpass_width","firstpass_height","script", "model_hash"
        ], defaults=[
            "loli", "nsfw", "None", "None", 80,
            "Euler a", False, False, 1, 1,
            7, -1, -1, 0, 0,
            0, False, 512, 512, False,
            0.8,0,0, "None", "None"
        ])

    def nums(src):
        b = []
        for i in src:
            if i in string.digits+'-':
                b.append(i)
        return int(''.join(b))
    def floats(src):
        b = []
        for i in src:
            if i in string.digits+'.':
                b.append(i)
        return float(''.join(b))
    
    mapping_string = { # 迫真萃取
        float: floats,
        int: nums,
        str: lambda x: x
    }
    
    parsed = parser(filter(rawinputs))
    if len(parsed) > 1: # 有东西，高级模式
        modify = {}
        for k, v in parsed.items():
            pk = k[:-1].replace(' ', '_')
            if pk in txt2img_inputs._fields:
                modify[pk] = mapping_string[type(txt2img_inputs._field_defaults[pk])](v)

        if sz := parsed.get('size:', ''):
            w, h = get_resolution(sz)
            modify['width'] = w
            modify['height'] = h

        if sz := parsed.get('seed resize from:', ''):
            w, h = get_resolution(sz)
            modify['seed_resize_from_w'] = w
            modify['seed_resize_from_h'] = h
        
        if 'denoising strength:' in parsed:
            modify['highres_fix'] = True
    else: # 简易模式
        inp = parsed['prompt:']

        if re.compile(r'[\u4e00-\u9fa5]').search(parsed.get('prompt:', '')): # 中文模糊模式
                
            import jieba
            with open('Assets/waifusd/cn_cheatsheet_dict.pkl', 'rb') as f:
                d = pickle.load(f)
            jieba.load_userdict('Assets/waifusd/cn_cheatsheet.dict')
            if glossary is None:
                glossary = DiffusionGlossary.chk(ent.pid)
            for wk, wv in glossary.wordmap.items():
                jieba.add_word(wk)
                d[wk] = [wv]
            c = list(jieba.cut(inp))
            cn_neg_tokens = ['不要', '别']
            tokens = [
                [], []
            ]
            unk = []
            hint_position = 0

            for sp in c:
                if sp in cn_neg_tokens:
                    hint_position = 1
                else:
                    if v := d.get(sp):
                        tokens[hint_position].append(random.choice(v))
                    else:
                        unk.append(sp)

            w, h = get_resolution(inp)
            
            reply.append(f'不在词库中的单词：{unk}\n')
            if ('-n' not in ent.meta and '-no_trans' not in ent.meta) and unk:
                from basicutils.rpc.translate import Baidu
                translated = Baidu.trans('zh', 'en', '\n'.join(unk)).split('\n')
                tokens[0] += translated
                logger.debug(f'tokens={tokens}')
                reply.append(f'已启用机器翻译：{translated}\n')
            modify = {
                'width': w,
                'height': h,
                'prompt': ','.join(tokens[0]),
                'negative_prompt': ','.join(tokens[1]),
            }
            if '-d' in ent.meta or '-debug' in ent.meta:
                reply.append(f"modify={modify}, unk={unk}, c={c}\n")
            
        else:
            modify = {
                'prompt': inp,
                'negative_prompt': ''
            }
        modify['negative_prompt'] += ',nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
        modify['prompt'] += ',((masterpiece)), best quality, illustration, beautiful, (an extremely delicate and beautiful), illustration, (extremely detailed cg)'

    img2img_flag = False
    def dict_intersect(d1: dict, d2: dict): return {k: v for k, v in d1.items() if k in d2}

    for i in ent.chain:
        if i.type == 'Image':
            img2img_flag = True
            import magic
            i: Image
            imginp = requests.get(i.url).content
            # modify['type'] = 0
            # imgb64 = str(base64.b64encode(imginp))
            # logger.debug(f'len imgb64={len(imgb64)}')
            modify['img'] = 'data:' + magic.from_buffer(imginp, mime=True) + ';base64,' + base64.b64encode(imginp).decode('utf-8')
            # logger.debug(f'len modifyimg={len(modifyimg)}')

            img2img_inputs = collections.namedtuple('img2img_inputs',
                field_names=[
                    "mode", # 0: img2img 1: inpaint 2: batch img2img
                    "prompt", "negative_prompt", "prompt_style", "prompt_style2", "img",
                    "crop_option", # 一个包含{image: "data:image/jpeg;base64,/", mask: "data:image/png;base64,iVBORw0KGgoAAA"}这样的对象，Crop and resize和Resize and fill用到
                    "unk_null7", "unk_null8",
                    "draw_mask", "steps", "sampler", "mask_blur", # 我猜这个12是mask blur
                    "original", "restore_faces", "tiling", "batch_count", "batch_size",
                    "cfg_scale", "denoising_strength", "seed", "variation_seed", 
                    "variation_strength", "seed_resize_from_h", "seed_resize_from_w", 
                    "dont_crop", "height", "width", "mode_name", "unk_false29",
                    "unk_32_30", "inpaint_masked", "unk_empty_32", "unk_empty_33",
                    "script",
                ], defaults=[
                    0,
                    "loli", "nsfw", "None", "None", "data:image/png;base64,iVBORw0KGgoAAAANSUhE",
                    None, None, None,
                    "Draw mask", 50, "Euler a", 4,
                    "original", False, False, 1, 1,
                    7, 0.85, -1, -1, 
                    0, 0, 0,
                    True, 768, 512, "Just resize", False,
                    32, "Inpaint masked", "", "",
                    "None",
            ])

            args = img2img_inputs(**dict_intersect(modify, img2img_inputs._field_defaults)) + (
                "<ul>↵<li><code>CFG Scale</code> should be 2 or lower.</li>↵</ul>↵", True, True, "", "",
                True, 50, True, 1, 0,
                False, 4, 1, """<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>""", 128,
                8, ["left", "right", "up", "down"], 1, 0.05, 128,
                4, "fill", ["left", "right", "up", "down"], False, False,
                None, "", """<p style="margin-bottom:0.75em">Will upscale the image to twice the dimensions; use width and height sliders to set tile size</p>""", 64, "None",
                "Seed", "", "Nothing", "", True,
                False, False, None, "", "<div class='error'>"
            )
            logger.debug(f'len args.img={args[5]}')

            req_json = {'fn_index':33,'data':args}
            break

    if not img2img_flag:
        args = txt2img_inputs(**dict_intersect(modify, txt2img_inputs._field_defaults))
        req_json =  {
            "enable_hr": False,
            "denoising_strength": args.denoising_strength,
            "firstphase_width": args.firstpass_width,
            "firstphase_height": args.firstpass_height,
            "hr_scale": 2,
            "hr_upscaler": "string",
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "prompt": args.prompt,
            # "styles": [
            #     "string"
            # ],
            "seed": args.seed,
            "subseed": args.sub_seed,
            "subseed_strength": args.subseed_strength,
            "seed_resize_from_h": args.seed_resize_from_h,
            "seed_resize_from_w": args.seed_resize_from_w,
            "sampler_name": args.sampler,
            "batch_size": args.batch_size,
            "n_iter": 1,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "width": args.width,
            "height": args.height,
            "restore_faces": args.restore_faces,
            "tiling": args.tiling,
            "negative_prompt": args.negative_prompt,
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {} if args.model_hash == "None" else {'sd_model_checkpoint': args.model_hash},
            "override_settings_restore_afterwards": True,
            # "script_args": [],
            # "sampler_index": "Euler",
            # "script_name": "string"
        }
        req = ses.post(f"{apibase}/sdapi/v1/txt2img", json=req_json)
        j = req.json()
        ret = [Image(base64=x) for x in j.pop('images')]
        logger.info(j)
        return reply + ret + [Plain(j['info'])]


    req = ses.post(f"{apibase}/api/predict", json=req_json)
    j = req.json()['data']
    bts = ses.get(f"{apibase}/file={j[0][0]['name']}").content

    # return f"{req.request.body}\nj={j}"
    return reply + [Image(base64=base64.b64encode(bts)), Plain(filter(j[2]))]
    




def 开车(ent: CoreEntity):
    """#开车 [#car]
    fufu不在线，那交给i宝了vol.2
    格式:
        #开车 <typ>
        typ可选字段:
            nice   开好车
            ero    开痛车
            kusa   开灵车
            kawaii 开校车
            tank   开战车
    
    """
    from cfg import setu_api
    typ = ent.chain.pop_first_cmd()
    allow = [
        'ero', 
        'kawaii', 
        'nice', 
        'tank',
        'kusa'
    ]
    if not typ:
        typ = 'nice'
    if typ == 'tank':
        return Image(url=setu_api + 'autotank')
    if typ not in allow:
        return f'交警提示：您的车型{typ}不能上路'

    j = requests.get(setu_api + f'random?typ={typ}').json()
    return [
        Image(url=setu_api + f'bin/{j["id"]}'),
        Plain(
f"""{j['title']}
https://www.pixiv.net/artworks/{j['id']}""")]

def 聊天(ent: CoreEntity):
    """#chat [$, ￥, !, ！]
    感谢以下这些中文LLM项目：
    ChatRWKV, https://github.com/BlinkDL/ChatRWKV
    ChatGLM, https://github.com/THUDM/ChatGLM-6B
    """
    attrs = ent.chain.tostr()
    if attrs in CONST.subscribes:
        sni: Sniffer = Sniffer.overwrite(ent.pid,'#chat')
        sni.add(
                '#chat', [
                TriggerRule(r'.*'),
            ]
        )
        return '【Chat】已激活'
    elif attrs in CONST.unsubscribes:
        Sniffer.clear(ent.pid, '#chat')
        return '【Chat】已禁用'
        

    inputstr = ent.chain.tostr().strip()
    resp = requests.post(f'http://127.0.0.1:7007/i', json={'m': inputstr}).json()
    if 'sniffer_invoke' in ent.meta:
        for _ in CONST.ASK_TRIGGER:
            if re.search(_, attrs, re.S): return ''.join(resp)
        return []
    return ''.join(resp)

def 跟ChatGPT聊天(ent: CoreEntity):
    """#chatgpt []
    chatgpt
    """
    pro = ent.chain.tostr()
    # if ('-n' not in ent.meta and '-no_trans' not in ent.meta) and re.compile(r'[\u4e00-\u9fa5]').search(pro):
        # return '\n\n'.join(OpenAI.chat(pro))
    # else:
    return '\n\n'.join(OpenAI.chat(pro, translate=False))


functionMap = {
}

shortMap = {
}

functionDescript = {
}