import os
import json
import random
from langchain import PromptTemplate

from .config import ROOT_PATH

def generate_prompt(tokenizer, type:int, prompt_length=32, total_vocab_size=50257, args=None):
    prefix = [tokenizer.bos_token_id] if args.model.startswith('facebook/opt') or args.model.startswith('llama') or args.model.startswith('tloen/alpaca') else []
    if type == 1: # random
        prompt_ids = prefix + [random.randint(0, total_vocab_size) for i in range(prompt_length)]
    elif type == 2: # sample front
        target_sentence = "Git is the largest revision control and collaboration system available for development. Git has replaced larger, more costly systems across the globe and has become the de facto standard tool for coders. But for some companies, small or large, housing code on a third-party cloud storage service might be a no-go. If that's the case, the only solution is in-house. For some, that means setting up a server and running a Git repository for the housing of proprietary or open source code. However, for some companies (especially those on the smaller side), having the resources (and time) to set up a server dedicated for Git storage may not be an option. That being the case, what do you do? Fortunately there's a solution, one that's incredibly simple. Said solution is Gitstorage, an easy-to-deploy appliance dedicated to housing your Git repositories. Each appliance is a single board computer (based on the Raspberry Pi). The device is smaller than a credit card, has no moving parts, generates no heat, is wall-mountable, is powered by a standard USB (or included mini USB), and offers a standard ethernet connection. The full specs are: Dimensions - 3.44\" × 2.93\" × 1.28\" (87.4 mm × 74.3 mm × 32.5 mm) Weight - 2.08 oz (59 g) Wall mount - 4 screws Ambient temperature - 32 °F - 104 °F (0 °C - 40 °C) Memory capacity - 16 GB (GS-16) 64 GB (GS-64) Storage for git repos - 10.6 GB (GS-16) 58.6 GB (GS-64) Certifications - CE, FCC Processor - H2 quadcore Cortex-A7 with 512 MB RAM Power supply - Standard USB Connectors - 1 × 10/100 MBit/s Ethernet, USB-A, Power (USB Micro-B) Web interface languages - English (US), French, German Price (MSRP) - $399 USD (GS-16) $499 USD (GS-64) But how well does the Gitstorage appliance work? Is it really that easy to deploy? Let\'s deploy one and find out. SEE: How to build a successful developer career (free PDF) (TechRepublic) Setup The setup of the Gitstorage is remarkably simple: Unpack the box. Plug the device into your network (you\'ll need a Cat5 cable). Connect the power cable. Wait 60 seconds. At this point, things get a bit complicated. According to the directions, you should then be able to point a browser to http://gitst.net and the Gitstorage interface will appear. I tried that on both a Linux desktop and a MacBook Pro. Neither machine could find the device. In fact, if I attempted to ping the gitst.net address, I received a WAN IP address that didn\'t respond. The only way I was able to reach my Gitstorage device was to log into my router, look for gitstorage among the connected devices, and find out the IP address of the device. Once I had that IP address, I could point my browser to that address and login with user root and password password. At that point, the setup wizard is presented (Figure A). Figure A The steps to the setup wizard are: Language selection EULA Name the device Device root CA creation or import (optional) Encryption password Admin setup (email/password) Dropbox setup (optional) Email setup (optional) Once I completed the wizard, trouble in paradise arose. During the first round, the final screen was blank. After a reboot, I had to walk through the wizard again. This time around the final screen appeared, the All set link didn\'t work. So I returned to the IP address and was presented with a login screen. I attempted to use the admin email/password I\'d setup during the wizard, but that wouldn\'t work. I then attempted root/password ... again to no avail. After another reboot (unplug, wait a few seconds, plug back in), I was (once again) sent to the setup wizard (only this time, half-way through). Once again, the final screen links wouldn\'t work. Fortunately, I was sent two devices, so I unplugged the first (a GS-16) and plugged in the second (a GS-64). This time around, everything went smoothly and I was able to log into the Gitstorage interface (Figure B). Figure B Usage From the main interface, your first task is to create users. Click on the Users button and add the necessary information for a new user (Figure C). Figure C You can now create a new repository. However, new repositories can only be created by the Root user. This is a problem. Why? Remember that admin user created during setup? I was unable to login with that user. So the only user with root privileges is root and the password is, well, not even remotely secure. Changing that password isn\'t nearly as intuitive as you might think (at least not from an admin perspective). Instead of the root user password change option being in the Settings sections, you must click on the Root user button in the upper right corner. From the popup menu (Figure D), click Account. Figure D In the resulting window, click Password. When prompted, type (and verify) the new password for the root user. Log out and log back in with your new credentials. Now click on the Repositories entry in the left navigation, click the Create button, give the repository a name, and click Submit. Once you\'ve created the repository, click on the Settings entry for it and then click the Add user button, so you can add users to the repository (otherwise the root user will be the only one with access). SEE: 10 Terminal commands to speed your work on the Mac (free PDF) (TechRepublic) Smooth sailing And that\'s pretty much all there is to setting up a Gitstorage device. Although I did have one hiccup with the first appliance, setting up the second resulted in some pretty smooth sailing for using an in-house Git repository. If you\'re looking for an incredibly simple solution for code collaboration (and you don\'t have the resources to setup your own Git server), I highly recommend a Gitstorage device. It\'s a simple, small, and elegant solution that should serve you well. Automatically sign up for TechRepublic\'s Cloud Insights Newsletter for more hot tips and tricks. Subscribe Also see"
        prompt_ids = tokenizer.encode(target_sentence)[:prompt_length]
    elif type == 3:
        target_sentence = "Git is the largest revision control and collaboration system available for development. Git has replaced larger, more costly systems across the globe and has become the de facto standard tool for coders. But for some companies, small or large, housing code on a third-party cloud storage service might be a no-go. If that's the case, the only solution is in-house. For some, that means setting up a server and running a Git repository for the housing of proprietary or open source code. However, for some companies (especially those on the smaller side), having the resources (and time) to set up a server dedicated for Git storage may not be an option. That being the case, what do you do? Fortunately there's a solution, one that's incredibly simple. Said solution is Gitstorage, an easy-to-deploy appliance dedicated to housing your Git repositories. Each appliance is a single board computer (based on the Raspberry Pi). The device is smaller than a credit card, has no moving parts, generates no heat, is wall-mountable, is powered by a standard USB (or included mini USB), and offers a standard ethernet connection. The full specs are: Dimensions - 3.44\" × 2.93\" × 1.28\" (87.4 mm × 74.3 mm × 32.5 mm) Weight - 2.08 oz (59 g) Wall mount - 4 screws Ambient temperature - 32 °F - 104 °F (0 °C - 40 °C) Memory capacity - 16 GB (GS-16) 64 GB (GS-64) Storage for git repos - 10.6 GB (GS-16) 58.6 GB (GS-64) Certifications - CE, FCC Processor - H2 quadcore Cortex-A7 with 512 MB RAM Power supply - Standard USB Connectors - 1 × 10/100 MBit/s Ethernet, USB-A, Power (USB Micro-B) Web interface languages - English (US), French, German Price (MSRP) - $399 USD (GS-16) $499 USD (GS-64) But how well does the Gitstorage appliance work? Is it really that easy to deploy? Let\'s deploy one and find out. SEE: How to build a successful developer career (free PDF) (TechRepublic) Setup The setup of the Gitstorage is remarkably simple: Unpack the box. Plug the device into your network (you\'ll need a Cat5 cable). Connect the power cable. Wait 60 seconds. At this point, things get a bit complicated. According to the directions, you should then be able to point a browser to http://gitst.net and the Gitstorage interface will appear. I tried that on both a Linux desktop and a MacBook Pro. Neither machine could find the device. In fact, if I attempted to ping the gitst.net address, I received a WAN IP address that didn\'t respond. The only way I was able to reach my Gitstorage device was to log into my router, look for gitstorage among the connected devices, and find out the IP address of the device. Once I had that IP address, I could point my browser to that address and login with user root and password password. At that point, the setup wizard is presented (Figure A). Figure A The steps to the setup wizard are: Language selection EULA Name the device Device root CA creation or import (optional) Encryption password Admin setup (email/password) Dropbox setup (optional) Email setup (optional) Once I completed the wizard, trouble in paradise arose. During the first round, the final screen was blank. After a reboot, I had to walk through the wizard again. This time around the final screen appeared, the All set link didn\'t work. So I returned to the IP address and was presented with a login screen. I attempted to use the admin email/password I\'d setup during the wizard, but that wouldn\'t work. I then attempted root/password ... again to no avail. After another reboot (unplug, wait a few seconds, plug back in), I was (once again) sent to the setup wizard (only this time, half-way through). Once again, the final screen links wouldn\'t work. Fortunately, I was sent two devices, so I unplugged the first (a GS-16) and plugged in the second (a GS-64). This time around, everything went smoothly and I was able to log into the Gitstorage interface (Figure B). Figure B Usage From the main interface, your first task is to create users. Click on the Users button and add the necessary information for a new user (Figure C). Figure C You can now create a new repository. However, new repositories can only be created by the Root user. This is a problem. Why? Remember that admin user created during setup? I was unable to login with that user. So the only user with root privileges is root and the password is, well, not even remotely secure. Changing that password isn\'t nearly as intuitive as you might think (at least not from an admin perspective). Instead of the root user password change option being in the Settings sections, you must click on the Root user button in the upper right corner. From the popup menu (Figure D), click Account. Figure D In the resulting window, click Password. When prompted, type (and verify) the new password for the root user. Log out and log back in with your new credentials. Now click on the Repositories entry in the left navigation, click the Create button, give the repository a name, and click Submit. Once you\'ve created the repository, click on the Settings entry for it and then click the Add user button, so you can add users to the repository (otherwise the root user will be the only one with access). SEE: 10 Terminal commands to speed your work on the Mac (free PDF) (TechRepublic) Smooth sailing And that\'s pretty much all there is to setting up a Gitstorage device. Although I did have one hiccup with the first appliance, setting up the second resulted in some pretty smooth sailing for using an in-house Git repository. If you\'re looking for an incredibly simple solution for code collaboration (and you don\'t have the resources to setup your own Git server), I highly recommend a Gitstorage device. It\'s a simple, small, and elegant solution that should serve you well. Automatically sign up for TechRepublic\'s Cloud Insights Newsletter for more hot tips and tricks. Subscribe Also see"
        target_seq = tokenizer.encode(target_sentence)
        start = len(target_seq) // 2
        prompt_ids = prefix + target_seq[start: start + prompt_length]
    elif type == 4:
        target_sentence = "Need somebody with expertise on automobiles regarding troubleshooting solutions like; diagnosing problems/errors present both visually & within engine parts in order to figure out what's causing them (like lack of oil or power issues) & suggest required replacements while recording down details such fuel consumption type etc., First inquiry – “Car won't start although battery is full charged”"
        prompt_ids = tokenizer.encode(target_sentence)
        # .remove(tokenizer.eos_token_id)
    else:
        target_sentence = "I want to act as a Statistician. I will provide you with details related with statistics. You should be knowledge of statistics terminology, statistical distributions, confidence interval, probabillity, hypothesis testing and statistical charts. My first request is 'I need help calculating how many million banknotes are in active use in the world'"
        prompt_ids = tokenizer.encode(target_sentence)
    return prompt_ids

def prepare_prompts(tokenizer, args):
    prompts = []
    prompts.append(generate_prompt(tokenizer, 1, 16, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 1, 32, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 1, 64, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 2, 16, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 2, 32, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 2, 64, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 3, 16, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 3, 32, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 3, 64, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 4, 16, args.max_length, args))
    prompts.append(generate_prompt(tokenizer, 5, 16, args.max_length, args))
    return prompts


dolly_prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

dolly_prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

from . import templates
import copy
import torch

convert_name = {
    'alpaca':'alpaca',
    "tloen/alpaca-lora-7b":"alpaca",
    'oasst':'oasst',
    "facebook/opt-125m":"naive",
    "facebook/opt-1.3b":"naive",
    'llama/7B':'naive',
    'llama/30B':'naive',
    'gpt2-large':'naive',
    "dolly/7b":"dolly",
    "stablelm":"stablelm",
    "vicuna":"vicuna_v1.1",
    "pythia":"naive",
    "mosaic-instruct":"dolly",
    "mosaic":"dolly",
    "koala":"koala_v1",
    "nous":"alpaca",
    "wizardlm":"wizard",
    "stablevicuna":"stablevicuna",
    "guanaco":"guanaco",
    "chatglm":"naive",
}

class TemplateFactory():
    '''
        1. Use a sentence to get template and then encode
        2. Extract the template part of the encoded
        3. 
    '''
    def __init__(self, model_name, trigger_token_length, tokenizer, embedding) -> None:
        self.model_name = model_name
        self.trigger_token_length = trigger_token_length
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.add_additional_prompt("")
    
    def add_additional_prompt(self, prefix_sentence):
        conv : templates.Conversation = templates.conv_templates[convert_name[self.model_name]].copy()
        
        if prefix_sentence != "" or 'alpaca' in self.model_name: 
            prefix_sentence += ' '
        demo_sentence = self.tokenizer.decode([7993] * self.trigger_token_length)
        conv.append_message(conv.roles[0], prefix_sentence + demo_sentence)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.set_template(prompt)
    
    def add_infix_prompt(self, infix_sentence):
        conv : templates.Conversation = templates.conv_templates[convert_name[self.model_name]].copy()
        
        if infix_sentence != "" or 'alpaca' in self.model_name: 
            infix_sentence = ' ' + infix_sentence
        demo_sentence = self.tokenizer.decode([7993] * self.trigger_token_length)
        conv.append_message(conv.roles[0],  demo_sentence + infix_sentence)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.set_template(prompt)

    def set_template(self, prompt):
        tokenizer = self.tokenizer
        trigger_token_length = self.trigger_token_length
        embedding = self.embedding

        input_ids = tokenizer.encode(prompt)
        print(prompt)
        print(input_ids)
        total_length = len(input_ids)

        prefix_len = max(index for index, item in enumerate(input_ids) if item == 7993) - trigger_token_length + 1
        self.prefix_tokens = input_ids[:prefix_len]
        self.tail_tokens = input_ids[prefix_len+trigger_token_length:]
        
        self.prefix_embeds = embedding[input_ids[:prefix_len]].detach().unsqueeze(0)
        self.tail_embeds = embedding[input_ids[prefix_len+trigger_token_length:]].detach().unsqueeze(0)
        
        self.template_length = total_length - trigger_token_length
        self.response_offset = prefix_len+trigger_token_length
        self.prefix_length = prefix_len
        self.template_w_trigger_length = total_length

    def get_input_embeddings(self, inputs_embeds):
        front_part = inputs_embeds[:, :self.trigger_token_length]
        tail_part = inputs_embeds[:, self.trigger_token_length:]
        concated = torch.concat(
            [self.prefix_embeds, front_part, self.tail_embeds, tail_part], dim=1)
        return concated
    
    def get_input_tokens(self, inputs_tokens):
        return self.prefix_tokens + inputs_tokens + self.tail_tokens
    
def get_normal_init(tokenizer):
    alpaca_data = json.load(open("/PATH/TO/ICA/alpaca-lora/alpaca_data.json", "r"))
    trigger_text_init = alpaca_data[6]['instruction'] + alpaca_data[6]['output']
    prefix_len = len(tokenizer.encode(""))
    trigger_token_init = tokenizer.encode(trigger_text_init)[prefix_len:]
    return trigger_token_init


def load_data(tokenizer, sample_num, args, alpaca_only=False, shareGPT_only=False):
    alpaca_data_path = os.path.join(NORMAL_DATA_DIR, 'alpaca_data.json')
    shareGPT_data_path = os.path.join(NORMAL_DATA_DIR, 'ShareGPT_unfiltered_cleaned_split.json')
    alpaca_data = json.load(open(alpaca_data_path, "r"))
    shareGPT_data = json.load(open(shareGPT_data_path, "r"))
    if alpaca_only:
        return_text = []
        for text in alpaca_data:
            if text['input'] == '':
                if len(tokenizer.encode(text['instruction'])) < args.max_length:
                    return_text.append(text['instruction'])
            if len(return_text) == sample_num:
                return return_text
    elif shareGPT_only:
        return_text = []
        for text in shareGPT_data:
            conversation = text['conversations']
            if conversation == []:
                continue
            if conversation[0]['from'] == 'human':
                if len(tokenizer.encode(conversation[0]['value'])) < args.max_length:
                    return_text.append(conversation[0]['value'])
            if len(return_text) == sample_num:
                return return_text
    else:
        return_text = []
        for text in alpaca_data:
            if text['input'] == '':
                if len(tokenizer.encode(text['instruction'])) < args.max_length:
                    return_text.append(text['instruction'])
            if len(return_text) == sample_num/2:
                break
        for text in shareGPT_data:
            conversation = text['conversations']
            if conversation == []:
                continue
            if conversation[0]['from'] == 'human':
                if len(tokenizer.encode(conversation[0]['value'])) < args.max_length:
                    return_text.append(conversation[0]['value'])
            if len(return_text) == sample_num:
                return return_text
        return return_text