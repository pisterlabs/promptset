import os, sys, json
os.system("pip install gradio==3.19.1")
import openai
import gradio as gr

from loguru import logger
import paddlehub as hub
import random
from encoder import get_encoder

openai.api_key = os.getenv("OPENAI_API_KEY")

from utils import get_tmt_client, getTextTrans_tmt
tmt_client = get_tmt_client()

def getTextTrans(text, source='zh', target='en'):
    def is_chinese(string):
        for ch in string:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
        
    if not is_chinese(text) and target == 'en': 
        return text
       
    try:
        text_translation = getTextTrans_tmt(tmt_client, text, source, target)
        return text_translation
    except Exception as e:
        return text 

start_work = """async() => {
    function isMobile() {
        try {
            document.createEvent("TouchEvent"); return true;
        } catch(e) {
            return false; 
        }
    }
	function getClientHeight()
	{
	  var clientHeight=0;
	  if(document.body.clientHeight&&document.documentElement.clientHeight) {
		var clientHeight = (document.body.clientHeight<document.documentElement.clientHeight)?document.body.clientHeight:document.documentElement.clientHeight;
	  } else {
		var clientHeight = (document.body.clientHeight>document.documentElement.clientHeight)?document.body.clientHeight:document.documentElement.clientHeight;
	  }
	  return clientHeight;
	}
 
    function setNativeValue(element, value) {
      const valueSetter = Object.getOwnPropertyDescriptor(element.__proto__, 'value').set;
      const prototype = Object.getPrototypeOf(element);
      const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value').set;
      
      if (valueSetter && valueSetter !== prototypeValueSetter) {
            prototypeValueSetter.call(element, value);
      } else {
            valueSetter.call(element, value);
      }
      element.dispatchEvent(new Event('input', { bubbles: true }));
    }
   function get_clear_innerHTML(innerHTML) {
        innerHTML = innerHTML.replace(/<p>|<\\/p>|\\n/g, '');
        regexp = /\\★☆(.*?)\\☆★/;
        match = innerHTML.match(regexp);
        if (match) {
            innerHTML = match[1];
        } 
        return innerHTML;
    }    
    function save_conversation(chatbot) {        
        var conversations = new Array();
        var conversations_clear = new Array();
        for (var i = 0; i < chatbot.children.length; i++) {
            testid_icon = '☟:'; //'user'
            if (chatbot.children[i].dataset['testid'] == 'bot') {
                testid_icon = '☝:'; //'bot'
            }            
            innerHTML = chatbot.children[i].innerHTML;
            conversations.push(testid_icon + innerHTML);
            if (innerHTML.indexOf("<img") == -1 && innerHTML.indexOf("null_") == -1) {              
                conversations_clear.push(testid_icon + get_clear_innerHTML(innerHTML));
            }
        }
        var json_str = JSON.stringify(conversations);
        setNativeValue(window['chat_his'], JSON.stringify(conversations_clear));
        localStorage.setItem('chatgpt_conversations', json_str);
    }
    function img_click(img) {
        this_width = parseInt(img.style.width) + 20;
        if (this_width > 100) {
            this_width = 20;
        }
        img.style.width = this_width + "%";
        img.style.height = img.offsetWidth + 'px'; 
    }
    function load_conversation(chatbot) {
        var json_str = localStorage.getItem('chatgpt_conversations');
        if (json_str) {
            var conversations_clear = new Array();
            conversations = JSON.parse(json_str);
            for (var i = 0; i < conversations.length; i++) {
                innerHTML = conversations[i];
                if (innerHTML.indexOf("☝:") == -1) {
                    className = "message user svelte-134zwfa";
                    bgcolor = "#16a34a"; 
                    testid = "user";
                    testid_icon = '☟:'; //'user'
                } else {
                    className = "message bot svelte-134zwfa";
                    bgcolor = "#2563eb";  
                    testid = "bot";
                    testid_icon = '☝:'; //'bot'
                }
                var new_div = document.createElement("div");
                new_div.className = className;
                new_div.style.backgroundColor = bgcolor;
                new_div.dataset.testid = testid;
                if (innerHTML.indexOf("data:image/jpeg") >= 0) { 
                    new_div.style.width = "20%"; 
                    new_div.style.padding = "0.2rem"; 
                    new_div.onclick = function(e) {
                        img_click(this);
                    }
                    setTimeout(function(){
                        new_div.style.height = new_div.offsetWidth + 'px'; 
                        new_div.children[0].setAttribute('style', 'max-width: none; width:100%');
                    }, 10);                    
                }   
                innerHTML = innerHTML.replace("☝:", "");
                innerHTML = innerHTML.replace("☟:", "");                             
                new_div.innerHTML = innerHTML;
                if (innerHTML.indexOf("null_") != -1) {
                    new_div.style.display = 'none';
                }                
                chatbot.appendChild(new_div);
                
                if (innerHTML.indexOf("<img") == -1 && innerHTML.indexOf("null_") == -1) {
                    conversations_clear.push(testid_icon + get_clear_innerHTML(innerHTML));
                }                
            }
            setNativeValue(window['chat_his'], JSON.stringify(conversations_clear));
            setTimeout(function(){
                window['chat_bot1'].children[1].scrollTop = window['chat_bot1'].children[1].scrollHeight;
            }, 500);             
        }
    }
    var gradioEl = document.querySelector('body > gradio-app').shadowRoot;
    if (!gradioEl) {
        gradioEl = document.querySelector('body > gradio-app');
    }
    
    if (typeof window['gradioEl'] === 'undefined') {
        window['gradioEl'] = gradioEl;
       
        const page1 = window['gradioEl'].querySelectorAll('#page_1')[0];
        const page2 = window['gradioEl'].querySelectorAll('#page_2')[0]; 
    
        page1.style.display = "none";
        page2.style.display = "block";
        window['div_count'] = 0;
        window['chat_radio_0'] = window['gradioEl'].querySelectorAll('#chat_radio')[0].querySelectorAll('input[name=radio-chat_radio]')[0];
        window['chat_radio_1'] = window['gradioEl'].querySelectorAll('#chat_radio')[0].querySelectorAll('input[name=radio-chat_radio]')[1];        
        window['chat_bot'] = window['gradioEl'].querySelectorAll('#chat_bot')[0];
        window['chat_bot1'] = window['gradioEl'].querySelectorAll('#chat_bot1')[0];  
        window['my_prompt'] = window['gradioEl'].querySelectorAll('#my_prompt')[0].querySelectorAll('textarea')[0];
        window['my_prompt_en'] = window['gradioEl'].querySelectorAll('#my_prompt_en')[0].querySelectorAll('textarea')[0];
        window['chat_his'] = window['gradioEl'].querySelectorAll('#chat_history')[0].querySelectorAll('textarea')[0];
        chat_row = window['gradioEl'].querySelectorAll('#chat_row')[0]; 
        prompt_row = window['gradioEl'].querySelectorAll('#prompt_row')[0]; 
        window['chat_bot1'].children[1].children[0].textContent = '';
        
        clientHeight = getClientHeight();
        if (isMobile()) {
            output_htmls = window['gradioEl'].querySelectorAll('.output-html');
            for (var i = 0; i < output_htmls.length; i++) {
               output_htmls[i].style.display = "none";
            }
            new_height = (clientHeight - 250) + 'px';
        } else {
            new_height = (clientHeight - 350) + 'px';
        }
        chat_row.style.height = new_height;
        window['chat_bot'].style.height = new_height;
        window['chat_bot'].children[1].style.height = new_height;
        window['chat_bot1'].style.height = new_height;
        window['chat_bot1'].children[1].style.height = new_height;
        window['chat_bot1'].children[0].style.top = (parseInt(window['chat_bot1'].style.height)-window['chat_bot1'].children[0].offsetHeight-2) + 'px';
        prompt_row.children[0].style.flex = 'auto';
        prompt_row.children[0].style.width = '100%';
        window['gradioEl'].querySelectorAll('#chat_radio')[0].style.flex = 'auto';
        window['gradioEl'].querySelectorAll('#chat_radio')[0].style.width = '100%';        
        prompt_row.children[0].setAttribute('style','flex-direction: inherit; flex: 1 1 auto; width: 100%;border-color: green;border-width: 1px !important;')
        window['chat_bot1'].children[1].setAttribute('style', 'border-bottom-right-radius:0;top:unset;bottom:0;padding-left:0.1rem');
        window['gradioEl'].querySelectorAll('#btns_row')[0].children[0].setAttribute('style', 'min-width: min(10px, 100%); flex-grow: 1');
        window['gradioEl'].querySelectorAll('#btns_row')[0].children[1].setAttribute('style', 'min-width: min(10px, 100%); flex-grow: 1');
        
        load_conversation(window['chat_bot1'].children[1].children[0]);
        window['chat_bot1'].children[1].scrollTop = window['chat_bot1'].children[1].scrollHeight;
        
        window['gradioEl'].querySelectorAll('#clear-btn')[0].onclick = function(e){
            if (confirm('Clear all outputs?')==true) {
                for (var i = window['chat_bot'].children[1].children[0].children.length-1; i >= 0; i--) {
                    window['chat_bot'].children[1].children[0].removeChild(window['chat_bot'].children[1].children[0].children[i]); 
                }
                for (var i = window['chat_bot1'].children[1].children[0].children.length-1; i >= 0; i--) {
                    window['chat_bot1'].children[1].children[0].removeChild(window['chat_bot1'].children[1].children[0].children[i]); 
                }                
                window['div_count'] = 0;
                save_conversation(window['chat_bot1'].children[1].children[0]);
            }
        }
 
        function set_buttons(action) {
            window['submit-btn'].disabled = action;
            window['clear-btn'].disabled = action;
            window['chat_radio_0'].disabled = action;
            window['chat_radio_1'].disabled = action; 
            btn_color = 'color:#000';
            if (action) {
                btn_color = 'color:#ccc';
            }
            window['submit-btn'].setAttribute('style', btn_color);
            window['clear-btn'].setAttribute('style', btn_color);
            window['chat_radio_0'].setAttribute('style', btn_color);
            window['chat_radio_1'].setAttribute('style', btn_color);             
        }  
        window['prevPrompt'] = '';
        window['doCheckPrompt'] = 0;
        window['prevImgSrc'] = '';
        window['checkChange'] = function checkChange() {
            try {
                if (window['chat_radio_0'].checked) {
                    dot_flashing = window['chat_bot'].children[1].children[0].querySelectorAll('.dot-flashing');
                    if (window['chat_bot'].children[1].children[0].children.length > window['div_count'] && dot_flashing.length == 0) {
                        new_len = window['chat_bot'].children[1].children[0].children.length - window['div_count'];
                        for (var i = 0; i < new_len; i++) { 
                            new_div = window['chat_bot'].children[1].children[0].children[window['div_count'] + i].cloneNode(true);
                            window['chat_bot1'].children[1].children[0].appendChild(new_div);
                        }
                        window['div_count'] = window['chat_bot'].children[1].children[0].children.length;
                        window['chat_bot1'].children[1].scrollTop = window['chat_bot1'].children[1].scrollHeight;
                        save_conversation(window['chat_bot1'].children[1].children[0]);
                    }
                    if (window['chat_bot'].children[0].children.length > 1) {
                        set_buttons(true);
                        window['chat_bot1'].children[0].textContent = window['chat_bot'].children[0].children[1].textContent;
                    } else {
                        set_buttons(false);
                        window['chat_bot1'].children[0].textContent = '';
                    }
                } else {
                    img_index = 0;
                    draw_prompt_en = window['my_prompt_en'].value;
                    if (window['doCheckPrompt'] == 0 && window['prevPrompt'] != draw_prompt_en) {
                            console.log('_____draw_prompt_en___[' + draw_prompt_en + ']_');
                            window['doCheckPrompt'] = 1;
                            window['prevPrompt'] = draw_prompt_en;
 
                            tabitems = window['gradioEl'].querySelectorAll('.tabitem');
                            for (var i = 0; i < tabitems.length; i++) {   
                                inputText = tabitems[i].children[0].children[1].children[0].querySelectorAll('input')[0];
                                setNativeValue(inputText, draw_prompt_en);
                            }                            
                            setTimeout(function() {
                                window['draw_prompt'] = window['my_prompt'].value; 
                                btns = window['gradioEl'].querySelectorAll('button');
                                for (var i = 0; i < btns.length; i++) {
                                    if (['Generate image','Run'].includes(btns[i].innerText)) {
                                        btns[i].click();                
                                    }
                                }
                                window['doCheckPrompt'] = 0;
                            }, 10);                   
                    }
                    tabitems = window['gradioEl'].querySelectorAll('.tabitem');
                    imgs = tabitems[img_index].children[0].children[1].children[1].querySelectorAll("img");
                    if (imgs.length > 0) {
                        if (window['prevImgSrc'] !== imgs[0].src) {
                            var user_div = document.createElement("div");
                            user_div.className = "message user svelte-134zwfa";
                            user_div.style.backgroundColor = "#16a34a"; 
                            user_div.dataset.testid = 'user';
                            user_div.innerHTML = "<p>作画: " + window['draw_prompt'] + "</p><img></img>";
                            window['chat_bot1'].children[1].children[0].appendChild(user_div);
                            var bot_div = document.createElement("div");
                            bot_div.className = "message bot svelte-134zwfa";
                            bot_div.style.backgroundColor = "#2563eb"; 
                            bot_div.style.width = "20%"; 
                            bot_div.dataset.testid = 'bot';
                            bot_div.onclick = function(e){
                                img_click(this);
                            }
                            setTimeout(function(){
                                bot_div.style.height = bot_div.offsetWidth + 'px'; 
                                bot_div.children[0].setAttribute('style', 'max-width:none; width:100%');
                            }, 10);                             
                            bot_div.style.padding = "0.2rem"; 
                            bot_div.appendChild(imgs[0].cloneNode(true));
                            window['chat_bot1'].children[1].children[0].appendChild(bot_div);
                            
                            window['chat_bot1'].children[1].scrollTop = window['chat_bot1'].children[1].scrollHeight;
                            window['prevImgSrc'] = imgs[0].src;
                            save_conversation(window['chat_bot1'].children[1].children[0]);
                        }
                    }
                    if (tabitems[img_index].children[0].children[1].children[1].children[0].children.length > 1) {
                        tips = tabitems[img_index].children[0].children[1].children[1].children[0].textContent;
                        if (tips.indexOf("Error") == -1) {
                            set_buttons(true);
                        } else {
                            set_buttons(false);
                        }
                        window['chat_bot1'].children[0].textContent = '作画中 ' + tips;
                    } else {
                        set_buttons(false);
                        window['chat_bot1'].children[0].textContent = '';
                    } 
                }
              
            } catch(e) {
            }        
        }
        window['checkChange_interval'] = window.setInterval("window.checkChange()", 500);         
    }
   
    return false;
}"""

space_ids = {
            "spaces/stabilityai/stable-diffusion":"Stable Diffusion 2.1",
            # "spaces/runwayml/stable-diffusion-v1-5":"Stable Diffusion 1.5",
            # "spaces/stabilityai/stable-diffusion-1":"Stable Diffusion 1.0",
            }

tab_actions = []
tab_titles = []

for space_id in space_ids.keys():
    print(space_id, space_ids[space_id])
    try:
        tab = gr.Interface.load(space_id)
        tab_actions.append(tab)
        tab_titles.append(space_ids[space_id])
    except Exception as e:
        logger.info(f"load_fail__{space_id}_{e}")
        
token_encoder = get_encoder()
total_tokens = 4096
max_output_tokens = 1024
max_input_tokens = total_tokens - max_output_tokens

def set_openai_api_key(api_key):
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        openai.api_key = api_key

def get_response_from_openai(input, chat_history, model_radio):
    error_1 = 'You exceeded your current quota, please check your plan and billing details.'
    def openai_create(input_list, model_radio):
        try:
            # print(f'input_list={input_list}')
            input_list_len = len(input_list)
            out_prompt = ''
            messages = []
            if model_radio == 'GPT-3.0':
                out_prompt = 'AI:'
            for i in range(input_list_len):
                input = input_list[input_list_len-i-1].replace("<br>", '\n\n')
                if input.startswith("Openai said:"):
                    input = "☝:"
    
                if input.startswith("☝:"):
                    if model_radio == 'GPT-3.0':
                        out_prompt = input.replace("☝:", "AI:") + '\n' + out_prompt
                    else:
                        out_prompt = input.replace("☝:", "") + out_prompt
                        messages.insert(0, {"role": "assistant", "content": input.replace("☝:", "")})
                elif input.startswith("☟:"):
                    if model_radio == 'GPT-3.0':
                        out_prompt = input.replace("☟:", "Human:") + '\n' + out_prompt
                    else:
                        out_prompt = input.replace("☟:", "") + out_prompt
                        messages.insert(0, {"role": "user", "content": input.replace("☟:", "")})
                tokens = token_encoder.encode(out_prompt)
                if len(tokens) > max_input_tokens: 
                    break
                
            if model_radio == 'GPT-3.0':
                # print(out_prompt)
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=out_prompt,
                    temperature=0.7,
                    max_tokens=max_output_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[" Human:", " AI:"]
                )
                # print(f'response_3.0__:{response}')
                ret = response.choices[0].text
            else:
                # print(messages)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_output_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[" Human:", " AI:"]
                )                            
                # print(f'response_3.5__:{response}')                
                ret = response.choices[0].message['content']
            if ret.startswith("\n\n"):
                ret = ret.replace("\n\n", '') 
            ret = ret.replace('\n', '<br>')  
            if ret == '':
                ret = f"Openai said: I'm too tired."   
            return ret, response.usage                           
        except Exception as e:
            logger.info(f"openai_create_error__{e}")
            ret = f"Openai said: {e} Perhaps enter your OpenAI API key."
            return ret, {"completion_tokens": -1, "prompt_tokens": -1, "total_tokens": -1}
    
    # logger.info(f'chat_history = {chat_history}')
    chat_history_list = []
    chat_history = chat_history.replace("<p>", "").replace("</p>", "")
    if chat_history != '':        
        chat_history_list = json.loads(chat_history)
    chat_history_list.append(f'☟:{input}')
        
    output, response_usage = openai_create(chat_history_list, model_radio)
    logger.info(f'response_usage={response_usage}')
    return output
    
def chat(input0, input1, chat_radio, model_radio, all_chat_history, chat_history):
    all_chat = []
    if all_chat_history != '':
        all_chat = json.loads(all_chat_history)

    if len(input0) == 0:
        return all_chat, json.dumps(all_chat), input0, input1
            
    if chat_radio == "Talk to chatGPT":
        response = get_response_from_openai(input0, chat_history, model_radio)
        all_chat.append((input0, response))
        return all_chat, json.dumps(all_chat), '', input1
    else:
        prompt_en = getTextTrans(input0, source='zh', target='en') + f',{random.randint(0,sys.maxsize)}'
        return all_chat, json.dumps(all_chat), input0, prompt_en
        
def chat_radio_change(chat_radio):
    if chat_radio == "Talk to chatGPT":
        return gr.Radio.update(visible=True), gr.Text.update(visible=True)
    else:
        return gr.Radio.update(visible=False), gr.Text.update(visible=False)

with gr.Blocks(title='Talk to chatGPT') as demo:    
    with gr.Row(elem_id="page_0", visible=False) as page_0:
        gr.HTML("<p>You can duplicating this space and use your own session token: <a style='display:inline-block' href='https://huggingface.co/spaces/yizhangliu/chatGPT?duplicate=true'><img src='https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14' alt='Duplicate Space'></a></p>")
    with gr.Group(elem_id="page_1", visible=True) as page_1:
        with gr.Box():            
            with gr.Row():
                start_button = gr.Button("Let's talk to chatGPT!", elem_id="start-btn", visible=True) 
                start_button.click(fn=None, inputs=[], outputs=[], _js=start_work)
                
    with gr.Row(elem_id="page_2", visible=False) as page_2:        
        with gr.Row(elem_id="chat_row"):
            chatbot = gr.Chatbot(elem_id="chat_bot", visible=False).style(color_map=("green", "blue"))
            chatbot1 = gr.Chatbot(elem_id="chat_bot1").style(color_map=("green", "blue"))
        with gr.Row(elem_id="prompt_row"):
            prompt_input0 = gr.Textbox(lines=2, label="input", elem_id="my_prompt", show_label=True)
            prompt_input1 = gr.Textbox(lines=4, label="prompt", elem_id="my_prompt_en", visible=False)
            chat_history = gr.Textbox(lines=4, label="chat_history", elem_id="chat_history", visible=False)
            all_chat_history = gr.Textbox(lines=4, label="会话上下文：", elem_id="all_chat_history", visible=False)

            chat_radio = gr.Radio(["Talk to chatGPT", "Text to Image"], elem_id="chat_radio",value="Talk to chatGPT", show_label=False, visible=True)
            model_radio = gr.Radio(["GPT-3.0", "GPT-3.5"], elem_id="model_radio", value="GPT-3.5", 
                                label='GPT model: ', show_label=True,interactive=True, visible=True) 
            openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
                                    show_label=False, lines=1, type='password')
        with gr.Row(elem_id="btns_row"):
            with gr.Column(id="submit_col"):
                submit_btn = gr.Button(value = "submit",elem_id="submit-btn").style(
                        margin=True,
                        rounded=(True, True, True, True),
                        width=100
                    )
            with gr.Column(id="clear_col"):
                clear_btn = gr.Button(value = "clear outputs", elem_id="clear-btn").style(
                        margin=True,
                        rounded=(True, True, True, True),
                        width=100
                    )
            submit_btn.click(fn=chat, 
                             inputs=[prompt_input0, prompt_input1, chat_radio, model_radio, all_chat_history, chat_history], 
                             outputs=[chatbot, all_chat_history, prompt_input0, prompt_input1],
                            )
        with gr.Row(elem_id='tab_img', visible=False).style(height=5):
           tab_img = gr.TabbedInterface(tab_actions, tab_titles)  

        openai_api_key_textbox.change(set_openai_api_key,
                                      inputs=[openai_api_key_textbox],
                                      outputs=[])
        openai_api_key_textbox.submit(set_openai_api_key,
                                      inputs=[openai_api_key_textbox],
                                      outputs=[])
        chat_radio.change(fn=chat_radio_change, 
                        inputs=[chat_radio], 
                        outputs=[model_radio, openai_api_key_textbox],
                        ) 

demo.launch(debug = True)

