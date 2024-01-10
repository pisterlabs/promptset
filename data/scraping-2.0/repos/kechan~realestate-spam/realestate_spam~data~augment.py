from typing import List, Tuple

import openai, time, re, json, gc, tempfile, os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
# will use openai API gpt-3.5-turbo and/or manually in chat.openai.com (chatGPT)

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3), retry_error_cls=RetryError)
def _retry_get_completion(model, temperature, max_tokens, messages):  
  response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
  return response.choices[0].message["content"]



class AugGPT:
  def __init__(self, model="gpt-3.5-turbo", temperature=0.5, max_tokens=1024):
    self.model = model
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.chatgpt_responses_file = None   # temp file to store chatgpt responses

    if openai.api_key is None: assert False, 'openai.api_key must be set'     

  def get_aug_messages(self, 
                       messages: List[str], 
                       batch_size=2, 
                       num_of_aug_mesg_per_mesg=5, 
                       return_raw_response_only=False,
                       verbose=True):
    raise NotImplementedError("This method should be overridden by subclass")

  def reconstruct_augs_df(self, messages: List[str], response_list: List[List[str]]):
    """
    Given the original messages and the response_list, reconstruct the augmented messages
    """
    raise NotImplementedError("This method should be overridden by subclass")
    
  def _format_chatgpt_responses(self, responses: List[str], batch_size) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Depending on the model and the prompt, the raw response mostly like
    need some nontrival formatting and error handling. 

    Input: 
    messages: List[str] - a list of raw responses from chatgpt
    batch_size: int - the number of messages in each batch


    Output:
    responses: List[str] - the raw responses from chatgpt
    response_list: A list of strings for each source message (e.g. 5 for each source message if you ask chatGPT to generate 5 messages)
    badly_formed_responses: List[str] - a list of raw responses that are badly formed
    """
    raise NotImplementedError("This method should be overridden by subclass")
       
  # openai.ChatCompletion.create
  def get_completion(self, prompt):
    messages = [{"role": "user", "content": prompt}]
    return _retry_get_completion(self.model, self.temperature, self.max_tokens, messages)
    


  def _partial_json_parse(self, payload, keys):
    """
    Helper func to fix raw response that get truncated. Given the keys as guidance, it will try its
    best to parse the raw response into a dictionary, and abandon any key/value that's malformed due
    to truncation. (LLM has finite context or other stochastic behavior, so this can happen.)
    """
    results = {}

    for key in keys:
        # Build a regular expression to match the key and its associated array
        regex = f'"{key}"\\s*:\\s*\\[(.*?)\\](?=,|\\}})'
        match = re.search(regex, payload, re.DOTALL)

        if match:
            array_string = match.group(1)
            # Split the array string on commas, but only if they are not enclosed in double quotes
            array_values = re.findall(r'"([^"]*)"', array_string)
            results[key] = array_values

    return results


class TestAugGPT(AugGPT):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_aug_messages(self, messages: List[str], batch_size=2, num_of_aug_mesg_per_mesg=5, return_raw_response_only=False, verbose=True) -> Tuple[pd.DataFrame, List[str], List[List[str]], List[str]]:
    '''
    Input: 
    messages: a list of source messages to be data augmented.
    batch_size: the # of source messages to submit to chatGPT per completion request.
    num_of_aug_mesg_per_mesg: the # of augmented messages to generate per source message.

    Output:
    aug_df: a dataframe with columns: message, message_augs where message is the original message and message_augs is corresponding augmented messages
    chatgpt_raw_resp: the list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    '''
    prompt_generator = self.get_prompt_generator(messages, batch_size=batch_size, num_of_aug_mesg_per_mesg=num_of_aug_mesg_per_mesg)
    chatgpt_responses = []
    for k, prompt in enumerate(prompt_generator):
      if verbose:
        print(f"Processing batch {k+1} of {len(messages)//batch_size + 1} using prompt:\n{prompt}")
      else:
        print(f"Processing batch {k+1} of {len(messages)//batch_size + 1}")
              
      try:
        response = self.get_completion(prompt)
        chatgpt_responses.append(response)
        time.sleep(1)
      except Exception as e:  # possibly no response, or malformed response
        print(f"Error: {e}")
        # just put in empty placeholders
        chatgpt_responses.append('{' + ''.join([f'"text{i}": [], ' for i in range(num_of_aug_mesg_per_mesg)]) + '}')
        time.sleep(5)    # sleep a bit more after a failure
        continue

    # chatGPT response cost $$, so save them in a temp file generated on the fly
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('\n'.join(chatgpt_responses))
      self.chatgpt_responses_file = f.name
      print('Raw chatgpt responses saved to: ', self.chatgpt_responses_file)

    if return_raw_response_only:
      return None, chatgpt_responses, [[]], []

    # attempt to format and fix any malformed responses if possible     
    chatgpt_responses, response_list, badly_formed_responses = self._format_chatgpt_responses(chatgpt_responses, batch_size=batch_size)

    # len(response_list) can be > than len(messages) if last batch is not full, so truncate response_list
    response_list = response_list[:len(messages)]

    test_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})
    test_augs_df['n'] = test_augs_df.message_aug.apply(len)
    test_augs_df['cat_message_aug'] = test_augs_df.message_aug.apply(lambda x: ' | '.join(x))

    return test_augs_df, chatgpt_responses, response_list, badly_formed_responses

  def reconstruct_augs_df(self, messages: List[str], response_list: List[List[str]]) -> pd.DataFrame:
    """
    chatGPT can give malformed json (or no response at all). After response_list is manually fixed,
    we can reconstruct the test_augs_df

    input: list of messages and list of list of augmented messages
    output: dataframe with original message
    """
    test_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})
    test_augs_df['n'] = test_augs_df.message_aug.apply(len)
    test_augs_df['cat_message_aug'] = test_augs_df.message_aug.apply(lambda x: ' | '.join(x))

    return test_augs_df

    
  def get_prompt_generator(self, messages: List[str], batch_size, num_of_aug_mesg_per_mesg):
    def add_backticks(messages: List[str]) -> str:
      return "\n".join([f"```{m}```" for m in messages])
    
    sys_prompt = f"""
You are AugGPT able to take test messages delimited by ``` as inputs and output 5 \
samples for each input with similar semantic meaning as data augmentation. You must \
include the string 'test' somewhere to indicate they are test messages. Provide \
responses with pure JSON format with key "text" following by number and value the list \
of {num_of_aug_mesg_per_mesg} samples. Important: randomize PII wherever you can. Limit to each sentence to less than 20 words.

    """
    
    for i in range(0, len(messages), batch_size):
      batch_messages = messages[i:i+batch_size]
      backticked_messages = add_backticks(batch_messages)

      final_prompt = f"{sys_prompt}\n{backticked_messages}\n"
      yield final_prompt
      
  
  def _format_chatgpt_responses(self, responses: List[str], batch_size) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Output: 
    responses: a list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    """
    badly_formed_responses = []
    response_list = []
    for response in responses:
      try:
        response_dict = eval(response)
        for i in range(batch_size):
          response_list.append(response_dict.get(f'text{i}', []))
        # response_list.append(response_dict.get('text1', []))
        # response_list.append(response_dict.get('text2', []))
      except Exception as e:
        print(f"Parsing Error: {e}")
        try:
          badly_formed_responses.append(response)
          # try again with partial_json_parse
          keys = [f'text{i}' for i in range(batch_size)]
          response_dict = self._partial_json_parse(response, keys)
          for i in range(batch_size):
            response_list.append(response_dict.get(f'text{i}', []))
          # response_list.append(response_dict.get('text1', []))
          # response_list.append(response_dict.get('text2', []))
        except Exception as e:
          print(f"Error: {e}")
          for i in range(batch_size):
            response_list.append([])

    return responses, response_list, badly_formed_responses


class KvCoreAugGPT(AugGPT):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_aug_messages(self, messages: List[str], batch_size=2, num_of_aug_mesg_per_mesg=5, return_raw_response_only=False, verbose=True) -> Tuple[pd.DataFrame, List[str], List[List[str]], List[str]]:
    '''
    Input: 
    messages: a list of source messages to be data augmented.
    batch_size: the # of source messages to submit to chatGPT per completion request.
    num_of_aug_mesg_per_mesg: the # of augmented messages to generate per source message.

    Output:
    aug_df: a dataframe with columns: message, message_augs where message is the original message and message_augs is corresponding augmented messages
    chatgpt_raw_resp: the list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    '''
    prompt_generator = self.get_prompt_generator(messages, batch_size=batch_size, num_of_aug_mesg_per_mesg=num_of_aug_mesg_per_mesg)
    chatgpt_responses = []
    for k, prompt in enumerate(prompt_generator):
      if verbose:
        print(f"Processing batch {k+1} of {len(messages)//batch_size + 1} using prompt:\n{prompt}")
      else:
        print(f"Processing batch {k+1} of {len(messages)//batch_size + 1}")

      try:
        response = self.get_completion(prompt)
        if not response.startswith('{"m0": ["kvCore'): response = ('{"m0": ["kvCore ' + response)
        chatgpt_responses.append(response)
        time.sleep(1)
      except Exception as e:   # possible timeout, no response, or malformed response
        print(f"Error: {e}")
        # just put in empty placeholders
        # for i in range(batch_size): chatgpt_responses.append(f'"m{i}": "", ')   # put in empty for the whole batch if failed.
        chatgpt_responses.append('{' + ''.join([f'"m{i}": "", ' for i in batch_size]) + '}')
        time.sleep(5)
        continue    # sleep a bit more after a failure

    # chatGPT response cost $$, so save them in a temp file generated on the fly
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('\n'.join(chatgpt_responses))
      self.chatgpt_responses_file = f.name
      print('Raw chatgpt responses saved to: ', self.chatgpt_responses_file)

    if return_raw_response_only:
      return None, chatgpt_responses, [[]], []

    # attempt to format and fix any malformed responses if possible
    chatgpt_responses, response_list, badly_formed_responses = self._format_chatgpt_responses(chatgpt_responses, batch_size=batch_size)

    # len(response_list) can be > than len(messages) if last batch is not full, so truncate response_list
    response_list = response_list[:len(messages)]

    kvcore_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})
    kvcore_augs_df['n'] = kvcore_augs_df.message_aug.apply(len)
    kvcore_augs_df['cat_message_aug'] = kvcore_augs_df.message_aug.apply(lambda x: ' | '.join(x))

    return kvcore_augs_df, chatgpt_responses, response_list, badly_formed_responses


  def reconstruct_augs_df(self, messages: List[str], response_list: List[List[str]]):
    """
    chatGPT can give malformed json (or no response at all). After response_list is manually fixed,
    we can reconstruct the test_augs_df

    input: list of messages and list of list of augmented messages
    output: dataframe with original message
    """
    kvcore_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})
    kvcore_augs_df['n'] = kvcore_augs_df.message_aug.apply(len)
    kvcore_augs_df['cat_message_aug'] = kvcore_augs_df.message_aug.apply(lambda x: ' | '.join(x))

    return kvcore_augs_df


  def get_prompt_generator(self, messages: List[str], batch_size, num_of_aug_mesg_per_mesg):
    sys_prompt = f"""
You are AugGPT able to take a contact message and return {num_of_aug_mesg_per_mesg} variations with similar format as data augmentation. 
Provide responses in pure JSON format. Important: just randomize PII wherever you can without redacting. Limit 
each to less than 50 words. I can provide more than 1 message via a python dict, and you should response with key 
"m0" for 1st input message containing your 5 generated messages, and key "m1" for the 2nd, and so on. The input 
messages are independent of each other.
    """

    for i in range(0, len(messages), batch_size):
      batch_messages = messages[i:i+batch_size]
      input_n_format_prompt = "Me: {"
      for i, message in enumerate(batch_messages):
        input_n_format_prompt += f'"m{i}": "{message}", '
      input_n_format_prompt += "}"
      input_n_format_prompt += "\nYou: {\"m0\": [\"kvCore"

      final_prompt = f"{sys_prompt}\n\n{input_n_format_prompt}"
      yield final_prompt

  def _format_chatgpt_responses(self, responses: List[str], batch_size) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Output: 
    responses: a list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    """
    badly_formed_responses = []
    response_list = []
    for response in responses:
      try:
        response_dict = eval(response.replace('\n', ' '))
        for i in range(batch_size):
          response_list.append(response_dict.get(f'm{i}', []))
      except Exception as e:
        print(f"Parsing Error: {e}")
        try:
          badly_formed_responses.append(response)
          # try again with partial_json_parse
          response_dict = self._partial_json_parse(response, [f'm{i}' for i in range(batch_size)])
          for i in range(batch_size):
            response_list.append(response_dict.get(f'm{i}', []))

        except Exception as e:
          print(f"Error: {e}")
          for i in range(batch_size):
            response_list.append([])

    return responses, response_list, badly_formed_responses


class ChineseAugGPT(AugGPT):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_aug_messages(self, messages: List[str], batch_size=1, num_of_aug_mesg_per_mesg=1, return_raw_response_only=False, verbose=True) -> Tuple[pd.DataFrame, List[str], List[List[str]], List[str]]:
    '''
    Input: 
    messages: a list of source messages to be data augmented (source should be in english)
    batch_size: the # of source messages to submit to chatGPT per completion request. (only 1 allowed for now)
    num_of_aug_mesg_per_mesg: the # of augmented messages to generate per source message (only 1 for chinese augmentation from translation)

    Output:
    aug_df: a dataframe with columns: message, message_augs where message is the original message and message_augs is corresponding augmented messages
    chatgpt_raw_resp: the list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    '''
    assert batch_size == 1, "batch_size must be 1 for chinese augmentation"
    assert num_of_aug_mesg_per_mesg == 1, "num_of_aug_mesg_per_mesg must be 1 for chinese augmentation"

    prompt_generator = self.get_prompt_generator(messages, batch_size=batch_size, num_of_aug_mesg_per_mesg=num_of_aug_mesg_per_mesg)
    chatgpt_responses = []
    for k, prompt in enumerate(prompt_generator):
      print(f'Processing {k} using prompt: {prompt}')
      try:
        response = self.get_completion(prompt)
        chatgpt_responses.append(response)
        time.sleep(1)

      except RetryError as e: 
        print(f"Max retries exceeded for k: {k}, prompt: {prompt}")
        print(f"Error: {e}") 
        chatgpt_responses.append(' ')       # just put in empty placeholders
        continue

      except Exception as e:  # possibly no response, or malformed response
        print(f"Error: {e}")        
        chatgpt_responses.append(' ')       # just put in empty placeholders
        time.sleep(5)    # sleep a bit more after a failure
        continue

    # chatGPT response cost $$, so save them in a temp file generated on the fly
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
      f.write('\n'.join(chatgpt_responses))
      self.chatgpt_responses_file = f.name
      print('Raw chatgpt responses saved to: ', self.chatgpt_responses_file)

    if return_raw_response_only:
      return None, chatgpt_responses, [[]], []
    
    chatgpt_responses, response_list, badly_formed_responses = self._format_chatgpt_responses(chatgpt_responses, batch_size=batch_size)

    # len(response_list) can be > than len(messages) if last batch is not full, so truncate response_list
    response_list = response_list[:len(messages)]

    chinese_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})

    return chinese_augs_df, chatgpt_responses, response_list, badly_formed_responses


  def get_prompt_generator(self, messages: List[str], batch_size=1, num_of_aug_mesg_per_mesg=1):
    assert batch_size == 1, "batch_size must be 1 for chinese augmentation"
    assert num_of_aug_mesg_per_mesg == 1, "num_of_aug_mesg_per_mesg must be 1 for chinese augmentation"
    
    for i in range(0, len(messages), batch_size):
      batch_messages = messages[i:i+batch_size]
      
      prompt = f"""Please translate the paragraph delimited by ``` (3 backticks) to Chinese:\n```\n{batch_messages[0]}\n```
      """
      yield prompt

  def _format_chatgpt_responses(self, responses: List[str], batch_size) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Output: 
    responses: a list of raw responses from chatgpt (for safe keeping, this is expensive to obtain)
    response_list: post-formatted list of responses from chatgpt, empty => unrecoverd malformed response
    badly_formed_responses: list of raw malformed responses (for debugging)
    """
    badly_formed_responses = []
    response_list = responses.copy()
    
    return responses, response_list, badly_formed_responses

  def reconstruct_augs_df(self, messages: List[str], response_list: List[List[str]]) -> pd.DataFrame:
    """
    chatGPT can give malformed json (or no response at all). After response_list is manually fixed,
    we can reconstruct the test_augs_df

    input: list of messages and list of list of augmented messages
    output: dataframe with original message
    """
    chinese_augs_df = pd.DataFrame(data={'message': messages, 'message_aug': response_list})

    return chinese_augs_df
  

class AugmentedDataGenerator:
  def __init__(self, external_data_df, train_df, filter_with_train=True):
    # dataframe with columns: NOTE, NOTE_AUG where NOTE is the original message and NOTE_AUG is corresponding augmented messages

    # train_df (after split)    
    self.train_df = train_df.copy()
    self.incl_contact_struct_info = 'CONTACT_STRUCT_INFO' in self.train_df.columns 

    # find all external_data_df source note (NOTE) thats a substring in at least one of train_df.NOTE
    # Skip if external_data_df does not have source (NOTE),

    if 'NOTE' in external_data_df.columns and filter_with_train:
      train_notes = list(self.train_df.NOTE.unique())
      external_source_notes = list(external_data_df.NOTE.unique())
      
      wanted_external_source_notes = []
      for txt in external_source_notes:
        for train_note in train_notes:
          if txt in train_note:
            wanted_external_source_notes.append(txt)
            break
      
      self.external_data_df = external_data_df.q("NOTE.isin(@wanted_external_source_notes)").copy()
    else:
      self.external_data_df = external_data_df.copy()

  def apply_additional_filters(self, expr: str):
    '''
    Filter self.external_data_df by expr
    '''
    self.external_data_df = self.external_data_df.q(expr).copy()
    gc.collect()


class TestAugmentedDataGenerator(AugmentedDataGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.expressions = [
        (r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b", True),
        ("Website Contact Request - ", False),
        ("Website Information Request - ", False),
        ("Website Home Valuation Request - ", False),
        ("Website Dream Home Request - ", False),
        ("Website Careers Request", False),
        ("Website Join Team Request", False),
        ("New Exclusive Buying", False),
        ("New Exclusive Selling", False),
        ("Website Listing Info Request - ", False),
        ("Website Free Home Evaluation Request - ", False),
        (r"MLS® \w*\d+\s*-?\s*", True),
        (r"MLS® \w*\d+Website Listing Info Request - ", True),
        (r"MLS&reg; \w*\d+\s*-?\s*", True),
        (r"MLS&reg; \w*\d+Website Listing Info Request - ", True),
        (r"Single Property Website\s*:\s*New MLS® \w*\d+", True),
        (r"Single Property Website\s*:?\s*", True),
        (r"New Listing ID:\s*\w*\d+\s*", True),
        (r"Listing ID:\s*\w*\d+\s*", True),
        (r"(?i)(buying|selling|both)\s+house\s+([$\d,]+)\s+-\s+([$\d,]+)", True),
        # ("^\s*House\s*", True),
        (r"[\\n\s]House[\\n\s]", True),
        ("[\\n\s]Buying House[\\n\s]", True),
        ("[\\n\s]Selling House[\\n\s]", True),
        ("[\\n\s]Both House[\\n\s]", True)
    ]

  def generate_aug_df(self, n):
    touch_entry_ids, source_notes, notes, display_names, contact_struct_infos = [], [], [], [], []
    for i in tqdm(range(n)):
      source_message, new_message, display_name, contact_struct_info, matches, chatgpt_augmented_mesg = self.generate_new_message()
      source_notes.append(source_message)
      notes.append(new_message)
      display_names.append(display_name)
      contact_struct_infos.append(contact_struct_info)
      touch_entry_ids.append(-np.random.randint(0, 1000000))

    if self.incl_contact_struct_info:
      aug_test_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids, 
                                      'SOURCE_NOTE': source_notes, 
                                      'NOTE': notes, 
                                      'DISPLAY_NAME': display_names, 
                                      'CONTACT_STRUCT_INFO': contact_struct_infos, 
                                      'class_label': 'TEST'})
    else:
      aug_test_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids, 
                                      'SOURCE_NOTE': source_notes, 
                                      'NOTE': notes, 
                                      'DISPLAY_NAME': display_names, 
                                      'class_label': 'TEST'})
  
    return aug_test_df
  
  def generate_new_message(self):
    # randomly pick a source message and DISPLAY_NAME from train_df that's of label == 'TEST'
    source_message = self.train_df.q("class_label == 'TEST'").sample(1).NOTE.values[0]
    
    if self.incl_contact_struct_info:
      display_name, contact_struct_info = self.train_df.sample(1)[['DISPLAY_NAME', 'CONTACT_STRUCT_INFO']].values[0]
    else:
      display_name = self.train_df.sample(1)[['DISPLAY_NAME']].values[0][0]

    matches = self._find_pattern_matches(source_message)

    sample_external_data_df = self.external_data_df.sample(1)
    original_message = sample_external_data_df.NOTE.values[0]
    augmented_message = sample_external_data_df.NOTE_AUG.values[0]

    # randomize between ' ' and '\n'
    sep_c1 = ' ' if np.random.rand() > 0.5 else '\n'
    sep_c2 = ' ' if np.random.rand() > 0.5 else '\n'
    
    try:
      if len(matches) == 0:
        new_message = augmented_message
      else:
        new_message = self._replace_with_message(source_message, matches, sep_c1 + augmented_message + sep_c2)
    except Exception as e:
      print(source_message, matches, augmented_message)
      print(f"Exception: {e}")
      raise e
    
    if self.incl_contact_struct_info:
      return source_message, new_message, display_name, contact_struct_info, matches, augmented_message
    else:
      return source_message, new_message, display_name, None, matches, augmented_message
  
  def _replace_with_message(self, message, ranges, test_message):
    # sort ranges by start position
    ranges.sort(key=lambda x: x[0])

    # merge overlapping ranges
    merged_ranges = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))

    # create a list to hold the parts of the message
    parts = []
    previous_end = 0
    for start, end in merged_ranges:
        if start != previous_end:
            parts.append(test_message)
        parts.append(message[start:end])
        previous_end = end

    if previous_end != len(message):
        parts.append(test_message)
    
    # join the parts into a single string
    result = ''.join(parts)
    
    return result
  
  def _find_pattern_matches(self, message):

    matches = []
    for expression, is_regex in self.expressions:
        if is_regex:
            pattern = re.compile(expression)
            for match in pattern.finditer(message):
                matches.append((match.start(), match.end()))
        else:
            start = 0
            while start >= 0:
                start = message.find(expression, start)
                if start >= 0:
                    end = start + len(expression)
                    matches.append((start, end))
                    start = end

    return matches


class KvCoreAugmentedDataGenerator(AugmentedDataGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_aug_df(self, n):
    touch_entry_ids, source_notes, notes, display_names, contact_struct_infos = [], [], [], [], []
    for i in tqdm(range(n)):
      source_message, new_message, display_name, contact_struct_info = self.generate_new_message()
      source_notes.append(source_message)
      notes.append(new_message)
      display_names.append(display_name)
      contact_struct_infos.append(contact_struct_info)
      touch_entry_ids.append(-np.random.randint(0, 1000000))

    if self.incl_contact_struct_info:
      aug_kvcore_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids,
                                          'SOURCE_NOTE': source_notes,
                                          'NOTE': notes, 
                                          'DISPLAY_NAME': display_names, 
                                          'CONTACT_STRUCT_INFO': contact_struct_infos, 
                                          'class_label': 'NOT_SPAM'})    # all kvcore messages are not spam
    else:
      aug_kvcore_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids,
                                          'SOURCE_NOTE': source_notes,
                                          'NOTE': notes, 
                                          'DISPLAY_NAME': display_names, 
                                          'class_label': 'NOT_SPAM'})
    
    return aug_kvcore_df
  
  def generate_new_message(self):
    # randomly pick a source message and DISPLAY_NAME from train_df that's starts with 'kvCore found'
    sample_external_data_df = self.external_data_df.sample(1)

    if self.incl_contact_struct_info:
      display_name, contact_struct_info = self.train_df.sample(1)[['DISPLAY_NAME', 'CONTACT_STRUCT_INFO']].values[0]   # use all names possible
    else:
      display_name = self.train_df.sample(1)[['DISPLAY_NAME']].values[0][0]   

    source_message = sample_external_data_df.NOTE.values[0]
    augmented_message = sample_external_data_df.NOTE_AUG.values[0]

    if self.incl_contact_struct_info:
      return source_message, augmented_message, display_name, contact_struct_info
    else:
      return source_message, augmented_message, display_name, None

  
class KaggleToxicAugmentedDataGenerator(AugmentedDataGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.expressions = [
        (r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b", True),
        ("Website Contact Request - ", False),
        ("Website Information Request - ", False),
        ("Website Home Valuation Request - ", False),
        ("Website Dream Home Request - ", False),
        ("Website Careers Request", False),
        ("Website Join Team Request", False),
        ("New Exclusive Buying", False),
        ("New Exclusive Selling", False),
        ("Website Listing Info Request - ", False),
        ("Website Free Home Evaluation Request - ", False),
        (r"MLS® \w*\d+\s*-?\s*", True),
        (r"MLS® \w*\d+Website Listing Info Request - ", True),
        (r"MLS&reg; \w*\d+\s*-?\s*", True),
        (r"MLS&reg; \w*\d+Website Listing Info Request - ", True),
        (r"Single Property Website\s*:\s*New MLS® \w*\d+", True),
        (r"Single Property Website\s*:?\s*", True),
        (r"New Listing ID:\s*\w*\d+\s*", True),
        (r"Listing ID:\s*\w*\d+\s*", True),
        (r"(?i)(buying|selling|both)\s+house\s+([$\d,]+)\s+-\s+([$\d,]+)", True),
        # ("^\s*House\s*", True),
        (r"[\\n\s]House[\\n\s]", True),
        ("[\\n\s]Buying House[\\n\s]", True),
        ("[\\n\s]Selling House[\\n\s]", True),
        ("[\\n\s]Both House[\\n\s]", True)
    ]



  def generate_aug_df(self, n):
    touch_entry_ids, source_notes, notes, display_names, contact_struct_infos = [], [], [], [], []
    for i in tqdm(range(n)):
      source_message, new_message, display_name, contact_struct_info, matches, chatgpt_augmented_mesg = self.generate_new_message()
      source_notes.append(source_message)
      notes.append(new_message)
      display_names.append(display_name)
      contact_struct_infos.append(contact_struct_info)
      touch_entry_ids.append(-np.random.randint(0, 1000000))

    if self.incl_contact_struct_info:
      aug_kaggle_toxic_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids,
                                          'SOURCE_NOTE': source_notes,
                                          'NOTE': notes,
                                          'DISPLAY_NAME': display_names,
                                          'CONTACT_STRUCT_INFO': contact_struct_infos,
                                          'class_label': 'SPAM'})    # all toxic messages are SPAM
    else:
      aug_kaggle_toxic_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids,
                                          'SOURCE_NOTE': source_notes,
                                          'NOTE': notes,
                                          'DISPLAY_NAME': display_names,
                                          'class_label': 'SPAM'})
    
    return aug_kaggle_toxic_df
  
  def generate_new_message(self):
    # randomly independently sample a source message and DISPLAY_NAME from train_df 

    source_message = self.train_df.sample(1).NOTE.values[0]
    if self.incl_contact_struct_info:
      display_name, contact_struct_info = self.train_df.sample(1)[['DISPLAY_NAME', 'CONTACT_STRUCT_INFO']].values[0]
    else:
      display_name = self.train_df.sample(1)[['DISPLAY_NAME']].values[0][0]

    matches = self._find_pattern_matches(source_message)   # identify common sys generated pattern pecurliar to first touch entry NOTE

    sample_external_data_df = self.external_data_df.sample(1)   # sample a toxic message from kaggle toxic dataset
    augmented_message = sample_external_data_df.NOTE_AUG.values[0]

    # randomize separators between ' ' and '\n'
    sep_c1 = ' ' if np.random.rand() > 0.5 else '\n'
    sep_c2 = ' ' if np.random.rand() > 0.5 else '\n'

    try:
      new_message = self._replace_with_message(source_message, matches, sep_c1 + augmented_message + sep_c2)
    except Exception as e:
      print(f'source_message: {source_message}, matches: {matches}, augmented_message: {augmented_message}')
      print(f"Exception: {e}")
      raise e
    
    if self.incl_contact_struct_info:
      return source_message, new_message, display_name, contact_struct_info, matches, augmented_message
    else:
      return source_message, new_message, display_name, None, matches, augmented_message
  
  def _replace_with_message(self, message, ranges, test_message):
    # if range is empty, return the text_message 
    if len(ranges) == 0:
      return test_message
    
    # sort ranges by start position
    ranges.sort(key=lambda x: x[0])

    # merge overlapping ranges
    merged_ranges = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))

    # create a list to hold the parts of the message
    parts = []
    previous_end = 0
    for start, end in merged_ranges:
        if start != previous_end:
            parts.append(test_message)
        parts.append(message[start:end])
        previous_end = end

    if previous_end != len(message):
        parts.append(test_message)
    
    # join the parts into a single string
    result = ''.join(parts)
    
    return result

  def _find_pattern_matches(self, message):

    matches = []
    for expression, is_regex in self.expressions:
        if is_regex:
            pattern = re.compile(expression)
            for match in pattern.finditer(message):
                matches.append((match.start(), match.end()))
        else:
            start = 0
            while start >= 0:
                start = message.find(expression, start)
                if start >= 0:
                    end = start + len(expression)
                    matches.append((start, end))
                    start = end

    return matches


class ChineseAugmentedDataGenerator(AugmentedDataGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.expressions = [
        (r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b", True),
        ("Website Contact Request - ", False),
        ("Website Information Request - ", False),
        ("Website Home Valuation Request - ", False),
        ("Website Dream Home Request - ", False),
        ("Website Careers Request", False),
        ("Website Join Team Request", False),
        ("New Exclusive Buying", False),
        ("New Exclusive Selling", False),
        ("Website Listing Info Request - ", False),
        ("Website Free Home Evaluation Request - ", False),
        (r"MLS® \w*\d+\s*-?\s*", True),
        (r"MLS® \w*\d+Website Listing Info Request - ", True),
        (r"MLS&reg; \w*\d+\s*-?\s*", True),
        (r"MLS&reg; \w*\d+Website Listing Info Request - ", True),
        (r"Single Property Website\s*:\s*New MLS® \w*\d+", True),
        (r"Single Property Website\s*:?\s*", True),
        (r"New Listing ID:\s*\w*\d+\s*", True),
        (r"Listing ID:\s*\w*\d+\s*", True),
        (r"(?i)(buying|selling|both)\s+house\s+([$\d,]+)\s+-\s+([$\d,]+)", True),
        # ("^\s*House\s*", True),
        (r"[\\n\s]House[\\n\s]", True),
        ("[\\n\s]Buying House[\\n\s]", True),
        ("[\\n\s]Selling House[\\n\s]", True),
        ("[\\n\s]Both House[\\n\s]", True)
    ]


  def generate_aug_df(self, n, label, chinese_display_names=None):
    touch_entry_ids, source_notes, notes, display_names, contact_struct_infos = [], [], [], [], []
    query = f'class_label == "{label}"'
    for i in tqdm(range(n)):
      source_message, new_message, display_name, contact_struct_info, matches, chatgpt_augmented_mesg = self.generate_new_message(query=query, chinese_display_names=chinese_display_names)
      source_notes.append(source_message)
      notes.append(new_message)
      display_names.append(display_name)
      contact_struct_infos.append(contact_struct_info)
      touch_entry_ids.append(-np.random.randint(0, 1000000))

    if self.incl_contact_struct_info:
      aug_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids, 
                                      'SOURCE_NOTE': source_notes, 
                                      'NOTE': notes, 
                                      'DISPLAY_NAME': display_names, 
                                      'CONTACT_STRUCT_INFO': contact_struct_infos, 
                                      'class_label': label})
    else:
      aug_df = pd.DataFrame(data={'TOUCH_ENTRY_ID': touch_entry_ids, 
                                      'SOURCE_NOTE': source_notes, 
                                      'NOTE': notes, 
                                      'DISPLAY_NAME': display_names, 
                                      'class_label': label})
  
    return aug_df
  
  def generate_new_message(self, query, chinese_display_names=None):
    source_message = self.train_df.q(query).sample(1).NOTE.values[0]

    # sample a chinese names
    if self.incl_contact_struct_info and 'lang' in self.train_df.columns:
      display_name, contact_struct_info = self.train_df.q(f"lang == 'zh'").sample(1)[['DISPLAY_NAME', 'CONTACT_STRUCT_INFO']].values[0]
    elif chinese_display_names is not None:
      # sample a display_name from the list chinese_display_names
      display_name = np.random.choice(chinese_display_names)

    matches = self._find_pattern_matches(source_message)

    sample_external_data_df = self.external_data_df.sample(1)
    # original_message = sample_external_data_df.NOTE.values[0]
    augmented_message = sample_external_data_df.NOTE_AUG.values[0]

    # randomize between ' ' and '\n'
    sep_c1 = ' ' if np.random.rand() > 0.5 else '\n'
    sep_c2 = ' ' if np.random.rand() > 0.5 else '\n'

    try:
      if len(matches) == 0:
        new_message = augmented_message
      else:
        new_message = self._replace_with_message(source_message, matches, sep_c1 + augmented_message + sep_c2)
    except Exception as e:
      print(source_message, matches, augmented_message)
      print(f"Exception: {e}")
      raise e

    if self.incl_contact_struct_info:
      return source_message, new_message, display_name, contact_struct_info, matches, augmented_message
    else:
      return source_message, new_message, display_name, None, matches, augmented_message
  
  def _replace_with_message(self, message, ranges, test_message):
    # sort ranges by start position
    ranges.sort(key=lambda x: x[0])

    # merge overlapping ranges
    merged_ranges = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))

    # create a list to hold the parts of the message
    parts = []
    previous_end = 0
    for start, end in merged_ranges:
        if start != previous_end:
            parts.append(test_message)
        parts.append(message[start:end])
        previous_end = end

    if previous_end != len(message):
        parts.append(test_message)
    
    # join the parts into a single string
    result = ''.join(parts)
    
    return result
  
  def _find_pattern_matches(self, message):

    matches = []
    for expression, is_regex in self.expressions:
        if is_regex:
            pattern = re.compile(expression)
            for match in pattern.finditer(message):
                matches.append((match.start(), match.end()))
        else:
            start = 0
            while start >= 0:
                start = message.find(expression, start)
                if start >= 0:
                    end = start + len(expression)
                    matches.append((start, end))
                    start = end

    return matches
    



if __name__ == '__main__':
  messages = ['This is a test message', 'Test per Lorna', "Hi mom, This is just Shelly and I'm testing something with our lead generation forms."]

  aug_gpt = AugGPT()
  test_augs_df, response_list, badly_formed_responses = aug_gpt.get_test_data_aug(messages)

  # examine test_augs_df and badly_formed_responses for any problems.

  # chatgpt_test_augs_df = pd.read_feather(data/'chatgpt_test_augs_df')
  # aug_gen = AugmentedDataGenerator(chatgpt_test_augs_df, train_df)
  """
test_augs_df.n.value_counts()
_df = test_augs_df.q('n == 0')#.iloc[:5]
_df
fix_idxs = _df.index
print(fix_idxs)

# assign key m1, m2, ... to each message, and then serialize the dictionary

print(json.dumps({f'm{k}': m for k, m in enumerate(_df.NOTE.values)}))

# paste the prompt + context into chat.openai.com, 
You are AugGPT able to take a test message return 5 messages with similar semantic meaning as data augmentation. You must include the string 'test' somewhere to indicate they are test messages.  Provide responses with pure JSON format. Important: randomize PII wherever you can. Limit to each sentence to less than 20 words. I can provide more than 1 message via a python dict, and you should response with key "m0" for 1st input message containing your 5 generated messages, and key "m1" for the 2nd, and so on. The input messages are independent of each other.

Me: [put json dumps here]
You:

# And then paste the chatgpt's reponse as payload
payload = {"m0": ["I'm sending a test message to make sure you receive it.", "Please ... }

for k, idx in enumerate(fix_idxs):
  response_list[idx] = payload[f'm{k}']

# if there are problems, fix them manually using chat.openai.com and then reconstruct test_augs_df
test_augs_df = aug_gpt.reconstruct_test_augs_df(messages, response_list)
  """