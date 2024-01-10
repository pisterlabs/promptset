import os
import openai

import time

from utils.BookProcessing import rechunk_book

BEGIN_ANSWER_TAG = "### BEGIN ANSWER ###"

openai.api_key = os.getenv("OPENAI_API_KEY")

# openai.api_key_path = "../gptapikey"

with open("../GPTkeys/ar_key.txt", "r") as f:
  openai.api_key = f.read().strip()

def ParaphraseChunk(chunk):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system",
       "content": "You are a helpful assistant that paraphrases book snippets. Begin your answer with a {} tag.".format(BEGIN_ANSWER_TAG)},
      {"role": "user", "content": 'Paraphrase the following book excerpt: "{}"'.format(chunk)}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer


def SummarizeChunk(chunk):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant that summarizes book snippets. Begin your answer with a {} tag.".format(BEGIN_ANSWER_TAG)},
      {"role": "user", "content": 'Summarize the following book excerpt: "{}". Start your answer with a "{}" tag.'.format(chunk, BEGIN_ANSWER_TAG)} # Add "in under 500 words?
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:

    raise ValueError("GPT response does not have a {} tag. Response: {}".format(BEGIN_ANSWER_TAG, response_content))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer

def SummarizeChunkRetry(chunk):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant that summarizes book snippets. Begin your answer with a {} tag.".format(BEGIN_ANSWER_TAG)},
      {"role": "user", "content": 'Summarize the following book excerpt: "{}". Make sure to begin your answer with a {} tag!'.format(chunk, BEGIN_ANSWER_TAG)} # Add "in under 500 words?
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:

    return response_content, "GPT response does not have a {} tag. Response: {}".format(BEGIN_ANSWER_TAG, response_content)

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer, None

def SummarizeSummaries(chunk_summaries, max_length=10000):

  meta_chunks = rechunk_book(["Summary {}: {}\n".format(ind, summary) for ind, summary in enumerate(chunk_summaries)], max_length)
  result = []

  for ind, m in enumerate(meta_chunks):
    
    if ind % 10 == 0:
      print("Processed {} meta chunks".format(ind + 1))

    for attempt in range(10):
      try:
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system",
             "content": "You are a helpful assistant that summarizes book scene summaries. Begin your answer with a {} tag.".format(
               BEGIN_ANSWER_TAG)},
            {"role": "user",
             "content": 'Summarize the following book excerpt summaries into one summary: "{}". Make sure to begin your answer with a {} tag!'.format(m, BEGIN_ANSWER_TAG)}  # Add "in under 500 words?
          ]
        )
        cur_response = response["choices"][0]["message"]["content"]
    
        if BEGIN_ANSWER_TAG not in cur_response:
          raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

        resp_parts = cur_response.split(BEGIN_ANSWER_TAG)

        if len(resp_parts) != 2:
          raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

        cur_response = resp_parts[-1]

        cur_response = cur_response.split("### END ANSWER ###")[0]
        cur_response = cur_response.split("### END_ANSWER ###")[0]
        cur_response = cur_response.split("###END ANSWER###")[0]
        cur_response = cur_response.split("###END_ANSWER###")[0]
    
        if not cur_response.strip():
          raise ValueError("GPT gave an empty response")
    
        result.append(cur_response)
        break
      except (
      openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError, openai.error.InvalidRequestError,
      openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        print("Openai rate limit error {}. Sleeping for 5 seconds.".format(e))
        time.sleep(5)
      except ValueError as e:
        print(e)
        print("Retrying to summarize chunk")
      except Exception as e:
        if "That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at" in str(
                e) or "Bad gateway" in str(e):
          print("Openai weird overloaded exception {}, {}. Sleeping for 10 seconds.".format(e, type(e).__name__))
          time.sleep(10)
        else:
          raise e

  return result


def SummarizeSummariesDetailedEvents(chunk_summaries, max_length=10000):
  meta_chunks = rechunk_book(["Summary {}: {}\n".format(ind, summary) for ind, summary in enumerate(chunk_summaries)],
                             max_length)
  result = []

  for ind, m in enumerate(meta_chunks):

    if ind % 10 == 0:
      print("Processed {} meta chunks".format(ind + 1))

    for attempt in range(10):
      try:
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system",
             "content": "You are a helpful assistant that summarizes book scene summaries. Begin your answer with a {} tag.".format(
               BEGIN_ANSWER_TAG)},
            {"role": "user",
             "content": 'Summarize the following scene summaries into one plot summary: "{}". Make sure to list the key events and plot developments. Make sure to begin your answer with a {} tag!'.format(
               m, BEGIN_ANSWER_TAG)}  # Add "in under 500 words?
          ]
        )
        cur_response = response["choices"][0]["message"]["content"]

        if BEGIN_ANSWER_TAG not in cur_response:
          raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

        resp_parts = cur_response.split(BEGIN_ANSWER_TAG)

        if len(resp_parts) != 2:
          raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

        cur_response = resp_parts[-1]

        cur_response = cur_response.split("### END ANSWER ###")[0]
        cur_response = cur_response.split("### END_ANSWER ###")[0]
        cur_response = cur_response.split("###END ANSWER###")[0]
        cur_response = cur_response.split("###END_ANSWER###")[0]

        if not cur_response.strip():
          raise ValueError("GPT gave an empty response")

        result.append(cur_response)
        break
      except (
              openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError,
              openai.error.InvalidRequestError,
              openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        print("Openai rate limit error {}. Sleeping for 5 seconds.".format(e))
        time.sleep(5)
      except ValueError as e:
        print(e)
        print("Retrying to summarize chunk")
      except Exception as e:
        if "That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at" in str(
                e) or "Bad gateway" in str(e):
          print("Openai weird overloaded exception {}, {}. Sleeping for 10 seconds.".format(e, type(e).__name__))
          time.sleep(10)
        else:
          raise e

  return result


def SummarizeSummariesDetailedEvents2(chunk_summaries, max_length=10000):
  meta_chunks = rechunk_book(["Summary {}: {}\n".format(ind, summary) for ind, summary in enumerate(chunk_summaries)],
                             max_length)
  result = []

  for ind, m in enumerate(meta_chunks):

    if ind % 10 == 0:
      print("Processed {} meta chunks".format(ind + 1))

    for attempt in range(10):
      try:
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system",
             "content": "You are a helpful assistant that summarizes book scene summaries. Begin your answer with a {} tag.".format(
               BEGIN_ANSWER_TAG)},
            {"role": "user",
             "content": 'Describe the events in following scene summaries into one plot summary: "{}". Make sure to list the key events and plot developments. Make sure to begin your answer with a {} tag!'.format(
               m, BEGIN_ANSWER_TAG)}  # Add "in under 500 words?
          ]
        )
        cur_response = response["choices"][0]["message"]["content"]

        if BEGIN_ANSWER_TAG not in cur_response:
          raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

        resp_parts = cur_response.split(BEGIN_ANSWER_TAG)

        if len(resp_parts) != 2:
          raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

        cur_response = resp_parts[-1]

        cur_response = cur_response.split("### END ANSWER ###")[0]
        cur_response = cur_response.split("### END_ANSWER ###")[0]
        cur_response = cur_response.split("###END ANSWER###")[0]
        cur_response = cur_response.split("###END_ANSWER###")[0]

        if not cur_response.strip():
          raise ValueError("GPT gave an empty response")

        result.append(cur_response)
        break
      except (
              openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError,
              openai.error.InvalidRequestError,
              openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        print("Openai rate limit error {}. Sleeping for 5 seconds.".format(e))
        time.sleep(5)
      except ValueError as e:
        print(e)
        print("Retrying to summarize chunk")
      except Exception as e:
        if "That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at" in str(
                e) or "Bad gateway" in str(e):
          print("Openai weird overloaded exception {}, {}. Sleeping for 10 seconds.".format(e, type(e).__name__))
          time.sleep(10)
        else:
          raise e

  return result

def CreateFalseSummary(chunk):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant that changes book snippet summaries. Begin your answer with a {} tag.".format(BEGIN_ANSWER_TAG)},
      {"role": "user", "content": 'Take the summary below and rephrase it in such a way that the described events are no longer the same, even though the setting remains the same. Keep your summary to the same length. Start your answer with a "{}" tag. \nInital summary: \n"{}"'.format(BEGIN_ANSWER_TAG, chunk)} # Add "in under 500 words?
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  answer = answer.split("### END ANSWER ###")[0] ### GPT 3 sometimes randomly appends this, for whatever reason.

  if not answer.strip():
    raise ValueError("GPT gave an empty response")
  
  return answer.strip()


def CreateFalseHierSummary(chunk):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system",
       "content": "You are a helpful assistant that changes book snippet summaries. Begin your answer with a {} tag.".format(
         BEGIN_ANSWER_TAG)},
      {"role": "user",
       "content": 'Take the summary below and rephrase it in such a way that the described events are no longer the same, even though the setting remains the same. Keep your summary to the same length and keep your summary structure the same. Start your answer with a "{}" tag. \nInital summary: \n"{}"'.format(
         BEGIN_ANSWER_TAG, chunk)}  # Add "in under 500 words?
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  answer = answer.split("### END ANSWER ###")[0]  ### GPT 3 sometimes randomly appends this, for whatever reason.

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer.strip()


def DescribeChunkRoleFreeform(book_summary, scene_summary):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system",
       "content": "You are a helpful assistant that analyses how book snippets relate to the general book plot. Begin your answer with a {} tag.".format(BEGIN_ANSWER_TAG)},
      {"role": "user", "content": 'Book summary: {}\nScene summary: {}\nIs this scene crucial to the plot? How does it advance the plot?'.format(book_summary, scene_summary)}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer


def SimplifySummary(initial_summary):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system",
       "content": "You are a helpful assistant that changes book snippet summaries. Begin your answer with a {} tag.".format(
         BEGIN_ANSWER_TAG)},
      {"role": "user",
       "content": 'Paraphrase the summary below to have the same meaning but avoid rare words and fancy language. Begin your answer with a "{}" tag. \nInital summary: \n"{}"'.format(
         BEGIN_ANSWER_TAG, initial_summary)}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  answer = answer.split("### END ANSWER ###")[0]  ### GPT 3 sometimes randomly appends this, for whatever reason.

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer.strip()


def SimplifySummary(initial_summary):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system",
       "content": "You are a helpful assistant that changes book snippet summaries. Begin your answer with a {} tag.".format(
         BEGIN_ANSWER_TAG)},
      {"role": "user",
       "content": 'Paraphrase the summary below to have the same meaning but avoid rare words and fancy language. Begin your answer with a "{}" tag. \nInital summary: \n"{}"'.format(
         BEGIN_ANSWER_TAG, initial_summary)}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]

  if BEGIN_ANSWER_TAG not in response_content:
    raise ValueError("GPT response does not have a {} tag".format(BEGIN_ANSWER_TAG))

  resp_parts = response_content.split(BEGIN_ANSWER_TAG)

  if len(resp_parts) != 2:
    raise ValueError("GPT gave multiple {} tags".format(BEGIN_ANSWER_TAG))

  answer = resp_parts[-1]

  answer = answer.split("### END ANSWER ###")[0]  ### GPT 3 sometimes randomly appends this, for whatever reason.

  if not answer.strip():
    raise ValueError("GPT gave an empty response")

  return answer.strip()

