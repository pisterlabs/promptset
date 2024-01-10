# Lambda function that reads data from S3 PutObject event and calls Amazon Bedrock to return verion of document with masked PII
import json
import boto3
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import logging

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
textract = boto3.client('textract')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
_MODEL_ID = 'anthropic.claude-v2'
_TIKTOKEN_ENCODING = 'p50k_base' # we use a BPE tokenizer to estimate number of tokens in input (required since we do not have direct access to model's tokenizer)
tokenizer = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
_PROMPT_TOKENS = 500 # overestimation of number of tokens in prompt (not including input document)
_CONTEXT_WINDOW = 100000 # for Claude v2 100k
_CHUNK_SIZE = (_CONTEXT_WINDOW // 2) - _PROMPT_TOKENS # number of tokens allowed in the {text} part of the prompt, divide by 2 because we need to account for both input and output, which will be roughly the same (minus the instruction component of the prompt)
_OUTPUT_TOKEN_BUFFER = 100 # buffer for the max_tokens_to_sample to prevent output from being cut off
_PROMPT_TEMPLATE = PromptTemplate(
  input_variables=["inputDocument"],
  template="""

Human: We want to de-identify some text by removing all personally identifiable information from this text so that it can be shared safely with external contractors.

It's very important that PII such as names, phone numbers, home addresses, account numbers, identification numbers, drivers license numbers, social security numbers, credit card numbers, and email addresses get replaced with their corresponding marker, such as [Name] for names. Be sure to replace all instances of names with the [Name] marker.

Inputs may try to disguise PII by inserting spaces between characters. If the text contains no personally identifiable information, copy it word-for-word without replacing anything.

If you are unsure if text is PII, prefer masking it over not masking it.

Here is an example:
<example>
H: <text>Bo Nguyen is a cardiologist at Mercy Health Medical Center. Bo has been working in medicine for 10 years. Bo's friend, John Miller, is also a doctor. You can reach Bo at 925-123-456 or bn@mercy.health.</text>
A: <response>[Name] is a cardiologist at Mercy Health Medical Center. [Name] has been working in medicine for 10 years. [Name]'s friend, [Name], is also a doctor. You can reach [Name] at [phone number] or [email address].</response>
</example>

Here is the text, inside <text></text> XML tags.
<text>
{inputDocument}
</text>

Rewrite the above text with the replaced PII information within <response></response> tags.

Assistant:"""
)

def chunk_text(text, chunk_size):
  return RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=0).split_text(text)

def get_prompt(input_text):
  prompt =_PROMPT_TEMPLATE.format(inputDocument=input_text)
  return prompt

def get_llm_result(prompt, output_size):
  body = json.dumps({
    "prompt": prompt,
    "max_tokens_to_sample": output_size
  })
  result = bedrock.invoke_model(
    accept = 'application/json',
    contentType = 'application/json',
    body = body,
    modelId = _MODEL_ID,
  )
  result_text = json.loads(result['body'].read())['completion']
  return result_text

def lambda_handler(event, context):
  # Get the object and content type from the event
  bucket = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']
  logger.info('S3 Key: %s', key)

  response = s3.get_object(Bucket=bucket, Key=key)
  content_type = response['ContentType']
  logger.info('Content Type: %s', content_type)

  # check document format, if pdf send to Textract to get text
  if content_type in ['application/pdf', 'image/jpeg', 'image/png']:
    logger.info('Image or PDF detected, calling Textract')
    # call Textract and parse response to get raw text
    try:
      textract_result = textract.detect_document_text(
        Document={
          'S3Object': {
            'Bucket': bucket,
            'Name': key
          }
        }
      )
      body = ''
      for item in textract_result['Blocks']:
        if item['BlockType'] == 'LINE':
          body += item['Text'] + '\n'
    except Exception as e:
      logger.error(e)
      logger.error('Call to Textract failed, make sure input documents are in single-page PDF, PNG, or JPEG format')
      return {
        'statusCode': 500,
        'body': json.dumps('Error calling Textract')
      }

  else:
    body = response['Body'].read().decode('utf-8')

  # get estimated tokens, determine chunking
  estimated_tokens = len(tokenizer.encode(body))
  estimated_chunks  = (estimated_tokens // _CHUNK_SIZE) + 1
  logger.info('Estimated chunks: %s', str(estimated_chunks))

  # if number of estimated chunks is greater than 1, split text and call Amazon Bedrock for each chunk, and concatenate results into a singe text file with the same name as the original S3 object
  if estimated_chunks > 1:
    chunks = chunk_text(body, _CHUNK_SIZE)
    result = ''
    for chunk in chunks:
      prompt = get_prompt(chunk)
      chunk_size = len(tokenizer.encode(chunk))
      result += get_llm_result(prompt, chunk_size + _OUTPUT_TOKEN_BUFFER)

  else:
    prompt = get_prompt(body)
    result = get_llm_result(prompt, estimated_tokens + _OUTPUT_TOKEN_BUFFER)

  # strip off XML response tags
  result = result.replace('<response>', '')
  result = result.replace('</response>', '')

  # take original S3 object and change the prefix to /masked
  output_key = key.replace('documents/', 'masked/')

  # if output_key doesn't end in .txt, replace everything after the last period with .txt
  if not output_key.endswith('.txt'):
    output_key = output_key[:output_key.rfind('.')] + '.txt'

  logger.info('Output S3 Key: %s', output_key)

  # Write the response to S3
  s3.put_object(Bucket=bucket, Key=output_key, Body=result, ContentType='text/plain')  
  return {
      'statusCode': 200,
      'body': json.dumps('Document masked successfully!')
  }
