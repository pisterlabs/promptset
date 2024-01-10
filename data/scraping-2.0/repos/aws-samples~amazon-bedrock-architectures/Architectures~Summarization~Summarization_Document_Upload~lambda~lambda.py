import json
import logging
import boto3
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import tiktoken

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
_OUTPUT_TOKEN_BUFFER = 100 # buffer for the max_tokens_to_sample to prevent output from being cut off
_MAX_SUMMARY_LENGTH = 300
_MAX_INPUT_SIZE = _CONTEXT_WINDOW - _PROMPT_TOKENS - _MAX_SUMMARY_LENGTH - _OUTPUT_TOKEN_BUFFER
_EXAMPLE_TEXT = """
H: <text>
Q. Sam, you have been through games like Louisville last week, but was it any different here to come back from the adversity with USC coming in, and just how did you sort of manage that over the last seven days?
SAM HARTMAN: Yeah, they're wild out there. It's awesome. I think it's like what Coach Freeman just said, we are a reflection of our head coach. It's been a bumpy season. You know, you start hot and you lose a close one to Ohio State, and so it's one of those things where, like you said, being in those situations before prepares you for the ridicule, the feeling, the pit in your stomach.
But like coach said, it was a really special week. I saw this thing, John Jones, I don't condone everything he does, but saw a thing where he talks about pre-fight. Talks about the butterflies are in formation when you get butterflies or a pit in your stomach. Not to say that that's some crazy cool message that's going to end up on some cool highlight, but it's what I felt the team felt.
I felt like all week we as a unit, and again, obviously on defense, one of the best defensive performances I've ever seen against one of the nest offenses in the country. Really just the mentality, the work, preparation, all kind of just aligned perfectly.
It's a credit to our head coach. You know, without his leadership and guidance through a new landscape where losses aren't acceptable, and not to say that other losses are, but it was something that just from day one, from Monday night when we were in there and guys are beat up and we're all kind of like, shoot, man, we got to go.
It was something we said all week, too, is what better opportunity than to come have USC come play at home. We got really good weather and I think we did exactly what we wanted to do all week, and it's a moment that I'll never forget. I hope there our fans out there that will never forget, and forever I can say when I came here and played USC we won and I'm 1-0.
Q. The TD to Tyree, just take me through the look, how that play developed.
SAM HARTMAN: Yeah, all week we kind of had a bead on some of their coverages of what they might run to certain formations, and got one there. It's a credit to him. I'm so glad you asked about CT. You don't see that anywhere really ever. You know, older guy like that. His persistence, who he is as a man will take him so much further than anyone can ever know, and that's something that I'll always be forever grateful for him.
To be an older guy and have some struggles and have to change positions, like that itself, you know, and he's had some bad stuff and some drops and some things you're like, oh, man you got to make that play. He just kept showing up.
You bring Faison in, an incredible player, and you're like, most guys, probably, you know, I'm going to take a step. Chris kept showing up. Chris helped Faison and Faison helped him get open on that play. It's a credit to the coaching after, the culture here, and just to Chris as a man. You don't find that everywhere, and I was so happy.
The first thing we said is, I told you to each other, because all week we been talking about it. It's going to come, it's going to happen, you're going to get that opportunity, and he was freaking wide open, so made it pretty easy for me.
Q. How critical was the touchdown to open the game after you guys get the turnover? Third down conversion, but also a touchdown to give you guys some confidence moving forward.
SAM HARTMAN: It's everything. We wanted to start fast. That's kind of our big three of this week. We really look back at the games that we played well in, and it was like a first-drive touchdown was huge. We kind of knew their offense was going to be able to score. Honestly our defense really shut them down and gave them a lot of frustrations.
It was big in the beginning of the game, but our defense kept us just on a groove and really good field position.
So like they say, defense wins championships. I guess I got to buy Al Golden another Ferrari. They ain't paying me enough to do that.
Q. What conversations did you have with the offensive line after a bit of a rough go last week, to come back out tonight and have a different performance?
SAM HARTMAN: I think it's all culture. I don't think anything I was going to say was going to change the way they showed up. I don't think it's anything to do with mindset, skill, anything.
Just Louisville is really good team, and they lost, so takes away that bluster. Did they?
Q. Yeah.
SAM HARTMAN: Okay, it starts with Zeke and the guys on the outside. They really set the tone, and Zeke was the first one in the building and he was the first one getting everyone motivated, and just kept saying all week, take your mind there. That was probably the best thing anyone could say. You're in the dog days of it, it's Tuesday, eight-week bender and we're rolling.
To get guys fricking going like we did this Saturday was incredible. They set tone for the rest of the season that you got to play Notre Dame.
Q. You kind of touched on this before, but you have been through wins and losses in your career. What's it like to go through a loss and then a win, the low and the high, like at this school?
SAM HARTMAN: I mean, it's incredible. We don't like saying I around here and you guys rarely hear it, but it's special for me. I hugged Coach Freeman after and I was like, I finally did it. Really our defense did it. I mean, I threw the ball is couple times.
But it's, again, kind of like I led off with, it'll be a special moment for me and I'm excited to get back in the locker room. Just to see the fans and the support we get continuously and the walk and just the football culture here, I hope it never changes. If I'm blessed to have kids I hope I can bring them back and they play a highlight. Probably won't be as cool as Joe Montana, but -- you know, I met Joe Montana today. That was pretty sweet. Probably add that, USC victory, Joe Montana.
And I think that that is going to be something I can kind of cherish for the rest of my life. And the memories with those guys in the locker room, to bring it back to the team, just Cam Hart, you see that guy and his disappointment and his frustrations of last week, and to see him bounce back and just see the entire team, just it was a full, complete game. I'm so grateful for that. Grateful for this fan base and everything.
I freaking love the Irish.
Q. Some joking aside, USC is above meeting Joe Montana. You've been in college football for a while now. What did you pick up this week, tonight, about this particular game, matchup up at Notre Dame?
SAM HARTMAN: I mean, all week it was just something for me that was new. I know going we had an Irish immersion program in California. I got to meet some people. Got to meet Jimmy Clausen. Everybody talked about it out there, right, beat USC, beat USC. It was, beat Ohio State before that.
It was everybody you meet walking to the airport, grabbing groceries at Trader Joe's, beat USC. We brought back the trophy and regained that, and it's been a tradition to win at home and that's something I'm really proud of and proud of this team for rallying around it.
I know the magnitude. I'll be training in California for the pro stuff and I'll have that little kind of -- I can walk a little bit higher and taller out there.
</text>
A: <summary>
Quarterback Sam Hartman discussed Notre Dame's comeback win against USC. He said the adversity after losing to Louisville challenged the team, but coach Marcus Freeman's leadership guided them through. Hartman said the team had a \"special week\" preparing for USC, one of the best offenses in the country. He credited the historic defensive performance that shut down USC. Offensively, Hartman discussed the importance of scoring a touchdown on the first drive to build confidence. He praised receiver Chris Tyree for his persistence and growth this season despite struggles. Their connection on a touchdown pass exemplified Tyree's hard work. Hartman also credited the offensive line for bouncing back after a rough game against Louisville. He said the culture and leadership of players like center Zeke Correll set the tone in practice to be ready for USC. Personally, Hartman called the USC win a special moment in his career that he'll cherish. He said the fan support and football culture at Notre Dame are incredible. Beating a storied rival like USC will stay with him forever.
</summary>
"""
_MAP_PROMPT_TEMPLATE = PromptTemplate(
  input_variables=["text", "example"],
  template="""

Human: Given some text, we want to distill the text into a summary of the main themes.

Write your summary within <summary></summary> tags.

Here is an example:
<example>
{example}
</example>

Here is the text, inside  <text></text> XML tags.
<text>
{text}
</text>

Write a concise summary of the above text.

Assistant:"""
)
_COMBINE_PROMPT_TEMPLATE = PromptTemplate(
  input_variables=["text", "example"],
  template="""

Human: Given a set of summaries, we want to distill them into a final, consolidated summary of the main themes.

Write your summary within <summary></summary> tags.

Here is an example:
<example>
{example}
</example>

Here is the text, inside <text></text> XML tags.

<text>
{text}
</text>

Write a <300 word concise summary of the above text within <summary></summary> tags.

Assistant:"""
)
_STUFF_PROMPT_TEMPLATE = PromptTemplate(
  input_variables=["text", "example"],
  template="""

Human: Given some text, we want to create a concise summary of the main themes.

Write your summary within <summary></summary> tags.

Here is an example:
<example>
{example}
</example>

Here is the text, inside <text></text> XML tags.
<text>
{text}
</text>

Write a <300 word concise summary of the above text within <summary></summary> tags.

Assistant:"""
)

def chunk_text(text, chunk_size):
  return RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=int(chunk_size / 100)).create_documents([text])

def estimate_num_chunks(text):
  estimated_tokens = len(tokenizer.encode(text))
  return (estimated_tokens // _MAX_INPUT_SIZE) + 1

def process_llm_output(text_output):
  # strip off XML response tags
  text_output = text_output.replace('<summary>', '')
  text_output = text_output.replace('</summary>', '')
  return text_output

def get_summary_short_doc(text_chunks, output_size):
  llm = Bedrock(
    model_id = _MODEL_ID,
    model_kwargs = {
      "max_tokens_to_sample": output_size  
    }
  )
  llm_chain = LLMChain(llm=llm, prompt=_STUFF_PROMPT_TEMPLATE)
  # Define StuffDocumentsChain
  stuff_chain = StuffDocumentsChain(
      llm_chain=llm_chain, document_variable_name="text", verbose = True
  )
  result = stuff_chain.run(input_documents = text_chunks, example = _EXAMPLE_TEXT)
  return process_llm_output(result)

def get_summary_large_doc(text_chunks, output_size):
  llm = Bedrock(
    model_id = _MODEL_ID,
    model_kwargs = {
      "max_tokens_to_sample": output_size  
    }
  )
  summary_chain = load_summarize_chain(
    llm=llm, 
    chain_type="map_reduce", 
    map_prompt=_MAP_PROMPT_TEMPLATE,
    combine_prompt=_COMBINE_PROMPT_TEMPLATE,
    verbose = True
  )
  result = summary_chain.run(input_documents = text_chunks, example = _EXAMPLE_TEXT)
  return process_llm_output(result)

# this Lambda function is invoked through API Gateway
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

  chunks = chunk_text(body, _MAX_INPUT_SIZE)
  logger.info('Estimated chunks: %s', str(len(chunks)))

  if len(chunks) > 1:
    result = get_summary_large_doc(chunks, _MAX_SUMMARY_LENGTH + _OUTPUT_TOKEN_BUFFER)
  else: 
    result = get_summary_short_doc(chunks, _MAX_SUMMARY_LENGTH + _OUTPUT_TOKEN_BUFFER)

  # take original S3 object and change the prefix to /masked
  output_key = key.replace('documents/', 'summaries/')

  # if output_key doesn't end in .txt, replace everything after the last period with .txt
  if not output_key.endswith('.txt'):
    output_key = output_key[:output_key.rfind('.')] + '.txt'

  logger.info('Output S3 Key: %s', output_key)

  # Write the response to S3
  s3.put_object(Bucket=bucket, Key=output_key, Body=result, ContentType='text/plain')  
  return {
      'statusCode': 200,
      'body': json.dumps('Summary created successfully!')
  }
