import json
import asyncio
import argparse
import logging
import coloredlogs
# from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm
from typing import List, Dict
from halpert import Halpert, Sample, Function
from halpert.util.openai import complete
from .samples import samples

ChatCompletionMessageParam = Dict

logger = logging.getLogger('halpert')


async def run_agent(
  sample: Sample,
  functions: List[Function],
  model: str,
) -> List[Sample.Evaluation.QuizItem]:
  messages: List[ChatCompletionMessageParam] = [{
    'role': 'system',
    'content': 'You are a helpful AI assistant. Follow the instructions and use the available functions to complete the task. Always call functions, and never respond with a text message! Do not make any assumptions about the task, and do not use any outside knowledge.',
  }, {
    'role': 'user',
    'content': sample.instructions,
  }]

  looping = True
  while looping:
    completion = complete(
      messages=messages,
      model=model,
      tools=[{
        'type': 'function',
        'function': {
          'name': f.slug,
          'description': f.description,
          'parameters': f.Input.schema(),
        },
      } for f in functions] + [{
        'type': 'function',
        'function': {
          'name': 'done',
          'description': 'Call this function when you are done with the task.',
          'parameters': { 'type': 'object', 'properties': {} },
        },
      }],
    )

    # logger.info(f'Agent Step: {completion.json(indent=2)}')
    logger.info(f'Agent Step: {json.dumps(completion, indent=2)}')

    choice = completion.choices[0]
    if choice.finish_reason != 'tool_calls':
      logger.warning(f'Unexpected finish reason: {choice.finish_reason}')
      break

    messages.append({
      'role': 'assistant',
      # 'tool_calls': choice.message.dict()['tool_calls'],
      'tool_calls': choice.message['tool_calls'],
    })

    for tc in choice.message.tool_calls:
      if tc.function.name == 'done':
        messages.pop()
        looping = False
        break
      elif fn := next((f for f in functions if f.slug == tc.function.name), None):
        output = await fn.call(fn.Input(**json.loads(tc.function.arguments)))
        messages.append({
          'role': 'tool',
          'tool_call_id': tc.id,
          'content': json.dumps(output.dict()),
        })

        logger.info(f'Function call: {fn.slug}({tc.function.arguments}) -> {json.dumps(output.dict(), indent=2)}')
      else:
        logger.warning(f'Unexpected function call: {tc.function.name}')
        looping = False
        break
  
  completion = complete(
    messages=[{
      'role': 'system',
      'content': 'You are a helpful AI assistant. Answer the questions based on the messages so far using the answer function. Question:\n' + '\n'.join([f'{i}. {q.question}' for i, q in enumerate(sample.expected.quiz)]),
    }] + messages[1:],
    tools=[{
      'type': 'function',
      'function': {
        'name': 'answer',
        'description': 'Call this function to answer all questions. If you do not know the answer to a specific question, enter an empty string. VERY IMPORTANT: answer all questions, even if you do not know the answer to some of them.',
        'parameters': {
          'type': 'object',
          'properties': {
            'num_questions': { 'type': 'integer' },
            'answers': {
              'type': 'array',
              'items': { 'type': 'string' },
            },
          },
          'required': ['answers'],
        },
      },
    }],
    model=model,
    tool_choice={ 'type': 'function', 'function': { 'name': 'answer' } },
  )

  # logger.info(f'Agent Questions: {completion.json(indent=2)}')
  logger.info(f'Agent Questions: {json.dumps(completion, indent=2)}')
  answers = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)['answers']

  return [
    Sample.Evaluation.QuizItem(question=q.question, answer=a)
    for q, a in zip(sample.expected.quiz, answers)
  ]


async def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
  parser.add_argument('--odoo-snapshot-dir', type=str)
  args = parser.parse_args()

  coloredlogs.install(fmt='%(levelname)s %(asctime)s %(name)s %(message)s', level=logging.DEBUG)
  logging.getLogger('openai').setLevel(logging.INFO)
  logging.getLogger('httpx').setLevel(logging.INFO)

  eval = Halpert(samples=samples, odoo_snapshot_dir=args.odoo_snapshot_dir)
  for sample in tqdm(eval.samples):
    sample_functions = eval.prepare(sample)
    logger.info(f'Running sample: {sample.name}')
    quiz = await run_agent(sample, sample_functions, args.model)
    logger.info(f'Quiz: {json.dumps([q.dict() for q in quiz], indent=2)}')
    eval.submit(sample, quiz)

  eval.evaluate()


if __name__ == '__main__':
  asyncio.run(run())
