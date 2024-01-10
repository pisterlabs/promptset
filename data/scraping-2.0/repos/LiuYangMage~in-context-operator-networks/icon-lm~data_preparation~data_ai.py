import data_io as io
from absl import app, flags, logging
import openai

def generate_captions_ai(prompt, model, api_key):
  print('-'*50, 'prompt begin', '-'*50)
  print(prompt)
  print('-'*50, 'prompt end', '-'*50)

  openai.api_key = api_key
  completion = openai.ChatCompletion.create(
                model= model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

  content = completion.choices[0].message.content
  return content


def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  if FLAGS.prompt_version == 'std':
    for prob in FLAGS.problem:
      prompt = io.read_whole_file('prompts/{}.md'.format(prob)) + '\n' + io.read_whole_file('prompts/shared.md')
      captions = generate_captions_ai(prompt, FLAGS.ai_model, FLAGS.api_key)
      io.write_whole_file('captions/{}.md'.format(prob), captions)
  elif FLAGS.prompt_version == '1007':
    for prob in FLAGS.problem:
      if prob in ['mfc_gparam', 'mfc_rhoparam']:
        prompt = io.read_whole_file('prompts_1007/{}.md'.format(prob))
      elif prob in ['ode1', 'ode2', 'ode3']:
        prompt = io.read_whole_file('prompts_1007/{}.md'.format(prob)) + '\n' + io.read_whole_file('prompts_1007/shared_ode.md')
      elif prob in ['pde1', 'pde2', 'pde3']:
        prompt = io.read_whole_file('prompts_1007/{}.md'.format(prob)) + '\n' + io.read_whole_file('prompts_1007/shared_pde.md')
      elif prob in ['series']:
        prompt = io.read_whole_file('prompts_1007/{}.md'.format(prob)) + '\n' + io.read_whole_file('prompts_1007/shared_series.md')
      else:
        raise ValueError('problem not found')
      captions = generate_captions_ai(prompt, FLAGS.ai_model, FLAGS.api_key)
      io.write_whole_file('captions_1009g/{}.md'.format(prob), captions)
  

  
if __name__ == "__main__":


  FLAGS = flags.FLAGS

  flags.DEFINE_list('problem', ['ode1', 'ode2', 'ode3', 'pde1', 'pde2', 'pde3', 'series', 'mfc_gparam', 'mfc_rhoparam'], 'problem to process')
  flags.DEFINE_string('api_key', None, 'the api key for the AI')
  flags.DEFINE_enum('ai_model', 'gpt-4', ['gpt-3.5-turbo','gpt-4'], 'the AI model')
  flags.DEFINE_string('prompt_version', '1007', 'prompt version')

  app.run(main)