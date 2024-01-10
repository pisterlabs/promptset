# tqdm library makes the loops show a smart progress meter
from tqdm import tqdm
import yaml

from code_generator import openai_util
from code_generator.model.completion import EvaluatedCompletion
from code_generator.model.context import Context
from code_generator.openai_util import pretty_print_conversation
from code_generator.prompt_generator import kickstart_prompt, decompose_prompt

tqdm.pandas()


def highlevel_processing(context: Context):
  """
  Iterates conversation steps with model till user high-level prompt is
  completed

  Args:
    context(Context): the current context
  """
  decompose(context, 3)
  # kickstart(context, 10)
  # low_level_task_processing(context, 10)

def decompose(context: Context, hard_stop_iteration_limit: int):
  # Initialize the progress bar
  progress_bar = tqdm(total=hard_stop_iteration_limit)
  context.current_status = "Task decomposition..."

  context.iteration_count = 0
  while (not context.goal_reached and
         context.iteration_count < hard_stop_iteration_limit):
    update_progress_bar_state(context, progress_bar)

    prompt = decompose_prompt(context)
    response = complete(context, prompt)
    progress_bar.update(1)


    # Update the progress bar
  if context.goal_reached:
    print(f"Goal reached in {context.iteration_count} steps")
  else:
    print("Goal not reached; hard stopped")


# def kickstart(context: Context, hard_stop_iteration_limit: int):
#   # Initialize the progress bar
#   progress_bar = tqdm(total=hard_stop_iteration_limit)
#   context.current_status = "Kickstart..."


  # Start the loop
  # while (not context.goal_reached and
  #        context.iteration_count < hard_stop_iteration_limit):
  #   # Update the progress bar description with current status
  #   progress_bar.set_description(f"Status: {context.current_status}")
  #
  #   # Update the progress bar with additional information
  #   progress_bar.set_postfix(
  #     PromptTUsed=context.prompt_tokens_used,
  #     CompletTUsed=context.completion_tokens_used,
  #     TotalTUsed=context.prompt_tokens_used + context.completion_tokens_used
  #   )
  #
  #   perform_step(context)
  #
  #   # Update the progress bar
  #   progress_bar.update(1)
  # if context.goal_reached:
  #   print(f"Goal reached in {context.iteration_count} steps")
  # else:
  #   print("Goal not reached; hard stopped")


# def low_level_task_processing(context: Context, hard_stop_iteration_limit: int):
#   """
#   :param context:
#   :param hard_stop_iteration_limit:  How many iterations can pass before
#   iteration is forcefully stopped
#   :return: true
#   """
#   # Initialize the progress bar
#   progress_bar = tqdm(total=hard_stop_iteration_limit)
#   # Start the loop
#   while (not context.goal_reached and
#          context.iteration_count < hard_stop_iteration_limit):
#     update_progress_bar_state(context, progress_bar)
#
#     perform_step(context)
#     progress_bar.update(1)
#
#     # Update the progress bar
#   if context.goal_reached:
#     print(f"Goal reached in {context.iteration_count} steps")
#   else:
#     print("Goal not reached; hard stopped")


def update_progress_bar_state(context, progress_bar):
  # Update the progress bar description with current status
  progress_bar.set_description(f"Status: {context.current_status}")
  # Update the progress bar with additional information
  progress_bar.set_postfix(
    PromptTUsed=context.prompt_tokens_used,
    CompletTUsed=context.completion_tokens_used,
    TotalTUsed=context.prompt_tokens_used + context.completion_tokens_used
  )

# def perform_step(context: Context):
#
#   if context.conversation.history[-1].completion_has_issues():
#     prompt =
#   if context.iteration_count == 0:
#     prompt = kickstart_prompt(context)
#   else:
#     prompt = None # TODO: update
#     context.current_status = "Coding..."
#
#   response = complete(context, prompt)
#
#   try:
#     parsed_response = yaml.safe_load(response)
#   except yaml.YAMLError as e:
#     # Handle the YAML parsing error here
#     print(f"Error parsing YAML: {e}")
#     context.write_down_completion_issue(str(e))
#     parsed_response = None  # You can set parsed_response to a default value or handle it accordingly
#
#   if parsed_response:
#     print(parsed_response)
#     # TODO: parse actions
#     pass


def complete(context, prompt):
  print(f"Prompt: {prompt}")
  pretty_print_conversation(prompt.to_messages())

  completion_response = openai_util.get_completion(prompt)

  context.prompt_tokens_used += (
    completion_response)['usage']["prompt_tokens"]
  context.completion_tokens_used += (
    completion_response)['usage']["completion_tokens"]
  context.total_tokens_used += (
    completion_response)['usage']["total_tokens"]
  if (len(completion_response['choices'])) > 1:
    print("!!! MULTIPLE CHOICES !!!")

  completion_for_evaluation = EvaluatedCompletion(completion_response)
  context.write_down(prompt, completion_for_evaluation)
  pretty_print_conversation(completion_response.to_messages())
  return completion_for_evaluation

