import langchain
import vertexai
import texttospeech

def get_reminders():
  """Gets the user's reminders."""
  reminders = []
  while True:
    task = input("What reminder do you want to set up? ")
    if task == "":
      break
    preparations = get_preparations(task)
    reminders.append({
        "task": task,
        "preparations": preparations,
    })
  return reminders

def get_preparations(task):
  """Gets the preparations needed for a task."""
  preparations = []
  chain = langchain.Chain()
  chain.add_training_data([
      (task, ["Prepare gym equipment", "Get dressed"]),
      (task, ["Get my laptop ready", "Make coffee"]),
  ])
  preparations = chain.predict(task)
  return preparations

def set_reminders(reminders):
  """Sets the reminders."""
  for reminder in reminders:
    task = reminder["task"]
    preparations = reminder["preparations"]
    total_time_to_prepare = sum(preparations)
    time_to_remind = total_time_to_prepare - 5
    text_to_speak = "{} minutes for {}. Remember to {}. Please, check in.".format(
        time_to_remind, task, ", ".join(preparations))
    texttospeech.speak(text_to_speak)
    if not input("Have you prepared everything? (Y/N) ") == "Y":
      while time_to_remind > 0:
        texttospeech.speak("{} minutes left for {}.".format(time_to_remind, task))
        time_to_remind -= 1

def main():
  """Main function."""
  reminders = get_reminders()
  set_reminders(reminders)

if __name__ == "__main__":
  main()
