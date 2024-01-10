import semantic_kernel as sk
kernel = sk.Kernel()

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-4", api_key, org_id))

routine_skill = kernel.import_semantic_skill_from_directory("plugins", "companion")
routine_function = routine_skill["DailyRoutine"]

print(routine_function("correr, limpar o quarto, formatar o notebook, revisar exercicios de estatística, arrumar a cozinha, estudar python"))

