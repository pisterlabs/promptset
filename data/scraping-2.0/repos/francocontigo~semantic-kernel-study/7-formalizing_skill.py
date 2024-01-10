import semantic_kernel as sk
kernel = sk.Kernel()

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-4", api_key, org_id))

formalizing_skill = kernel.import_semantic_skill_from_directory("plugins", "companion")
formalizing_function = formalizing_skill["FormalizingSpeech"]

print(formalizing_function("Não vou a festar por estar com muita preguiça de você"))

