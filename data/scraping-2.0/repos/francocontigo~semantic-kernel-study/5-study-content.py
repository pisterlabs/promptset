import semantic_kernel as sk
kernel = sk.Kernel()

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-4", api_key, org_id))

study_content_skill = kernel.import_semantic_skill_from_directory("plugins", "study")
study_content_function = study_content_skill["StudyContent"]

print(study_content_function("Banco de Dados e Pintura Corporal"))

