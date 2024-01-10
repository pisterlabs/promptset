from langchain.embeddings import HuggingFaceInstructEmbeddings


def get_model_for(model_type: str, model_config: dict):
	if model_config is None:
		return None

	if model_type == 'embedding':
		return HuggingFaceInstructEmbeddings(**model_config)

	return None
