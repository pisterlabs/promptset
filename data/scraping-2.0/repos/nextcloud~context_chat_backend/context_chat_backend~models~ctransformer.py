from langchain.llms.ctransformers import CTransformers


def get_model_for(model_type: str, model_config: dict):
	if model_config is None:
		return None

	if model_type == 'llm':
		return CTransformers(**model_config)

	return None
