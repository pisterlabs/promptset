from omegaconf import OmegaConf
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceHubEmbeddings

class EncoderFactory:
    """
    Factory class to initialize and provide encoder models based on configuration.

    Methods
    -------
    get_encoder(config: OmegaConf) -> Any:
        Returns the initialized encoder model based on the provided configuration.
    """
    @staticmethod
    def _init_openai_embeddings(config: OmegaConf) -> OpenAIEmbeddings:
        """Get langchain OpenAIEmbeddings wrapper for OpenAI API."""
        return OpenAIEmbeddings(**config.encoder.args)

    @staticmethod
    def _init_huggingface_embeddings(config: OmegaConf) -> HuggingFaceEmbeddings:
        """Get langchain HuggingFaceEmbeddings wrapper for Huggingface Embeddings Models."""
        repo_id = f"{config.encoder.user}/{config.encoder.name}"
        return HuggingFaceEmbeddings(
            model_name=repo_id, 
            cache_folder=config.encoder.cache_folder
        )

    @staticmethod
    def _init_huggingface_instruct_embeddings(config: OmegaConf) -> HuggingFaceInstructEmbeddings:
        """Get langchain HuggingFaceInstructEmbeddings wrapper for InstructorEmbedding Models."""
        instructor_args = {}
        if config.encoder.args.embed_instruction is not None:
            instructor_args['embed_instruction'] = config.encoder.args.embed_instruction
        if config.encoder.args.query_instruction is not None:
            instructor_args['query_instruction'] = config.encoder.args.query_instruction
        repo_id = f"{config.encoder.user}/{config.encoder.name}"
        return HuggingFaceInstructEmbeddings(
            model_name=repo_id, 
            **config.encoder.args
        )

    @staticmethod
    def _init_huggingface_hub_embeddings(config: OmegaConf) -> HuggingFaceHubEmbeddings:
        """Get langchain HuggingFaceHubEmbeddings wrapper for Huggingface free Inference API."""
        repo_id = f"{config.encoder.user}/{config.encoder.name}"
        return HuggingFaceHubEmbeddings(
            repo_id=repo_id, 
            **config.encoder.args
        )

    @staticmethod
    def get_encoder(config: OmegaConf):
        """
        Factory method to get the initialized encoder model based on the provided configuration.

        Parameters
        ----------
        config : OmegaConf
            Configuration specifying the type of encoder and its initialization parameters.

        Returns
        -------
        Any
            An initialized encoder model. The exact type depends on the configuration.

        Raises
        ------
        ValueError
            If the encoder type specified in the configuration is not supported.
        """
        encoder_initializers = {
            "openai": EncoderFactory._init_openai_embeddings,
            "huggingfaceembeddings": EncoderFactory._init_huggingface_embeddings,
            "huggingfaceinstructembeddings": EncoderFactory._init_huggingface_instruct_embeddings,
            "huggingfacehubembeddings": EncoderFactory._init_huggingface_hub_embeddings,
        }

        encoder_type = config.encoder.type.lower()
        if encoder_type not in encoder_initializers:
            raise ValueError(f"Unknown encoder type: {config.encoder.type}")

        return encoder_initializers[encoder_type](config)
