from typing import Any, List, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.vertexai import VertexAIModelGarden
from langchain.schema.output import Generation, GenerationChunk, LLMResult


class VertexAIModelGardenPeft(VertexAIModelGarden):
    """
    A class representing large language models served from Vertex AI Model Garden using Peft
    PyTorch runtime such as LLaMa2.

    Attributes:
        max_length (int): Token limit determines the maximum amount of text output from one prompt.
        top_k (int): How the model selects tokens for output, the next token is selected from
            among the top-k most probable tokens. Top-k is ignored for Code models.
    """

    max_length: int = 200
    top_k: int = 40

    def __init__(self, **kwargs):
        """
        Initialize the VertexAIModelGardenPeft instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the super class.
        """
        super().__init__(
            allowed_model_args=[
                "max_length",
                "top_k",
            ],
            **kwargs,
        )

    def _generate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = super()._generate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            **kwargs,
        )
        generations: List[List[GenerationChunk]] = []
        for prompt, result in zip(prompts, result.generations):
            chunks = [GenerationChunk(text=prediction.text) for prediction in result]
            chunks = self._strip_generation_context(prompt, chunks)
            generation = self._aggregate_response(
                chunks,
                run_manager=run_manager,
                verbose=self.verbose,
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = await super()._agenerate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            **kwargs,
        )
        generations: List[List[GenerationChunk]] = []
        for prompt, result in zip(prompts, result.generations):
            chunks = [GenerationChunk(text=prediction.text) for prediction in result]
            chunks = self._strip_generation_context(prompt, chunks)
            generation = self._aggregate_response(
                chunks,
                run_manager=run_manager,
                verbose=self.verbose,
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    def _aggregate_response(
        self,
        chunks: List[Generation],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for chunk in chunks:
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("Malformed response from VertexAIModelGarden")
        return final_chunk

    def _strip_generation_context(
        self,
        prompt: str,
        chunks: List[GenerationChunk],
    ) -> List[GenerationChunk]:
        context = self._format_generation_context(prompt)
        chunk_cursor = 0
        context_cursor = 0
        while chunk_cursor < len(chunks) and context_cursor < len(context):
            chunk = chunks[chunk_cursor]
            for c in chunk.text:
                if c == context[context_cursor]:
                    context_cursor += 1
                else:
                    break
            chunk_cursor += 1
        return chunks[chunk_cursor:] if chunk_cursor == context_cursor else chunks

    def _format_generation_context(self, prompt: str) -> str:
        return "\n".join(["Prompt:", prompt.strip(), "Output:", prompt])


# TODO: Support VLLM streaming inference
class VertexAIModelGardenVllm(VertexAIModelGarden):
    """
    A class representing large language models served from Vertex AI Model Garden using VLLM
    PyTorch runtime such as Mistral AI.

    Attributes:
        top_k (int): How the model selects tokens for output, the next token is selected from
            among the top-k most probable tokens. Top-k is ignored for Code models.
        n (int): Number of output sequences to return for the given prompt.
        best_of (int): Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty (float): Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty (float): Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature (float): Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p (float): Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        use_beam_search (bool): Whether to use beam search instead of sampling.
        length_penalty (float): Float that penalizes sequences based on their length.
            Used in beam search.
        stop_token_ids (Optional[List[int]]): List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        ignore_eos (bool): Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens (int): Maximum number of tokens to generate per output sequence.
        logprobs (Optional[int]): Number of log probabilities to return per output token.
            Note that the implementation follows the OpenAI API: The return
            result includes the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens. The API will always return the
            log probability of the sampled token, so there  may be up to
            `logprobs+1` elements in the response.
    """

    top_k: int = 40
    n: int = 1
    best_of: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: int = 200
    logprobs: Optional[int] = None

    def __init__(self, **kwargs):
        """
        Initialize the VertexAIModelGardenVllm instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the super class.
        """
        super().__init__(
            allowed_model_args=[
                "top_k",
                "n",
                "best_of",
                "presence_penalty",
                "frequency_penalty",
                "temperature",
                "top_p",
                "top_k",
                "use_beam_search",
                "length_penalty",
                "stop",
                "stop_token_ids",
                "ignore_eos",
                "max_tokens",
                "logprobs",
            ],
            **kwargs,
        )

    def _generate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = super()._generate(
            prompts=prompts,
            top_k=self.top_k,
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            use_beam_search=self.use_beam_search,
            length_penalty=self.length_penalty,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            **kwargs,
        )
        generations: List[List[GenerationChunk]] = []
        for prompt, result in zip(prompts, result.generations):
            chunks = [GenerationChunk(text=prediction.text) for prediction in result]
            chunks = self._strip_generation_context(prompt, chunks)
            generation = self._aggregate_response(
                chunks,
                run_manager=run_manager,
                verbose=self.verbose,
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = await super()._agenerate(
            prompts=prompts,
            top_k=self.top_k,
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            use_beam_search=self.use_beam_search,
            length_penalty=self.length_penalty,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            **kwargs,
        )
        generations: List[List[GenerationChunk]] = []
        for prompt, result in zip(prompts, result.generations):
            chunks = [GenerationChunk(text=prediction.text) for prediction in result]
            chunks = self._strip_generation_context(prompt, chunks)
            generation = self._aggregate_response(
                chunks,
                run_manager=run_manager,
                verbose=self.verbose,
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    def _aggregate_response(
        self,
        chunks: List[Generation],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for chunk in chunks:
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("Malformed response from VertexAIModelGarden")
        return final_chunk

    def _strip_generation_context(
        self,
        prompt: str,
        chunks: List[GenerationChunk],
    ) -> List[GenerationChunk]:
        context = self._format_generation_context(prompt)
        chunk_cursor = 0
        context_cursor = 0
        while chunk_cursor < len(chunks) and context_cursor < len(context):
            chunk = chunks[chunk_cursor]
            for c in chunk.text:
                if c == context[context_cursor]:
                    context_cursor += 1
                else:
                    break
            chunk_cursor += 1
        return chunks[chunk_cursor:] if chunk_cursor == context_cursor else chunks

    def _format_generation_context(self, prompt: str) -> str:
        return "\n".join(["Prompt:", prompt.strip(), "Output:", ""])
