from transformers import AutoTokenizer
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 256,
        temperature = 0.3,
        config = {'context_length':5000}
    )

template = """
             Write a summary of the following text delimited by triple backtics.
             Return your response which covers the key points of the text.
             ```{text}```
             SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = """
LLM Model Serving : An Interesting Challenge

Text Generation Excerpt

Large Language Models (LLMs) generate text in a two-step process: pre-fill, where the tokens in the input prompt are processed in parallel, and decoding, where text is generated one token or word at a time in an autoregressive fashion i.e. model predicts future values based on past values. Each generated token is appended to the input and fed back into the model to generate the next token. Generation stops when the LLM outputs a special stop token or when a user-defined condition is met (e.g., some maximum number of tokens has been generated). If you'd like more background on how LLMs use decoder blocks. Tokens can be words or sub-words; the exact rules for splitting text into tokens vary from model to model.
Important Attributes
   How quickly users start seeing the model's output after entering their query. Low waiting times for a response are essential in real-time interactions. This metric is driven by the time required to process the prompt and then generate the first output token/ word.

   Time to generate an output token for each user that is querying our system. This metric corresponds with how each user will perceive the "speed" of the model. For example, a time per token/ word of 100 milliseconds/word would be 10 tokens per second per user, or  approximately 450 words per minute, which is faster than a typical person can read.

The overall time it takes for the model to generate the full response for a user. Overall response latency can be calculated using the previous two metrics: delay before a generation/ transfer of data begins = time to start generating first word + generate word for each user  x the number of words to be generated.

The number of output tokens/ words per second an inference server can generate across all users and requests.

Goal here is the model to generate text as fast as possible for as many users as model serving system can support.
In short, there is a tradeoff between throughput and time per output token: if one can process say 100 user queries concurrently, that will have higher throughput compared to running the queries sequentially, but we'll take longer to generate output tokens for each user. Before one anchor oneself to specific latency targets such as "less than 20 ms per word", one should spend some time characterizing an expected input and desired output lengths.
Challenges In Serving 
Optimizing LLM inference benefits from general techniques such as:
LLM modified by combining different adjacent operators together often results in less delay before a generation/ transfer of data begins, also known as operator fusion.

Activations and weights are compressed to use a smaller number of bits e.g. 4/ 8 bit quantized vs. floating point.

The LLM model compressed with reduced neural connection based on techniques such as sparsity or distillation.

Tensor parallelism across multiple machines or pipeline parallelism for larger models.

there are many important Transformer-specific optimizations. Such as key-value caching in vector databases. The attention mechanism in decoder-only Transformer-based models is computationally inefficient. key-value caching, i.e., saving of intermediate keys/values for the attention layers, is used to preserve those results for later reuse, avoiding repeated computation.
Back To White-Board
Computations in LLMs are mainly dominated by matrix multiplication operations; these operations with small dimensions are typically memory-bandwidth-bound on most hardware. When generating tokens/ words in an autoregressive manner as explained before, one of the activation matrix dimensions (defined by batch size and number of tokens/ words in the sequence) is small at small batch sizes. Therefore, the speed is dependent on how quickly we can load model parameters from GPU memory to local caches/registers, rather than how quickly we can compute on loaded data. Available and achieved memory bandwidth in inference hardware is a better predictor of speed of token generation than their peak compute performance.
hardware utilization is very important in terms of model serving costs. GPUs are very expensive and we need them to do as much work as possible. Shared inference services can be used to keep costs low by combining workloads from many users. That is by filling in individual gaps and batching together overlapping requests. For example for large models like Llama2-70B, we only achieve good cost/performance at large batch sizes. Having an inference serving system that can operate at large batch sizes is critical for cost efficiency. However, a large batch means larger KV cache size, and that in turn increases the number of GPUs required to serve the model. There's a tussle here and shared service operators need to make some cost trade-offs and implement systems optimizations.
Resource Utilization 
As explained above inference for LLMs at smaller batch size, especially at decode time—is bottlenecked on how quickly we can load model parameters from the device memory to the compute units. Memory bandwidth dictates how quickly the data movement happens. To measure the underlying hardware's utilization, how we do Model Bandwidth Utilization.It is defined as (achieved memory bandwidth) / (peak memory bandwidth) where achieved memory bandwidth depends on  ((total model parameter size + key-value cache size) / time per output word or token).
For example, if a 7B parameter running with 16-bit precision has time per output word equal to 14ms, then it's moving 14GB of parameters in 14ms translating to 1TB/sec bandwidth usage!! If the peak bandwidth of the machine is 2TB/sec, we are running at an model bandwidth utilization of 50%. For simplicity, this example ignores key-value cache size, which is small for smaller batch sizes and shorter sequence lengths. Model bandwidth utilization values close to 100% imply that the inference system is effectively utilizing the available memory bandwidth. Model bandwidth utilization is also useful to compare different inference systems (hardware + software) in a normalized manner.
Throughput
We can trade off throughput and time per token by batching requests together. Grouping queries during GPU evaluation increases throughput compared to processing queries sequentially, but each query will take longer to complete (ignoring queueing effects).
There are a few common techniques for batching inference requests:
In static batching, client packs multiple prompts into requests and a response is returned after all sequences in the batch have been completed. The inference servers support this but do not require it.

In case of dynamic batching, user prompts are batched together on the fly inside the server. Typically, this method performs worse than static batching but can get close to optimal if responses are short or of uniform length. Does not work well when requests have different parameters

In continuous batching, the idea of batching requests together as they arrive was introduced Instead of waiting for all sequences in a batch to finish, it groups sequences together at the iteration level. It can achieve 10x-20x better throughput than dynamic batching.

In summary the continuous batching is usually the best approach for an enterprise with shared services, but there are situations where the other two might be better. In low queries per second environments, dynamic batching can outperform continuous batching. It is sometimes easier to implement low-level GPU optimizations in a simpler batching framework. For offline batch inference workloads, static batching can avoid significant overhead and achieve better throughput.
Tradeoff
Request latency increases with batch size. With one NVIDIA A100 GPU, for example, if we maximize throughput with a batch size of 64, latency increases by 4x while throughput increases by 14x. Shared inference services typically pick a balanced batch size. Users hosting their own models should decide the appropriate latency/throughput trade-off for their applications. In some applications, like chatbots, low latency for fast responses is the top priority. In other applications, like batched processing of unstructured PDFs, we might want to sacrifice the latency to process an individual document to process all of them fast in parallel.
"""

print(llm_chain.run(text))