# Importe as classes LLMChain e SharedMemory do módulo langchain.chains
from langchain.chains import LLMChain, SharedMemory

# Importe a classe HuggingFacePipeline do módulo langchain
from langchain import HuggingFacePipeline

# Importe a função pipeline do módulo transformers
from transformers import pipeline

# Defina a classe SummaryChain como uma subclasse de LLMChain e sobrescreva os métodos __init__ e execute
class SummaryChain(LLMChain):
    def __init__(self, shared_memory: SharedMemory):
        # Crie uma pipeline usando transformers com um modelo de sumarização do Hugging Face
        generator = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Envolva a pipeline usando HuggingFacePipeline do LangChain
        llm = HuggingFacePipeline(pipeline=generator)
        
        # Inicialize a classe pai com um template de prompt e um modelo llm
        super().__init__(prompt_template="Texto: {input}\nResumo: ", model=llm)
        
        # Inicialize o atributo de memória compartilhada
        self.shared_memory = shared_memory

    def execute(self, input: str) -> str:
        # Chame o método execute da classe pai com o input e obtenha a saída
        output = super().execute(input)
        
        # Retorne a saída como uma string
        return str(output)

# Para usar a classe SummaryChain, você precisa criar uma instância dela e passá-la ao construtor ChainExecutor
# junto com outros parâmetros. Por exemplo:

# Crie uma instância SharedMemory para a conversa
shared_memory_conversation = SharedMemory(ConversationMemory(), "conversation")

# Crie uma instância SummaryChain com a memória compartilhada da conversa
summary_chain = SummaryChain(shared_memory_conversation)

# Crie uma instância ChainExecutor com summary_chain como parâmetro
chain_executor = ChainExecutor(chains=[summary_chain])
