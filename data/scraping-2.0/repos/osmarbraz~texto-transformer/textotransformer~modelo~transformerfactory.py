# Import das bibliotecas.

# Biblioteca de logging
import logging  

# Biblioteca de logging
import logging  
# Biblioteca de tipos
from typing import Dict, Optional
# Biblioteca do transformer hunggingface
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import AlbertModel, AlbertForMaskedLM
from transformers import BertModel, BertForMaskedLM
from transformers import DistilBertModel, DistilBertForMaskedLM
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import OpenAIGPTModel, OpenAIGPTConfig, OpenAIGPTLMHeadModel 
from transformers import RobertaModel, RobertaForMaskedLM
from transformers import XLMRobertaModel, XLMRobertaForMaskedLM
from transformers import XLNetModel, XLNetConfig, XLNetLMHeadModel
from transformers import T5EncoderModel, T5Config, T5ForConditionalGeneration

# Bibliotecas próprias
from textotransformer.modelo.modeloargumentos import ModeloArgumentos
from textotransformer.modelo.transformeralbert import TransformerAlbert
from textotransformer.modelo.transformerbert import TransformerBert
from textotransformer.modelo.transformerdistilbert import TransformerDistilbert
from textotransformer.modelo.transformergpt2 import TransformerGPT2
from textotransformer.modelo.transformeropenaigpt import TransformerOpenAIGPT
from textotransformer.modelo.transformerroberta import TransformerRoberta
from textotransformer.modelo.transformerxlmroberta import TransformerXLMRoberta
from textotransformer.modelo.transformerxlnet import TransformerXLNet
from textotransformer.modelo.transformert5 import TransformerT5

# Objeto de logger
logger = logging.getLogger(__name__)

class TransformerFactory():
    '''
    Classe construtora de objetos Transformer de Texto-Transformer.
    Retorna um objeto Transformer de acordo com auto_model dos parâmetros. 
    '''
    
    @staticmethod
    def getTransformer(tipo_modelo_pretreinado: str = "simples",
                       modelo_args: ModeloArgumentos = None,
                       cache_dir: Optional[str] = None,
                       tokenizer_args: Dict = {}, 
                       tokenizer_name_or_path: str = None):
        ''' 
        Retorna um objeto Transformer de Texto-Transformer de acordo com auto_model. 
        Para o Albert que utiliza AlbertaModel, retorna um TransformerAlbert.
        Para o BERT que utiliza BertModel, retorna um TransformerBert.
        Para o RoBERTa que utiliza RobertaModel, retorna um TransformerRoberta.
        Para o Distilbert que utiliza DistilbertModel, retorna um TransformerDistilbert.
        Para o GTP2 que utiliza GPT2Model, retorna um TransformerGPT2.
            
        Parâmetros:
            `tipo_modelo_pretreinado` - Tipo de modelo pré-treinado. Pode ser "simples" para criar AutoModel (default) ou "mascara" para criar AutoModelForMaskedLM.
            `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
            `cache_dir` - Cache dir para Huggingface Transformers para armazenar/carregar modelos.
            `tokenizer_args` - Argumentos (chave, pares de valor) passados para o modelo Huggingface Tokenizer
            `tokenizer_name_or_path` - Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado.
        '''    
        
        # Recupera parâmetros do transformador dos argumentos e cria um dicionário para o AutoConfig
        modelo_args_config = {"output_attentions": modelo_args.output_attentions, 
                              "output_hidden_states": modelo_args.output_hidden_states}
    
        # Carrega a configuração do modelo pré-treinado
        auto_config = AutoConfig.from_pretrained(modelo_args.pretrained_model_name_or_path,
                                                 **modelo_args_config, 
                                                 cache_dir=cache_dir)
        
        # Carrega o modelo
        auto_model = TransformerFactory._carregar_modelo(tipo_modelo_pretreinado=tipo_modelo_pretreinado,
                                                         model_name_or_path=modelo_args.pretrained_model_name_or_path,
                                                         config=auto_config, 
                                                         cache_dir=cache_dir)
       
        # Carrega o tokenizador do modelo pré-treinado
        auto_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else  modelo_args.pretrained_model_name_or_path,
                                                       cache_dir=cache_dir, 
                                                       **tokenizer_args)
        
        # Se max_seq_length não foi especificado, tenta inferir do modelo
        if modelo_args.max_seq_len is None:
            if hasattr(auto_model, "config") and hasattr(auto_model.config, "max_position_embeddings") and hasattr(auto_tokenizer, "model_max_length"):
                modelo_args.max_seq_len = min(auto_model.config.max_position_embeddings,
                                              auto_tokenizer.model_max_length)

        # Define a classe do tokenizador
        if tokenizer_name_or_path is not None:
            auto_model.config.tokenizer_class = auto_tokenizer.__class__.__name__

        # Verifica qual o modelo deve ser retornado pelos parâmetros auto
        # Passa os parâmetros para o modelo carregado a classe Transformer especifica.
        return TransformerFactory.getInstanciaTransformer(auto_model=auto_model, 
                                                          auto_config=auto_config, 
                                                          auto_tokenizer=auto_tokenizer, 
                                                          modelo_args=modelo_args)
    
    # ============================ 
    @staticmethod
    def getInstanciaTransformer(auto_model: AutoModel, 
                                auto_config: AutoConfig, 
                                auto_tokenizer: AutoTokenizer, 
                                modelo_args:ModeloArgumentos = None):
        '''
        Retorna uma classe Transformer com um modelo de linguagem carregado de acordo com os parâmetros auto_model.

        Parâmetros:
            `auto_model` - Auto model modelo carregado.
            `auto_config` - Auto config carregado.
            `auto_tokenizer` - Auto tokenizer carregado.
            `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
        '''
        
        # Verifica qual o Transformer deve ser retornado pelo parâmetro auto_model.
        if isinstance(auto_model, (BertModel, BertForMaskedLM)):
            # BertModel
            # https://huggingface.co/docs/transformers/model_doc/bert
            return TransformerBert(auto_model=auto_model, 
                                   auto_config=auto_config, 
                                   auto_tokenizer=auto_tokenizer, 
                                   modelo_args=modelo_args)
        elif isinstance(auto_model, (AlbertModel, AlbertForMaskedLM)):
            # AlbertModel
            # https://huggingface.co/docs/transformers/model_doc/albert
            return TransformerAlbert(auto_model=auto_model, 
                                     auto_config=auto_config, 
                                     auto_tokenizer=auto_tokenizer, 
                                     modelo_args=modelo_args)
        elif isinstance(auto_model, (DistilBertModel, DistilBertForMaskedLM)):            
            # DistilBertModel
            # https://huggingface.co/docs/transformers/model_doc/distilbert
            return TransformerDistilbert(auto_model=auto_model, 
                                         auto_config=auto_config, 
                                         auto_tokenizer=auto_tokenizer, 
                                         modelo_args=modelo_args)
        elif isinstance(auto_model, (GPT2Model, GPT2LMHeadModel)):
            # GPT2Model
            # https://huggingface.co/docs/transformers/model_doc/gpt2
            return TransformerGPT2(auto_model=auto_model, 
                                   auto_config=auto_config, 
                                   auto_tokenizer=auto_tokenizer, 
                                   modelo_args=modelo_args)
        elif isinstance(auto_model, (OpenAIGPTModel, OpenAIGPTLMHeadModel)):
            # OpenAIGPTModel
            # https://huggingface.co/docs/transformers/model_doc/openai-gpt
            return TransformerOpenAIGPT(auto_model=auto_model, 
                                        auto_config=auto_config, 
                                        auto_tokenizer=auto_tokenizer, 
                                        modelo_args=modelo_args)
        elif isinstance(auto_model, (RobertaModel, RobertaForMaskedLM)):
            # RobertaModel
            # https://huggingface.co/docs/transformers/model_doc/roberta
            return TransformerRoberta(auto_model=auto_model, 
                                      auto_config=auto_config, 
                                      auto_tokenizer=auto_tokenizer, 
                                      modelo_args=modelo_args)
        elif isinstance(auto_model, (XLMRobertaModel, XLMRobertaForMaskedLM)):
            # XLMRobertaModel
            # https://huggingface.co/docs/transformers/model_doc/xlm-roberta
            return TransformerXLMRoberta(auto_model=auto_model, 
                                         auto_config=auto_config, 
                                         auto_tokenizer=auto_tokenizer, 
                                         modelo_args=modelo_args)              
        elif isinstance(auto_model, (XLNetModel, XLNetLMHeadModel)):
            # XLNetModel
            # https://huggingface.co/docs/transformers/model_doc/xlnet
            return TransformerXLNet(auto_model=auto_model, 
                                    auto_config=auto_config, 
                                    auto_tokenizer=auto_tokenizer, 
                                    modelo_args=modelo_args)
        elif isinstance(auto_model, (T5EncoderModel, T5ForConditionalGeneration)):
            # T5Model
            # https://huggingface.co/docs/transformers/model_doc/t5
            return TransformerT5(auto_model=auto_model, 
                                    auto_config=auto_config, 
                                    auto_tokenizer=auto_tokenizer, 
                                    modelo_args=modelo_args)
                      
        # Outros modelos vão aqui!
        
        else:
            logger.error("Modelo não suportado: \"{}\".".format(auto_model.__class__.__name__))
            
            return None
    
    # ============================ 
    @staticmethod
    def _carregar_modelo(tipo_modelo_pretreinado: str = "simples",
                         model_name_or_path: str = None, 
                         config = None, 
                         cache_dir = None):
        '''
        Carrega o modelo transformer de acordo com o tipo.

        Parâmetros:
            `tipo_modelo_pretreinado` - Tipo de modelo pré-treinado. Pode ser "simples" (default) ou "mascara".
            `model_name_or_path` - Nome ou caminho do modelo.
            `config` - Configuração do modelo.
            `cache_dir` - Diretório de cache.
        '''

        if tipo_modelo_pretreinado == "simples":
            # Carrega modelo pré-treinado simples usando AutoModel
            # https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel
            
            if isinstance(config, T5Config):
                return T5EncoderModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                      config=config, 
                                                      cache_dir=cache_dir) 
            else:
                return AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                 config=config, 
                                                 cache_dir=cache_dir)
            
        elif tipo_modelo_pretreinado == "mascara":
            # Carrega modelos pré-treinados para processamento de linguagem mascarada usando AutoModelForMaskedLM
            # https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMaskedLM
            
            if isinstance(config, GPT2Config):
                return GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       config=config, 
                                                       cache_dir=cache_dir)
          
            elif isinstance(config, OpenAIGPTConfig):
                return OpenAIGPTLMHeadModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                            config=config, 
                                                            cache_dir=cache_dir)
            elif isinstance(config, XLNetConfig):
                return XLNetLMHeadModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                        config=config, 
                                                        cache_dir=cache_dir)                
            else:
                return AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                            config=config, 
                                                            cache_dir=cache_dir)
        
        # Outros tipos de modelos vão aqui!
        
        else:
            logger.error("Tipo de modelo pré-treinaddo não suportado: \"{}\".".format(tipo_modelo_pretreinado))
            
            return None