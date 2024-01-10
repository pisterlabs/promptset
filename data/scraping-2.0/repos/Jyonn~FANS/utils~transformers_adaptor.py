import transformers

major_version = transformers.__version__.split('.')[0]

if major_version >= '4':
    # BERT
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as BertOutput
    from transformers.models.bert.modeling_bert import BertModel

    # BART
    from transformers.models.bart.modeling_bart import BartModel

    # GPT
    from transformers.models.openai.modeling_openai import OpenAIGPTModel
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
else:
    # BERT
    from transformers.modeling_outputs import BaseModelOutputWithPooling as BertOutput
    from transformers.modeling_bert import BertModel

    # BART
    from transformers.modeling_bart import BartModel

    # GPT
    from transformers.modeling_openai import OpenAIGPTModel
    from transformers.modeling_gpt2 import GPT2Model

from transformers.modeling_outputs import Seq2SeqModelOutput

BartOutput = Seq2SeqModelOutput
