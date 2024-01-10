import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from instruct_pipeline import InstructionTextGenerationPipeline
from langchain import FewShotPromptTemplate, PromptTemplate

import transformers
import warnings
def main():

    ignore_warnings = True
    if ignore_warnings:
        print("All the warnings will be ignored. Switch ignore_warning to False if needed")
        warnings.filterwarnings("ignore")
        
    name = 'mosaicml/mpt-30b-instruct'
    #name = "databricks/dolly-v2-12b"

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    #config.max_seq_len = 8192
    #config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    #config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    load_8bit = True
    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    
    # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    # device_map = {
    #     "transformer.word_embeddings": 0,
    #     "transformer.word_embeddings_layernorm": 0,
    #     "lm_head": "cpu",
    #     "transformer.h": 0,
    #     "transformer.ln_f": 0,
    # }
    
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     name,
    #     config=config,
    #     torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    #     trust_remote_code=True,
    #     load_in_8bit=load_8bit,
    #     device_map=device_map,
    #     quantization_config=quantization_config,
    # )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        device_map="cpu",
    )




    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    italian = False
    if italian:
        text_prompt = "Write me two lines about Giorgio Armani. In italian."

        print(f"\nPrompt: {text_prompt}\n")

        print(pipe(text_prompt))

        text_prompt2 = "Given a email text remove all the parts that have been added automatically. Extract the written email like in the following example.\n Example 1: Reclamo C45673 cp-2019-9.s per conto di Angelo P. RECLAMO EMAIL <da: ange.pinotto@lubiro.com> <a:buoenas@alipre.com> Oggetto: Ritardo rimborso prenotazione  Buongiorno, sono Angelo Pinotto vostro cliente dal 2015. Ho richiesto un rimborso in seguito ad una prenotazione presso uno dei vostri collaboratori (ULIMP) ma ancora non ho ricevuto il versamento. Da polizza il rimborso deve essere fatto entra 30gg, sono già passati 45gg e ancora non mi è stato rimborsato. Ho già contattato l\'assistenza ULIMP ma ancora non mi hanno risposto. Potete aiutarmi? Cordiali saluti, Angelo <mail: inviata il 20.8.2019 da <ange.pinotto@lubrio.com> Questa mail è protetta con AVAST antivirus. è conforma alle regolamentazione riguardo la protezione e trattamento dei dati GDPR art.240-7-16. Se questa email non ti riguarda ti preghiamo di eliminarla. Ogni sua ridistribuzione è vietata e punibile per legge.\n\n Answer 1: Oggetto: Ritardo rimborso prenotazione  Buongiorno, sono Angelo Pinotto vostro cliente dal 2015. Ho richiesto un rimborso in seguito ad una prenotazione presso uno dei vostri collaboratori (ULIMP) ma ancora non ho ricevuto il versamento. Da polizza il rimborso deve essere fatto entra 30gg, sono già passati 45gg e ancora non mi è stato rimborsato. Ho già contattato l\'assistenza ULIMP ma ancora non mi hanno risposto. Potete aiutarmi? Cordiali saluti, Angelo  \n\n Example 2:  6789m da Hleip.spa. cp-2018-9089-0-9.s reclamante Maria RECLAMO EMAIL <da: hleip.spa@ghk.com> <a:buoenas@alipre.com> Oggetto: Mancato versamento  Buongiorno, sono Maria legale rappresentate di hleipspa. E' stato richiesto in data 25.06.2018 un rimborso come specificato da polizza 3456791mt art.4 comma7  per il mio cliente al quale non è stato ancora effattuato. In quanto suo legale comunico la nostra intezione a procedere per vie legali presso autorità maggiori qualora non pervenga un rimosrso entro 15gg dalla data di ricezione della email Cordiali saluti,\
         AVV.Maria Mariaetti  <mail: inviata il 30.8.2019 This mail is protected by Mah ANTIVIRUS. Has been scanned and empty \
          from any source of malware or source of contamination. è conforma alle regolamentazione riguardo la protezione e trattamento dei dati GDPR art.240-7-16.\
           Se questa email non ti riguarda ti preghiamo di eliminarla. Ogni sua ridistribuzione è vietata e punibile per legge.\
            This mail has been sent from an authorized organization with respect to the art5 of LGEh .Please do nt reply to this email \
            if your are not the person/organization specified in the object \n\n Answer 2: ?"

        torch.cuda.empty_cache()

        print(f"\nPrompt: {text_prompt2}\n\n")
        print(pipe(text_prompt2)[0]['generated_text'].split('<|endoftext|>')[0])

    torch.cuda.empty_cache()
    print("\n ==================== LANGCHAIN: MPT-30B-instruct ====================")
    print("\t\tTask: Few-shot Prompting")

    # create our examples
    examples = [
        {
            "query": "Quanto ancora devo aspettare per una vostra risposta? Sono deluso, utilizzo i vostri servizi da moltissimi anni e siete sempre più lenti!",
            "answer": " Ritardo nella risposta"
        }, {
            "query": "Non è possibile che siano solo 50 euro!! Dovevano essere 150!!! Vergogna!!!!",
            "answer": "Importo liquidato"
        },
        {
            "query": "Mi fate schifo, solo 30 euro? Troppo pochi!!",
            "answer": "Importo liquidato"
        },
        {
            "query": "Ma mi rispondete o mi prendete in giro? cordialmente",
            "answer": "Ritardo nella risposta"
        },
        {
            "query": "Non sono ancora stato risarcito dei miei 60 euro nonostante siano passati più di 10gg",
            "answer": "Mancato risarcimento"
        },
        {
            "query": "Sono passati due mesi, quanto ancora devo aspettare?",
            "answer": "Ritardo nella risposta"
        }
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions ##  based on the following examples The following are emails from customers to an insurance company. The customer is typically complaining about something. \
    # prefix = """
    # Do a classification of the text among the classes: (Ritardo nella risposta, Importo liquidato, Mancato risarcimento, Nothing). Select just one class name based on the following examples.\
    # Here are some examples:
    # """
    prefix = """
        Classifica il seguente testo tra queste classi : (Ritardo nella risposta, Importo liquidato, Mancato risarcimento, Nothing). Seleziona una sola classe basandoti sugli esempi seguenti.\
        Qua gli esempi: 
        """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="" # \n
    )

    print("------Template:")

    query = "Avevate detto massimo una settimana, ne sono passate 3! Potete considerarmi???"
    print(few_shot_prompt_template.format(query=query))

    print('\n------Risposta:')
    print('\t', pipe(few_shot_prompt_template.format(query=query))[0]['generated_text'].split('<|endoftext|>')[0])

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
