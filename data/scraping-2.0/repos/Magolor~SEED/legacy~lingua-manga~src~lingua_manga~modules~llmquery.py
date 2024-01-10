from .default import *
from langchain.prompts import PromptTemplate

class LLMQueryPromptTemplate(PromptTemplate):
    def format(self, data) -> str:
        return self.template.format(
            task_desc = data.task_desc,
            examples_desc = add_indent(parse_examples(data.examples)),
            outputs_desc = "Instance #0:\n" + add_indent("\n".join([f"{o}=?" for o in data.outputs])) + "\n...\n",
            instance = "{instance}",
        )

LLMQUERY_PROMPT_TEMPALTE = LLMQueryPromptTemplate.from_template(
    "{task_desc}\n"
    "Examples:\n"
    "{examples_desc}\n"
    "Now consider the following instance(s):\n"
    "{instance}\n"
    "Please respond with the answer only. Please do not output any other responses or any explanations.\n"
    "Your respond should strictly align with the format of `Output:` in the above examples, including the punctuations. It should be in the following format for each instance:\n"
    "{outputs_desc}\n"
)

@LinguaManga.register
class LLMQueryModule(Module):
    __type__: str = 'module-llmquery'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        if self.task_desc is None: self.task_desc = ""
        if self.inputs_desc is None: self.inputs_desc = dict()
        if self.outputs_desc is None: self.outputs_desc = dict()
        if self.tools_desc is None: self.tools_desc = dict()
        self.inputs = list(self.inputs_desc.keys())
        self.outputs = list(self.outputs_desc.keys())

    def __compile__(self):
        prompt = LLMQUERY_PROMPT_TEMPALTE.format(data=self)
        build_example_code = "inputs = dict()\n"
        for key in self.inputs:
            build_example_code += f"{parameterize(key, param='inputs')} = {key}\n"
        # for key in self.outputs:
        #     build_example_code += f"{parameterize(key, param='outputs')} = '?'\n"
        code = (
            f"def {self.api()}:\n"+ 
            add_indent(f"'''")+"\n"+
            add_indent(prompt)+"\n"+
            add_indent(f"'''")+"\n"+
            add_indent(build_example_code)+"\n"+
            add_indent(f"prompt = {repr(prompt)}")+"\n"+
            add_indent(f"return LLMQueryExec(prompt, [inputs])[0]")+"\n"
        )
        
        batching_code = (
            f"def {self.batch_api()}:\n"+ 
            add_indent(f"prompt = {repr(prompt)}")+"\n"+
            add_indent(f"return LLMQueryExec(prompt, instances)")+"\n"
        )
        
        optimized_code = (
            f"{self.name}_cache = KVCache(path='./cache/')"+"\n"+
            f"{self.name}_simul_cls = CLSSimulator(path='./simuls/{self.name}/')"+"\n"+
            f"def {self.optimized_api()}:\n"+
            add_indent("results = list(); batch = list(); batch_keys = list()")+"\n"+
            add_indent("for i, inputs in TQDM(list(enumerate(instances))):")+"\n"+
            add_indent(f"    f = True", 2)+"\n"+
            add_indent(f"    key = '|'.join(['function={self.name}'] + "+"[f'{k}={v}' for k,v in inputs.items()])", 2)+"\n"+
            add_indent(f"    if f and cacher:", 2)+"\n"+
            add_indent(f"        _, response, d = {self.name}_cache.query(key)", 3)+"\n"+
            add_indent(f"        if (response is not None) and (d <= {0.3}*2):", 3)+"\n"+
            add_indent(f"            results.append(response); f = False; counter.add('cached')", 4)+"\n"+
            add_indent(f"    if f and simulator:", 2)+"\n"+
            add_indent(f"        logits = {self.name}_simul_cls.query(key).detach().numpy()", 3)+"\n"+
            add_indent(f"        value = logits.argmax()", 3)+"\n"+
            add_indent(f"        confidence = min(max(2*(logits[value]-0.5),0),1)", 3)+"\n"+
            add_indent(f"        print(i, confidence)", 3)+"\n"+
            add_indent(f"        if confidence >= {0.5}:", 3)+"\n"+
            add_indent(f"            response = dict(); response['{self.outputs[0]}'] = value; results.append(response); f = False; counter.add('simulated')", 4)+"\n"+
            add_indent(f"    if f:", 2)+"\n"+
            add_indent(f"        batch.append(inputs); batch_keys.append(key); counter.add('queried')", 3)+"\n"+
            add_indent(f"    if len(batch) >= batching or i==len(instances)-1:", 2)+"\n"+
            add_indent(f"         if batching==1:", 3)+"\n"+
            add_indent(f"             batch_responses = [{self.name}(**inputs)]", 4)+"\n"+
            add_indent(f"         else:", 3)+"\n"+
            add_indent(f"             batch_responses = {self.name}_batch(instances=batch)", 4)+"\n"+
            add_indent(f"         for key, response in zip(batch_keys, batch_responses):", 3)+"\n"+
            add_indent(f"             if cacher:", 4)+"\n"+
            add_indent(f"                 {self.name}_cache.update(key, response)", 5)+"\n"+
            add_indent(f"             if simulator:", 4)+"\n"+
            add_indent(f"                 value = response['{self.outputs[0]}']", 5)+"\n"+
            add_indent(f"                 {self.name}_simul_cls.update(key, value)", 5)+"\n"+
            add_indent(f"         results.extend(batch_responses)", 3)+"\n"+
            add_indent(f"         batch = list(); batch_keys = list()", 3)+"\n"+
            add_indent(f"return results")
        )
        return Cell(code="\n\n".join([code,batching_code,optimized_code]))