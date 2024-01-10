from typing import Type, List

from langchain.llms import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.schema.output_parser import BaseOutputParser, StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema.runnable import Runnable
from langchain_app.contexts import ExtractionContext, DocumentExtractionContext, StringExtractionContext
from langchain_app.templates import SimpleExtractionPromptTemplate, ExtractionPromptTemplate


class ExtractionChain:
    def __init__(self, stages:List[Runnable]=[]):
        self.stages = stages
    
    def add_stage(self, stage:Runnable):
        self.stages.append(stage)

    def generate_chain(self):
        chain = self.stages[0] if len(self.stages) > 0 else None
        for stage in self.stages[1:]:
            chain = chain | stage
        return chain
    
    def invoke(self, data):
        return self.generate_chain().invoke(data)

# class ExtractionChain:
#     def __init__(self, context:ExtractionContext, template:ExtractionPromptTemplate, model:BaseLLM, parser:BaseOutputParser):
#         self.context = context
#         self.template = template
#         self.model = model
#         self.parser = parser

#     def extract(self, data):
#         return self.generate_chain().invoke(data)

#     def generate_chain(self):
#         return (
#             {"context": self.context.get_runnable()}
#             | self.template.get_runnable()
#             | self.model
#             | self.parser
#         )
    
class SimpleDocumentStringExtractionChainBuilder:
    def __init__(self, source_data=None, schema=None):
        self.source_data = source_data
        self.schema = schema
        self.context = StringExtractionContext()
        self.template = SimpleExtractionPromptTemplate()
        self.model = OpenAI()
        self.parser = StrOutputParser()

    def build(self):
        context = self.context.get_runnable(self.source_data)
        template = self.template.get_runnable(self.schema)
        return ExtractionChain(context, template, self.model, self.parser)
    
    def extract(self, data="", source_data=None, schema=None):
        source_data = source_data or self.source_data
        schema = schema or self.schema
        return self.build().invoke(data)
    
class SimpleDocumentPDFExtractionChainBuilder:
    def __init__(self, artifact=None, schema=None, context=None, template=None, model=None, parser=None):
        self.artifact = artifact
        self.schema = schema
        self.context = context or DocumentExtractionContext(artifact)
        self.template = template or SimpleExtractionPromptTemplate(schema)
        self.model = model or OpenAI()
        self.parser = parser or SimpleJsonOutputParser()

    def add(self, artifact=None, schema=None, context=None, template=None, model=None, parser=None):
        self.artifact = artifact or self.artifact
        self.schema = schema or self.schema
        self.context = context or self.context
        self.template = template or self.template
        self.model = model or self.model
        self.parser = parser or self.parser
        return self

    def build(self):
        context = self.context.get_runnable(self.artifact)
        context_dict = {"context": context}
        template = self.template.get_runnable(self.schema)
        return ExtractionChain(stages=[context_dict, template, self.model, self.parser])
        #return ExtractionChain(scontext, template, self.model, self.parser)
    
    def extract(self, data="", artifact=None, schema=None):
        self.artifact = artifact or self.artifact
        self.schema = schema or self.schema
        return self.build().invoke(data)

    
# class SimpleDocumentPDFExtractionChainBuilder:
#     def __init__(self, artifact, schema):
#         self.source_data = artifact
#         self.schema = schema
#         self.context = DocumentExtractionContext(artifact)
#         self.template = SimpleExtractionPromptTemplate(schema)
#         self.model = OpenAI()
#         self.parser = StrOutputParser()

#     def build(self):
#         return ExtractionChain(self.context, self.template, self.model, self.parser)

# def extract_data_inline(data, schema):
#     extraction_chain = SimpleDocumentStringExtractionChainBuilder(data, schema).build()
#     return extraction_chain.extract(data="")



# def extract_data_pdf(artifact, schema):
#     extraction_chain = SimpleDocumentPDFExtractionChainBuilder(artifact, schema).build()
#     return extraction_chain.extract(data="")
