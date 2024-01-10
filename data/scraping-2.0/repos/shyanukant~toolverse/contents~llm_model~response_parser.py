from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

def parser():
    class ContentModel(BaseModel):
        heading : str = Field(description="This is heading")
        body : str = Field(description="This is body")

    class SEOModel(BaseModel):
        keyword : list = Field(description="list of keywords")
        hashtags : list = Field(description="list of hashtags")

    class PlatformModel(BaseModel):
        platform : ContentModel = Field(description="platforms")

    class ReponseModel(BaseModel):
        script : ContentModel = Field(description="full script of content ")
        social : PlatformModel = Field(description="social platforms")
        seo : SEOModel = Field(description="seo contents")

    output_parser = PydanticOutputParser(pydantic_object=ReponseModel)
    return output_parser