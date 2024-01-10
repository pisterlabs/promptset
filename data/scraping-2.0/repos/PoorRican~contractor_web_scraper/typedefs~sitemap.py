from langchain.pydantic_v1 import BaseModel, Field, HttpUrl


class SiteMap(BaseModel):
    homepage: HttpUrl = Field(description="The homepage of the site")
    about: HttpUrl = Field(..., description="The about page of the site")
    contact: HttpUrl = Field(..., description="The contact page of the site")
