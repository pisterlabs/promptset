from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain.llms import OpenAI
from langchain.chains import APIChain
from dotenv import load_dotenv
from langchain.chains.api import podcast_docs

load_dotenv()

# chain = get_openapi_chain("http://192.168.12.128:99/swagger/index.html?urls.primaryName=AllianzSF%20Api%20Portal%20-%20v1")

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IkI5MUFDNzEyMEJGMjc3NUYyMDcxMEYyN0U5OUZERDIwIiwidHlwIjoiYXQrand0In0.eyJuYmYiOjE2OTQxNTI3MTMsImV4cCI6MTcyNTY4ODcxMywiaXNzIjoiaHR0cDovLzE5Mi4xNjguMTIuMTI4OjgzIiwiYXVkIjoiQWxsaWFuelNGQXBpUG9ydGFsIiwiY2xpZW50X2lkIjoiQWxsaWFuelNGQXBpUG9ydGFsX0FwaSIsImlhdCI6MTY5NDE1MjcxMywic2NvcGUiOlsiQWxsaWFuelNGQXBpUG9ydGFsIl19.I6JrqFEGsOR0oA7d1YPYtO2ELkc3NbHI_i5SM8HMlazuFw0mhWzCNcJIIepf7pqcLd_4cW2xNyI5nKHoQY6ynR4h7V6XFvo7ClzkFn_BdFqEWQJoxn5fot8Uc4cBHqG4OizG0X0W4ID1Ik5quJxeAPcviV6dKdBb98FEpcCvbcbeBCAMX0RUPoO7ZZN4Fv_cbtvJOQYfVvyGLerHUOALOqxZO1ApmT3cniwqAiQmiri-Rw6hMLgrR3RwrZcRxwjVrmulL0rrmmi02xv0lt3KeJtQew1mPmhEup_8XJxDkLFFl6IF6UwP6bP499nyn7xSmbTu4wODqRPd_qTj6fm7rQ"
claims_docs = "http://192.168.12.128:85/swagger/1.0/swagger.json"
llm = OpenAI(temperature=0)
headers = {"Authorization": f"Bearer {token}"}
chain = APIChain.from_llm_and_api_docs(llm, claims_docs, headers=headers, verbose=True)
# chain.run("Can you retrieve the claims for the certificate number 123 and serial number 456 and external reference 789")
chain.run("search for a policy with external reference 789")
# chain.run("set the ai\temperature of the ai to 1")

# print(podcast_docs.PODCAST_DOCS)