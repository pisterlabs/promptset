from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.run("""
             직선 위에서 어떤 사람이 점 A에서 점 B까지 처음에 7.2m/s 의 일정한 속력으로 가다가 다시 점 B에서 A로 4.5 m/s의 일정한 속력으로 돌아올 때, 
이 사람이 전체 이동하는 동안의 평균 속력이랑 평균 속도를 이유와 함께 답해줘
             """)
