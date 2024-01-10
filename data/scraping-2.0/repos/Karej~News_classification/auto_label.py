import os
import openai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY_FREE")
openai.api_base = "https://api.chatanywhere.cn/v1"
context = "Tôi sẽ đưa cho bạn một bài báo, hãy chỉ ra tên công ty được nói đến trong bài báo ấy. Sau đây là bài báo:"
article = "Dưới tác động mạnh mẽ của COVID-19 đến ngành hàng không, Dịch vụ Hàng hóa Sài Gòn dự kiến lợi nhuận trước thuế năm 2020 sẽ giảm 16,3% so với thực hiện năm 2019, đây là năm đầu tiên trong suốt nhiều năm qua, Dịch vụ Hàng hóa Sài Gòn đặt kế hoạch tăng trưởng âm.Cụ thể, theo tài liệu họp ĐHĐCĐ năm 2020 vừa được Công ty Cổ phần Dịch vụ Hàng hóa Sài Gòn (SCSC - Mã: SCS) công bố, HĐQT SCSC dự kiến sẽ trình cổ đông thông qua kế hoạch kinh doanh 2020 với mục tiêu doanh thu thuần đạt 660 tỉ đồng và 450 tỉ đồng lợi nhuận trước thuế, lần lượt giảm 11,8% và 16,3% so với thực hiện năm 2019."
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=context + article,
  temperature=0,
  max_tokens=50,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

generated_text = response.choices[0].text.strip()

print(generated_text)
