import re
import openai

class SceneAnalyzer:
    def __init__(self, api_key, filename):
        self.api_key = api_key  # OpenAI GPT-4를 사용하기 위한 API 키
        self.filename = filename  # 자막 파일의 이름
        openai.api_key = self.api_key  # OpenAI 라이브러리에 API 키 설정
    # 자막의 내용을 분석하여 해당 장면의 강도를 산출하는 메소드
    def scene_analysis(self, content):
        response = openai.ChatCompletion.create(
            model="gpt-4-0613", # GPT-4 모델을 사용
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rates the intensity of described scenes on a scale of 0 (calm) to 10 (intense). Please respond with a whole number between 0 and 10."},
                {"role": "user", "content": f"This is a scene description: '{content}'."}, # 분석할 장면의 내용
            ]
        )
        
        # 모델의 응답을 숫자로 변환하여 강도를 산출
        # 만약 응답이 예상 범위를 벗어나거나 변환할 수 없는 경우, 중간 강도로 재시도
        try:
            intensity = round(float(response['choices'][0]['message']['content'].strip()))
            if 0 <= intensity <= 10:
                return intensity
            else:
                raise ValueError()
        except ValueError:
            print(f"Unexpected response: {response['choices'][0]['message']['content'].strip()}")
            print("Retrying with a median intensity...")
            
            # Retry with median intensity
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rates the intensity of described scenes on a scale of 0 (calm) to 10 (intense). Please respond with a whole number between 0 and 10."},
                    {"role": "user", "content": f"This is a scene description: '{content}'. The scene is of medium intensity."},
                ]
            )

            try:
                intensity = round(float(response['choices'][0]['message']['content'].strip()))
                if 0 <= intensity <= 10:
                    return intensity
                else:
                    print(f"Unexpected response on retry: {response['choices'][0]['message']['content'].strip()}")
                    return None
            except ValueError:
                print(f"Unexpected response on retry: {response['choices'][0]['message']['content'].strip()}")
                return None

    # 자막 파일에서 각 장면의 설명과 시간을 추출하는 메소드
    def extract_bracket_content_and_time(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        bracket_contents_with_time = []
        current_time = None
        
        pattern = re.compile(r'\[(.*?)\]')

        for line in lines:
            line = line.strip()
            if '-->' in line: 
                current_time = line
            else:
                matches = pattern.findall(line)
                if matches:
                    for match in matches:
                        bracket_contents_with_time.append((current_time, match))
                        
        return bracket_contents_with_time
        
    # 전체 자막 파일을 처리하여 각 장면의 강도를 산출하는 메소드
    def process_subtitle(self):
        bracket_contents_with_time = self.extract_bracket_content_and_time()
        
        for i in range(len(bracket_contents_with_time)):
            time, content = bracket_contents_with_time[i]
            intensity = self.scene_analysis(content)
            if intensity is not None:
                bracket_contents_with_time[i] = (time, intensity)

        return bracket_contents_with_time

if __name__ == "__main__":
    analyzer = SceneAnalyzer('api-key', 'srt_dir') # SceneAnalyzer 인스턴스 생성
    results = analyzer.process_subtitle() # 자막 파일 분석
    print(results) # 결과 출력
