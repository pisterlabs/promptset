from openai import OpenAI
import urllib.request
from dotenv import load_dotenv

### .envファイルにOPENAI_API_KEYを記入することでapi_keyのベタ書きを防ぐ
load_dotenv()


class Dalle2Communication(object):
    """DALLE II API向けのクラス。画像を取得をまとめる

    Args:
        object (_type_): _description_
    """

    def __init__(self, api_key=None, prompt_initial=None, blank_class=True):
        self.model = "dall-e-3"
        if api_key == None:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)
        self.default_prompt = "a white siamese cat"
        if blank_class:
            self.prompt_history = []
            self.images_url = []
        else:
            if prompt_initial == None:
                self.prompt_history = [self.default_prompt]
            else:
                self.prompt_history = [prompt_initial]
            self.images_url = [self.add_image_generation(self.prompt_history[0])]
        self.size = "1024x1024"
        self.quality = "standard"
        self.n = 1

    def add_image_generation(self, prompt):
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            quality=self.quality,
            n=self.n,
        )
        self.prompt_history.append(prompt)
        self.images_url.append(response.data[0].url)
        return self.images_url[-1]

    def download_image_latest(self, filename):
        if len(self.images_url) == 0:
            return "you need 1 image at least"
        else:
            self.dl_image_from_url(self.images_url[-1], filename=filename)

    def download_image_all(self, common_filename):
        if len(self.images_url) == 0:
            return "you need 1 image at least"
        else:
            for i, image_url in enumerate(self.images_url):
                self.dl_image_from_url(image_url, filename=f"{common_filename}_{i:03}.png")

    def dl_image_from_url(self, image_url, filename="output.png"):
        with urllib.request.urlopen(image_url) as web_file:
            data = web_file.read()
            with open(filename, mode="wb") as local_file:
                local_file.write(data)
    
    def change_image_size(self, x, y):
        if type(x) == int and type(y) == int:
            self.size = f"{x}x{y}"
        else:
            print("x and y must be int")

if __name__ == "__main__":
    d2c = Dalle2Communication()
    d2c.add_image_generation("a black siamese cat")
    d2c.download_image_latest("dalle_image_output_sample.png")
