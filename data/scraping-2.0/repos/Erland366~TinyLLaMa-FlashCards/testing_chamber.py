import argparse
import json
import os

import requests

from tinyllama_breakdown.quantization.awq_quantization import QuantizeAWQ
from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE_EN,
)
from tinyllama_breakdown.utils import (
    TokenCounter,
    compare_token_id_vs_en,
    parse_output,
    read_yaml_file,
)
from tinyllama_breakdown.vllm.api_client import vanilla_inference
from tinyllama_breakdown.vllm.langchain_chat import langchain_inference

EXAMPLE_INPUT = """### Instruction:
Saya ingin Anda berperan sebagai pembuat Anki Cards profesional, mampu membuat Anki Cards dari teks yang saya berikan.

Mengenai perumusan isi kartu, Anda berpegang pada dua prinsip:
Pertama, prinsip informasi minimum: Materi yang dipelajari harus dirumuskan sesederhana mungkin. Kesederhanaan tidak harus berarti kehilangan informasi dan melewatkan bagian yang sulit.
Kedua, optimalkan susunan kata: Susunan kata pada item Anda harus dioptimalkan untuk memastikan bahwa dalam waktu singkat lampu yang tepat di otak Anda menyala. Hal ini akan mengurangi tingkat kesalahan, meningkatkan spesifisitas, mengurangi waktu respons, dan membantu konsentrasi Anda.

### Input:
Ciri-ciri Laut Mati: Danau garam yang terletak di perbatasan antara Israel dan Yordania. Garis pantainya merupakan titik terendah di permukaan bumi, rata-rata 396 m di bawah permukaan laut. Panjangnya 74 km. Tujuh kali lebih asin (30% volume) dibandingkan lautan. Kepadatannya membuat perenang tetap bertahan. Hanya organisme sederhana yang dapat hidup di perairan asinnya.

### Response:
{"Q": "Dimana letak Laut Mati?", "A": "di perbatasan antara Israel dan Yordania"}
{"Q": "Apa titik terendah di permukaan bumi?", "A": "Garis pantai Laut Mati"}
{"Q": "Berapakah ketinggian rata-rata di mana Laut Mati berada?", "A": "400 meter (di bawah permukaan laut)"}
{"Q": "Berapa panjang Laut Mati?", "A": "70 km"}
{"Q": "Seberapa asinkah Laut Mati dibandingkan dengan lautan?", "A": "7 kali"}
{"Q": "Berapa volume kandungan garam di Laut Mati?", "A": "30%"}
{"Q": "Mengapa Laut Mati bisa membuat para perenang tetap bertahan?", "A": "karena kandungan garamnya tinggi"}
{"Q": "Mengapa Laut Mati disebut Laut Mati?", "A": "karena hanya organisme sederhana yang dapat hidup di dalamnya"}
{"Q": "Mengapa hanya organisme sederhana yang dapat hidup di Laut Mati?", "A": "karena kandungan garamnya tinggi"}

### Input:
Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.


### Response:
{"Q": "Apakah Sekolah Menegah Atas Negeri dan Swasta Tersebar Di Tanah Air?", "A": "Ya."}
{"Q": "Bagaimana Pemilihan Jurusan Dikelilingkingkan Pada Jenjang Pendidikan Pada SMK?", "A": "Hanya dilakukan pada jenjang pendidikan pada SMK."}
{"Q": "Bisa Juga Berlangsung Pada Siswa Siswi Yang Memilih Masuk Ke SMA?", "A": "Juga."}
{"Q": "Apakah Jurusan Yang Terdapat Pada SMA Antara Lain Jurusan IPA, IPS", "A": "Ya."}"""

EXAMPLE_INPUT_EN = """### Instructions:
I want you to act as a professional Anki Cards maker, able to create Anki Cards from the text I provide.

Regarding the formulation of the contents of the card, you adhere to two principles:
First, the principle of minimum information: The material studied should be formulated as simply as possible. Simplicity doesn't have to mean losing information and skipping difficult parts.
Second, optimize the wording: The wording of your items should be optimized to ensure that in no time the right lights in your brain turn on. This will reduce error rates, increase specificity, reduce response times, and help your concentration.

### Inputs:
Characteristics of the Dead Sea: Salt lake located on the border between Israel and Jordan. The coastline is the lowest point on the earth's surface, an average of 396 m below sea level. Its length is 74 km. Seven times saltier (30% by volume) than the ocean. Its density keeps swimmers afloat. Only simple organisms can live in its salty waters.

### Response:
{"Q": "Where is the Dead Sea?", "A": "on the border between Israel and Jordan"}
{"Q": "What is the lowest point on the earth's surface?", "A": "Dead Sea coastline"}
{"Q": "What is the average height at which the Dead Sea is located?", "A": "400 meters (below sea level)"}
{"Q": "How long is the Dead Sea?", "A": "70 km"}
{"Q": "How salty is the Dead Sea compared to the ocean?", "A": "7 times"}
{"Q": "What is the volume of salt content in the Dead Sea?", "A": "30%"}
{"Q": "Why can the Dead Sea keep swimmers afloat?", "A": "because it has a high salt content"}
{"Q": "Why is the Dead Sea called the Dead Sea?", "A": "because only simple organisms can live in it"}
{"Q": "Why can only simple organisms live in the Dead Sea?", "A": "because the salt content is high"}

### Inputs:
Public and private high schools are spread across various regions in the country. High school is one of the choices that many students choose after graduating from middle school. Continuing high school has a greater chance of entering a state university by choosing a college major that interests them.

Choosing a major is not only done at the vocational school level. However, this also applies to female students who choose to enter high school. The majors in high school include: science major, social studies and language major.


### Response:
{"Q": "Are Public and Private High Schools Widespread in the Country?", "A": "Yes."}
{"Q": "How is the choice of major related to the education level at vocational school?", "A": "Only carried out at the education level at vocational school."}
{"Q": "Can This Also Happen to Female Students Who Choose to Enter High School?", "A": "Also."}
{"Q": "Are there majors in high school, including science and social studies", "A": "Yes."}"""


def quantize_model():
    config = read_yaml_file("config/quantize_dummy.yaml")
    quantize_awq = QuantizeAWQ(config)
    quantize_awq.quantize()


def print_example_data():
    example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.
"""  # noqa: E50
    the_thing = GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE.format(
        input_user=example_prompt, response=""
    )
    print(the_thing)


def request_to_my_model():
    API_URL = "https://api-inference.huggingface.co/models/Erland/tinyllama-1.1B-chat-v0.3-dummy"
    API_TOKEN = os.getenv("HF_TOKEN_API")
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    data = query(
        {
            "inputs": EXAMPLE_INPUT,
            "options": {"wait_for_model": True}
            # "parameters": {"max_new_tokens": 250, "max_input_length": 1500},
        }
    )
    print(data)
    print(parse_output(data[0]["generated_text"]))


def main(args) -> None:
    example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

    Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.
    """  # noqa: E49
    example_prompt = GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE.format(
        input_user=example_prompt, response=""
    )
    print(example_prompt)
    print("=====================================")
    print("Start Inference")
    print("=====================================")
    if args.openai:
        langchain_inference(example_prompt)
    else:
        vanilla_inference(example_prompt, stream=args.stream, n=args.n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--openai", action="store_true")
    args = parser.parse_args()
    main(args)
