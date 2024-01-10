# Embeddings

# 임베딩은 모든 종류의 데이터를 표현하는 인공지능 고유의 방식으로, 모든 종류의 인공지능 기반 도구 및 알고리즘으로 작업하는 데 적합합니다.
# 임베딩은 텍스트, 이미지, 그리고 곧 오디오와 비디오를 나타낼 수 있습니다. 임베딩을 생성하는 데는 설치된 라이브러리를 로컬로 사용하거나 API를 호출하는 등 다양한 옵션이 있습니다.
# 크로마는 인기 있는 임베딩 제공업체에 대한 경량 래퍼를 제공하므로 앱에서 쉽게 사용할 수 있습니다.
# Chroma 컬렉션을 만들 때 임베딩 기능을 설정하면 자동으로 사용되거나 직접 호출할 수 있습니다.

# Chroma의 임베딩 함수를 가져오려면 chromadb.utils.embedding_함수 모듈을 가져옵니다.
from chromadb.utils import embedding_functions
import config, os


# Default: all-MiniLM-L6-v2 #######################################
# 기본적으로 Chroma는 임베딩을 생성할 때 Sentence Transformers all-MiniLM-L6-v2 모델을 사용합니다.
# 이 임베딩 모델은 다양한 작업에 사용할 수 있는 문장 및 문서 임베딩을 생성할 수 있습니다.
# 이 임베딩 기능은 컴퓨터에서 로컬로 실행되며 모델 파일을 다운로드해야 할 수 있습니다(자동으로 실행됨).
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Tip
# 임베딩 함수는 컬렉션에 연결할 수 있으며, 추가, 업데이트, 업서트 또는 쿼리를 호출할 때마다 사용됩니다.
# 또한 직접 사용할 수도 있어 디버깅에 유용할 수 있습니다.

val = default_ef("foo")
print(len(val), len(val[0]))  # (3,384) : 384차원 벡터, char 단위 임베딩
val = default_ef(["bare", "foo"])
print(len(val), len(val[0]))  # (2,384) : word 단위 임베딩 방법
val = default_ef("한국어는 어떨까")
print(len(val), len(val[0]))  # (8,384) : char 단위 임베딩 방법
val = default_ef(["한국어는", "어떨까"])
print(len(val), len(val[0]))  # (2,384) : word 단위 임베딩 방법
# -> [[0.05035809800028801, 0.0626462921500206, -0.061827320605516434...]]

# Sentence Transformers #######################################
# 크로마는 모든 Sentence 트랜스포머 모델을 사용하여 임베딩을 생성할 수도 있습니다.
# 관련 패키지 설치방법
# Rustup 설치 : https://rustup.rs/에서 다운로드 및 설치
# pip install sentence_transformers 명령 실행
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
val = sentence_transformer_ef("foo")
print(len(val), len(val[0]))  # (3,384) : 384차원 벡터, char 단위 임베딩
val = sentence_transformer_ef(["bare", "foo"])
print(len(val), len(val[0]))  # (2,384) : word 단위 임베딩 방법
# 선택적으로 model_name 인수를 전달하여 사용할 Sentence Transformer 모델을 선택할 수 있습니다.
# 기본적으로 Chroma는 all-MiniLM-L6-v2를 사용합니다. (사용가능 모델리스트 : https://www.sbert.net/docs/pretrained_models.html)

# OpenAI #######################################
# 크로마는 OpenAI의 임베딩 API에 대한 편리한 래퍼를 제공합니다. 이 임베딩 기능은 OpenAI의 서버에서 원격으로 실행되며 API 키가 필요합니다.
# OpenAI에 계정을 등록하면 API 키를 받을 수 있습니다.
# 이 임베딩 기능은 pip install openai로 설치할 수 있는 openai 파이썬 패키지에 의존합니다.
import config, os, pprint

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
)
val = openai_ef("foo")
print(len(val), len(val[0]))  # (3,1536) : 1536 차원 벡터, char 단위 임베딩
val = openai_ef(["bare", "foo"])
print(len(val), len(val[0]))  # (2,1536) : word 단위 임베딩 방법
val = openai_ef("한국어는 어떨까")
print(len(val), len(val[0]))  # (8,1536) : char 단위 임베딩 방법
val = openai_ef(["한국어는", "어떨까"])
print(len(val), len(val[0]))  # (2,1536) : word 단위 임베딩 방법

# Azure와 같은 다른 플랫폼에서 OpenAI 임베딩 모델을 사용하려면 api_base 및 api_type 매개 변수를 사용할 수 있습니다:
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_API_KEY",
    api_base="YOUR_API_BASE_PATH",
    api_type="azure",
    model_name="text-embedding-ada-002",
)
# 선택적 model_name 인수를 전달하여 사용할 OpenAI 임베딩 모델을 선택할 수 있습니다.
# 기본적으로 Chroma는 text-embedding-ada-002를 사용합니다. 사용 가능한 모든 모델 목록은 여기에서 확인할 수 있습니다.
# 임베딩 모델 : https://platform.openai.com/docs/guides/embeddings/what-are-embeddings


# Cohere ###################################################
# 크로마는 Cohere의 임베딩 API에 대한 편리한 래퍼도 제공합니다.
# 이 임베딩 기능은 Cohere의 서버에서 원격으로 실행되며 API 키가 필요합니다. Cohere에 계정을 등록하면 API 키를 받을 수 있습니다.
# 이 임베딩 기능은 파이썬 패키지를 사용하며, 파이썬 패키지는 pip install cohere로 설치할 수 있습니다.
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="YOUR_API_KEY", model_name="large"
)
cohere_ef(texts=["document1", "document2"])

# 선택적으로 model_name 인수를 전달하여 사용할 Cohere 임베딩 모델을 선택할 수 있습니다.
# 기본적으로 크로마는 대형 모델을 사용합니다. 사용 가능한 모델은 여기에서 임베딩 가져오기 섹션에서 확인할 수 있습니다.
# https://docs.cohere.com/reference/embed

# Multilingual model example (다국어 지원 예제)
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="YOUR_API_KEY", model_name="multilingual-22-12"
)

multilingual_texts = [
    "Hello from Cohere!",
    "مرحبًا من كوهير!",
    "Hallo von Cohere!",
    "Bonjour de Cohere!",
    "¡Hola desde Cohere!",
    "Olá do Cohere!",
    "Ciao da Cohere!",
    "您好，来自 Cohere！",
    "कोहेरे से नमस्ते!",
]

cohere_ef(texts=multilingual_texts)
# 다국어 모델에 대한 자세한 내용은 여기에서 확인할 수 있습니다.
# https://docs.cohere.com/docs/multilingual-language-models


# Instructor models ##########################################
# Instructor-embeddings 라이브러리는 특히 Cuda 지원 GPU가 탑재된 컴퓨터에서 실행할 때 사용할 수 있는 또 다른 옵션입니다.
# 이 라이브러리는 OpenAI를 대체할 수 있는 좋은 로컬 대안입니다(대용량 텍스트 임베딩 벤치마크 순위 참조).
# 임베딩 기능을 사용하려면 InstructorEmbedding 패키지가 필요합니다. 설치하려면 pip install InstructorEmbedding을 실행합니다.
# 세 가지 모델을 사용할 수 있습니다. 기본값은 hkunlp/instructor-base이며, 더 나은 성능을 위해 hkunlp/instructor-large 또는 hkunlp/instructor-xl을 사용할 수 있습니다.
# CPU(기본값)를 사용할지 또는 Cuda를 사용할지 지정할 수도 있습니다.
# Base model and cpu
ef = embedding_functions.InstructorEmbeddingFunction()
# GPU 모델
ef = embedding_functions.InstructorEmbeddingFunction(
    model_name="hkunlp/instructor-xl", device="cuda"
)
# large 모델과 xl 엘 모델은 각각 1.5GB와 5GB로, GPU에서 실행하는 데 가장 적합


# Google PaLM API 모델 ########################################################
# Google PaLM API는 현재 비공개 미리 보기 중이지만,
# 이 미리 보기에 참여 중인 경우 GooglePalmEmbeddingFunction을 통해 Chroma에서 사용할 수 있습니다.
# PaLM 임베딩 API를 사용하려면 google.generativeai Python 패키지가 설치되어 있고 API 키가 있어야 합니다.
palm_embedding = embedding_functions.GooglePalmEmbeddingFunction(
    api_key=api_key, model=model_name
)

# 허깅페이스 (HuggingFace) #######################################################
# 크로마는 HuggingFace의 임베딩 API에 대한 편리한 래퍼도 제공합니다.
# 이 임베딩 기능은 HuggingFace의 서버에서 원격으로 실행되며 API 키가 필요합니다. API 키는 HuggingFace에 계정을 등록하여 얻을 수 있습니다.
# 이 임베딩 기능은 pip 설치 요청으로 설치할 수 있는 요청 파이썬 패키지에 의존합니다.

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.environ["HUGGINGFACE_API_KEY"],
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
# 선택적 model_name 인수를 전달하여 사용할 허깅페이스 모델을 선택할 수 있습니다.
# 기본적으로 Chroma는 sentence-transformers/all-MiniLM-L6-v2를 사용합니다.
# 사용 가능한 모든 모델 목록은 여기에서 확인할 수 있습니다. (https://huggingface.co/models)

val = huggingface_ef("foo")
print(len(val))  # 384 차원 : word 단위 임베딩
val = huggingface_ef(["bare", "foo"])
print(len(val), len(val[0]))  # (2,384) : word 단위 임베딩 방법
val = huggingface_ef("한국어는 어떨까")
print(len(val))  # (384) : 문장 단위 임베딩 방법
val = huggingface_ef(["한국어는", "어떨까"])
print(len(val), len(val[0]))  # (2,384) : word 단위 임베딩 방법


# Custom Embedding 함수 ################################################################
# 임베딩함수 프로토콜을 구현하기만 하면 자신만의 임베딩 함수를 만들어 크로마와 함께 사용할 수 있습니다.
from chromadb import Documents, EmbeddingFunction, Embeddings


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings
