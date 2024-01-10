import boto3
import os
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io
from langchain import ConversationChain
import secrets
import time
import hashlib
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from app_utils import lambda_app_logger

logger = lambda_app_logger.get_logger()

# 画像生成の設定
IMG_PROMPT_PREFIX = "anime style, super fine illustration, girl"
NEGATIVE_PROMPT = "flat color, flat shading, nsfw,retro style, poor quality"
BUCKET_NAME = "univac-gpt-profile-imgs"

# S3の署名付きURLの有効期限
IMG_EXPIRATION = 3600 * 24 * 7

# ChatAIプロンプトの設定
OPENAI_SYSTEM_SETTING = """
画像生成のテンプレート文
"""

CHATAI_SETTINGS = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.2,
}
# TODO 設定値を環境変数へ
# TODO プロンプトの設定文を切り離す


def generate_random_key(length=8):
    unique_data = str(time.time()).encode("utf-8") + secrets.token_bytes(length)
    key_hash = hashlib.sha256(unique_data).hexdigest()
    return key_hash[:length]


def make_img_prompt_by_openai(profile):
    logger.info("Set ChatAI", extra={"settings": CHATAI_SETTINGS})
    # チャットモデル
    llm = ChatOpenAI(
        model_name=CHATAI_SETTINGS["model_name"],
        temperature=CHATAI_SETTINGS["temperature"],
    )

    # チャットプロンプトテンプレート
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(OPENAI_SYSTEM_SETTING),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    # メモリ
    memory = ConversationBufferWindowMemory(k=1, return_messages=True)

    # 会話チェーン
    conversation = ConversationChain(
        memory=memory, prompt=prompt, llm=llm, verbose=False
    )

    logger.info("Request to ChatAI", extra={"profile": profile})
    tmp_result = conversation.predict(input=profile)
    logger.info("Response from ChatAI", extra={"response": tmp_result})

    result = "{}, {}".format(IMG_PROMPT_PREFIX, tmp_result)

    return result


def upload_img_to_s3(user_id, s3_key, img_data):
    # boto3 クライアントを作成
    s3 = boto3.client("s3")

    # ユーザIDをプレフィックスとして使用
    key_with_prefix = f"{user_id}/{s3_key}"

    logger.info(
        "Upload to S3 start", extra={"bucket": BUCKET_NAME, "key": key_with_prefix}
    )
    # ローカルファイルをS3にアップロード
    s3.upload_fileobj(img_data, BUCKET_NAME, key_with_prefix)

    logger.info(
        "Upload to S3 completed", extra={"bucket": BUCKET_NAME, "key": key_with_prefix}
    )
    img_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key_with_prefix},
        ExpiresIn=IMG_EXPIRATION,
    )
    return img_url


def make_img_from_prompt(user_id, prompt_message):
    # APIインタフェースの準備
    stability_api = client.StabilityInference(
        key=os.environ["STABILITY_KEY"],
        verbose=True,
    )

    logger.info("Request to STABLE API", extra={"request": prompt_message})

    # テキストからの画像生成
    answers = stability_api.generate(
        [
            generation.Prompt(
                text=prompt_message, parameters=generation.PromptParameters(weight=1)
            ),
            generation.Prompt(
                text=NEGATIVE_PROMPT, parameters=generation.PromptParameters(weight=-1)
            ),
        ]
    )

    results = []

    # 結果の出力
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                print("NSFW")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img_data = io.BytesIO(artifact.binary)
                object_key = "img_{}.jpg".format(generate_random_key())
                # 画像をS3にアップロード
                img_url = upload_img_to_s3(user_id, object_key, img_data)
                results.append({"img_url": img_url})
    return results


def make_img(user_id, profile):
    prompt = make_img_prompt_by_openai(profile)
    results = make_img_from_prompt(user_id, prompt)
    # 画像は1枚のみ
    return results[0]
