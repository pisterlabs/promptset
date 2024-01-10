import os
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector
import tempfile
import logging
import sys
import time
import psycopg2

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger.addHandler(handler)

s3_client = boto3.client('s3')

# Embedding モデルの指定
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")


def lambda_handler(event, context):
    try:
        logger.info("Lambda function invoked")

        # S3イベントからファイル情報を取得
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        logger.info(f"File from S3: bucket_name: {bucket_name}, file_name: {file_key}")

        # S3からファイルを読み込む
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = file_obj['Body'].read()
        logger.info(f"Successfully read {file_key} from S3")

        # 一時ファイルにPDFを保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        logger.info(f"Saved {file_key} into temp file path: {temp_file_path}")

        documents = PyPDFLoader(temp_file_path).load()
        logger.info(f"Loaded {file_key} using PyPDFLoader")

        # delete temp file
        os.remove(temp_file_path)
        logger.info(f"Deleted temp file: {temp_file_path}")

        # ドキュメントを分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        # log how many docs were splitted into
        logger.info(f"Splitted into {len(docs)} documents")

        # DB 接続情報
        driver = "psycopg2"
        host = os.environ.get("PGVECTOR_HOST")
        port = "5432"
        database = "postgres"
        user = "postgres"
        password = os.environ.get("PGVECTOR_PASSWORD")

        # Langchain 側はなぜか f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}" になっているのであえて別で作る
        CONNECTION_STRING = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # DB に pgvector が入ってない場合インストール
        conn = psycopg2.connect(CONNECTION_STRING)
        cur = conn.cursor()

        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        extension_exists = cur.fetchone()

        if not extension_exists:
            cur.execute("CREATE EXTENSION vector;")
            conn.commit()
            logger.info("Created vector extension")

        cur.close()
        conn.close()

        # DB 接続情報
        CONNECTION_STRING_EMBEDDING = PGVector.connection_string_from_db_params(
            driver=driver,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

        # Collection Name は後ほど一般的な命名にする
        COLLECTION_NAME = "bedrock_documents"

        # ベクトル化と保存にかかった時間を計測
        start_time = time.time()

        # データベースに保存
        db = PGVector.from_documents(
            embedding=bedrock_embeddings,
            documents=docs,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING_EMBEDDING
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Took {elapsed_time:.1f} seconds for embedding and saving {len(docs)} documents to Postgres")

        return {
            'statusCode': 200,
            'body': f'{file_key} processed and stored successfully'
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error processing {file_key}'
        }
