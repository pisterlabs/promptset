import openai
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.config import settings
from app.schemas.user_schema import IUserCreate
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient, models
import pandas as pd

openai.api_key = settings.OPENAI_API_KEY


async def init_db(db_session: AsyncSession) -> None:
    input_datapath = "app/data/fine_food_reviews_with_embeddings_1k.csv"  # to save space, we provide a pre-filtered dataset
    df = pd.read_csv(input_datapath, index_col=0)

    is_cloud_qdrant = True
    qdrant_client = (
        QdrantClient(
            url=settings.QDRANT_CLOUD_URL, api_key=settings.QDRANT_CLOUD_API_KEY
        )
        if is_cloud_qdrant
        else QdrantClient(host=settings.QDRANT_HOST)
    )

    try:
        qdrant_client.recreate_collection(
            collection_name="my_docs",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        records = []
        for idx, row in df.iterrows():
            vector = [float(i) for i in row["embedding"].strip("[]").split(",")]
            records.append(
                models.Record(
                    id=idx,
                    vector=vector,
                    payload={
                        "product_id": row["ProductId"],
                        "user_id": row["UserId"],
                        "score": row["Score"],
                        "summary": row["Summary"],
                        "page_content": row["Text"],
                        "metadata": None,
                        "n_tokens": row["n_tokens"],
                    },
                )
            )

        qdrant_client.upload_records(
            collection_name="my_docs",
            records=records,
        )

    except Exception as e:
        print(e)
