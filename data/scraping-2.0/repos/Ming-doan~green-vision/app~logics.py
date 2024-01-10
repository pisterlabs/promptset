from .utils import db, storage
from typing import Any
from .yolo import yolo
from PIL import Image
from torch import Tensor
import openai
import os


def predict_pil_image(image: Image) -> list[dict[str, Any]]:
    # Predict image
    results = yolo.predict(image, 0.25)
    result_format = list(results)[0]

    # Get results
    boxs: Tensor = result_format.boxes.data
    labels = result_format.names

    # Get boxs result
    boxs_result = []
    for box in boxs:
        print(box)
        result_conf = box[4].item()
        if result_conf > 0.25:
            # Get the center point
            center_x = box[0].item()
            center_y = box[1].item()

            boxs_result.append({
                "pos": [int(center_x), int(center_y)],
                "name": labels[int(box[5].item())],
            })

    return boxs_result


def query_from_firebase(label: str) -> list[dict[str, Any]]:
    # Get collections
    trash_ref = db.collection("Trash")
    recycleDoc_ref = db.collection("RecycleDoc")

    found_trashs = trash_ref.where("name", "==", label).get()

    recommends = []

    if len(found_trashs) != 0:
        # Get list of recycle ids
        recycle_ids = found_trashs[0].get("recycleID")

        for recycle_id in recycle_ids:
            # Get recycle doc
            recycle_doc = recycleDoc_ref.document(recycle_id).get()

            # Get paths
            paths = []
            for path in recycle_doc.get('path'):
                if path != '':
                    print(path)
                    # Split path
                    bucket_name, object_path = path[len(
                        "gs://"):].split("/", 1)
                    # Convert to blob
                    bucket = storage.bucket(bucket_name)
                    blob = bucket.blob(object_path)
                    # Generate signed url
                    paths.append(blob.generate_signed_url(
                        expiration=3000000000))

            # Append to recommends
            recommends.append({
                'content': recycle_doc.get("content"),
                'path': paths,
                'title': recycle_doc.get("title")
            })

    return recommends


openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_response(prompt: str):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )

    return response
