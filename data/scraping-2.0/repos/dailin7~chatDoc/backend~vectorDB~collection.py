from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from backend.sources import qdrant_client, embedding
from qdrant_client.models import Distance, VectorParams
from ..utils import load_files
from qdrant_client.http.exceptions import UnexpectedResponse
from backend.utils import load_files, upsert_documents_to_qdrant
from backend.service.collection import add_filenames
from langchain.vectorstores import Qdrant

# test comment


@DeprecationWarning
@api_view(["POST"])
def create_collection(request):
    try:
        collection_name = request.POST["collection_name"]
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        return Response(status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_names(request):
    try:
        collection_res = qdrant_client.get_collections()
        res = [x.name for x in collection_res.collections]
        return Response(res, status=status.HTTP_200_OK)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_collection(request, collection_name: str):
    try:
        collection = qdrant_client.get_collection(collection_name=collection_name)
        return Response(collection, status=status.HTTP_200_OK)
    except UnexpectedResponse as e:
        return Response(
            {"message": f"Collection '{collection_name}' does not exists!"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["DELETE"])
def delete_collection(request, collection_name: str):
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        return Response(status=status.HTTP_200_OK)
    except UnexpectedResponse:
        return Response(
            {"message": f"Collection '{collection_name}' does not exists!"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def upload_files(request, collection_name: str):
    try:
        qdrant = Qdrant(
            client=qdrant_client, collection_name=collection_name, embeddings=embedding
        )
    except UnexpectedResponse:
        return Response(
            {"message": f"Collection '{collection_name}' does not exists!"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # TODO: change duplicate files and ask for overwrite confirmation
    files = request.FILES.getlist("files")
    filenames = [x.name for x in files]

    # add filenames to collection model
    add_filenames(collection_name, filenames)

    # add files to vector db

    pages = load_files(files)

    qdrant.add_documents(pages)

    try:
        upsert_documents_to_qdrant(pages, collection_name=collection_name)
        return Response(status=status.HTTP_200_OK)
    except Exception as e:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    # reload qa chain
