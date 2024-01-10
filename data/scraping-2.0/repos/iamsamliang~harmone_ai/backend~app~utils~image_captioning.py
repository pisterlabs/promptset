from transformers import pipeline
from datasets import load_dataset
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator


# Legacy
def get_captions(image_dir: str, device: str):
    """Convert the frames in image_dir to captions

    Args:
        image_dir (str): directory path
        device (str): "cuda", "cpu", or "mps"

    Returns:
        list[str]: list of captions in sequential order
    """
    # directory = "frames"
    # image_paths = [
    #     os.path.join(directory, file)
    #     for file in sorted(os.listdir(directory))  # sorting is necessary for correct times
    #     if file.endswith(("jpg", "jpeg", "png"))
    # ]

    dataset = load_dataset("imagefolder", data_dir=image_dir)
    dataset = dataset["train"]["image"]  # type: ignore
    pipe = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base", device=device
    )
    results = pipe(dataset)
    # data processing
    captions = [item["generated_text"] for result in results for item in result]
    return captions


# loader = ImageCaptionLoader(path_images=image_paths)
# list_docs = loader.load()
# print(list_docs)

# index = VectorstoreIndexCreator().from_loaders([loader])
# print(index)

# query = "What is happening in the image?"
# print(index.query(query))
