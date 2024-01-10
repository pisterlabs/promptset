import os
import numpy as np
import gradio as gr
from FastSam_segmentation import segment_image
from llamaindex_bm25_baseline import ClassicRetrieverBM25
from pill_info_extraction import gpt_ocr
from upload_images_vector_store import image_similarity_activeloop

from utils import get_index_and_nodes_after_visual_similarity, load_vector_store
from global_variable import (
    VECTOR_STORE_PATH_IMAGES_MASKED,
    VECTOR_STORE_PATH_IMAGES_NORMAL,
    VECTOR_STORE_PATH_DESCRIPTION,
)
from PIL import Image
import cv2
from llama_index.retrievers import BM25Retriever
import urllib.parse

VESTOR_STORE_IMAGES_MASKED = None
VESTOR_STORE_IMAGES_NORMAL = None
VECTOR_STORE_DESCRIPTION = None


def search_from_image(input_image):
    print("----------- starting search from image -----------")
    iframe_html = '<iframe src={url} width="570px" height="400px"/iframe>'

    iframe_url = f"https://app.activeloop.ai/visualizer/iframe?url={VECTOR_STORE_PATH_IMAGES_NORMAL}&query="

    global VESTOR_STORE_IMAGES_MASKED
    global VESTOR_STORE_IMAGES_NORMAL
    global VECTOR_STORE_DESCRIPTION
    desc = "Description pill "
    sd_eff = "Side-effects for the pill "
    if not VESTOR_STORE_IMAGES_MASKED:
        VESTOR_STORE_IMAGES_MASKED = load_vector_store(
            VECTOR_STORE_PATH_IMAGES_MASKED
        ).vectorstore
    if not VECTOR_STORE_DESCRIPTION:
        VECTOR_STORE_DESCRIPTION = load_vector_store(
            VECTOR_STORE_PATH_DESCRIPTION
        ).vectorstore
    # USED FOR NORMAL IMAGES RESEARCH
    # if not VESTOR_STORE_IMAGES_NORMAL:
    #     VESTOR_STORE_IMAGES_NORMAL = load_vector_store(
    #         VECTOR_STORE_PATH_IMAGES_NORMAL
    #     ).vectorstore
    os.makedirs("./test", exist_ok=True)

    image_path = "./test/input_image.png"
    image_path_masked = f"./test/{image_path.split('.')[0]}_masked.png"
    im = Image.fromarray(input_image)
    im.save(image_path)
    pill_text_extracted = gpt_ocr(image_path)
    # cv2.imwrite(image_path, input_image)

    # MASK IMAGE
    image_masked = segment_image([image_path], test=True)
    image_masked = image_masked[0]
    image_masked_pil = Image.fromarray(image_masked)
    image_masked_pil.save(image_path_masked)
    cv2.imwrite(image_path_masked, image_masked)

    # VISUAL SIMILARITY
    # similar_images = image_similarity_activeloop(VESTOR_STORE_IMAGES_NORMAL, image_path)
    similar_images_after_segmentation = image_similarity_activeloop(
        VESTOR_STORE_IMAGES_MASKED, image_path_masked
    )

    # check if the code of the input image is found in the retrieved similar images
    for el in similar_images_after_segmentation:
        if el["metadata"]["pill_text"] == pill_text_extracted:
            similar_images_after_segmentation.pop(
                similar_images_after_segmentation.index(el)
            )
            similar_images_after_segmentation.insert(0, el)

    filename_similar_images_to_retrieve_from_description_db = []
    for el in similar_images_after_segmentation["filename"]:
        filename_similar_images_to_retrieve_from_description_db.append(el)
        break

    (
        _,
        nodes,
        _,
        filtered_elements,
    ) = get_index_and_nodes_after_visual_similarity(
        filename_similar_images_to_retrieve_from_description_db
    )  # node_0 is related to filtered_elements[0], ...

    # EXCLUDE THE 3 MOST SIMILAR IMAGES
    most_similar_3_images_filenames = [
        similar_images_after_segmentation["filename"][0],
        similar_images_after_segmentation["filename"][1],
        similar_images_after_segmentation["filename"][2],
    ]
    most_similar_3_node_ids = [
        filtered_elements["filename"].index(filename)
        for filename in most_similar_3_images_filenames
    ]
    nodes = [el for idx, el in enumerate(nodes) if idx not in most_similar_3_node_ids]
    # DESCRIPTION SIMILARITY
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    hybrid_only_bm25_retriever = ClassicRetrieverBM25(bm25_retriever)
    # most similar image (visually)
    id_most_similar = filtered_elements["filename"].index(
        similar_images_after_segmentation["filename"][0]
    )

    # description or the most similar image ==> use the id as key to retrieve the description
    description = filtered_elements["text"][id_most_similar]
    # ==> return sorted nodes based on description similarity
    nodes_bm25_response = hybrid_only_bm25_retriever.retrieve(description)

    # take the first 3 elements (most similar) given the description
    most_similar_id = most_similar_3_node_ids
    most_similar_images_filenames_metadata = [
        [
            filtered_elements["filename"][el],
            filtered_elements["metadata"][el],
            filtered_elements["text"][el],
        ]
        for el in most_similar_id
    ]
    output_name1 = most_similar_images_filenames_metadata[0][1]["name"]
    output_name2 = most_similar_images_filenames_metadata[1][1]["name"]
    output_name3 = most_similar_images_filenames_metadata[2][1]["name"]
    output_side_effects1 = most_similar_images_filenames_metadata[0][1]["side-effects"]
    output_side_effects2 = most_similar_images_filenames_metadata[1][1]["side-effects"]
    output_side_effects3 = most_similar_images_filenames_metadata[2][1]["side-effects"]
    output_description1 = most_similar_images_filenames_metadata[0][2]
    output_description2 = most_similar_images_filenames_metadata[1][2]
    output_description3 = most_similar_images_filenames_metadata[2][2]

    # most dissimilar element
    most_dissimilar_ids = nodes_bm25_response[-3:]
    most_dissimilar_id = [
        int(el.node_id.split("_")[1]) for el in most_dissimilar_ids
    ]  # i.e. from node_1, node_5, node_9 to 1, 5, 9
    most_dissimilar_images_filenames_metadata = [
        [
            filtered_elements["filename"][el],
            filtered_elements["metadata"][el],
            filtered_elements["text"][el],
        ]
        for el in most_dissimilar_id
    ]
    output_name4 = most_dissimilar_images_filenames_metadata[0][1]["name"]
    output_name5 = most_dissimilar_images_filenames_metadata[1][1]["name"]
    output_name6 = most_dissimilar_images_filenames_metadata[2][1]["name"]
    output_side_effects4 = most_dissimilar_images_filenames_metadata[0][1][
        "side-effects"
    ]
    output_side_effects5 = most_dissimilar_images_filenames_metadata[1][1][
        "side-effects"
    ]
    output_side_effects6 = most_dissimilar_images_filenames_metadata[2][1][
        "side-effects"
    ]
    output_description4 = most_dissimilar_images_filenames_metadata[0][2]
    output_description5 = most_dissimilar_images_filenames_metadata[1][2]
    output_description6 = most_dissimilar_images_filenames_metadata[2][2]

    # use the filename as key to retrieve the image and description given the most similar and dissimilar from the description
    filename_for_visualizer = [
        el[0]
        for el in most_similar_images_filenames_metadata
        + most_dissimilar_images_filenames_metadata
    ]
    id_most_similar_images = [
        similar_images_after_segmentation["filename"].index(el[0])
        for el in most_similar_images_filenames_metadata
    ]
    id_most_dissimilar_images = [
        similar_images_after_segmentation["filename"].index(el[0])
        for el in most_dissimilar_images_filenames_metadata
    ]

    most_similar_images = [
        similar_images_after_segmentation["image"][el] for el in id_most_similar_images
    ]
    most_dissimilar_images = [
        similar_images_after_segmentation["image"][el]
        for el in id_most_dissimilar_images
    ]
    images = [
        Image.fromarray(el) for el in most_similar_images + most_dissimilar_images
    ]
    query = "select image where filename == "
    # queries = [f"{query}'{el}'" for el in filename_for_visualizer]  # masked images
    queries = [
        f"""{query}'images/{el.split("/")[1].split("_masked")[0]}.jpg'"""
        for el in filename_for_visualizer
    ]  # normal images
    urls = [iframe_url + urllib.parse.quote(el) for el in queries]
    # url = iframe_url + urllib.parse.quote(query)
    # html = iframe_html.format(url=url)
    htmls = [iframe_html.format(url=url) for url in urls]

    return (
        htmls[0],
        htmls[1],
        htmls[2],
        htmls[3],
        htmls[4],
        htmls[5],
        gr.Textbox(label=f"{desc} {output_name1}", value=output_description1),
        gr.Textbox(label=f"{desc} {output_name2}", value=output_description2),
        gr.Textbox(label=f"{desc} {output_name3}", value=output_description3),
        gr.Textbox(label=f"{desc} {output_name4}", value=output_description4),
        gr.Textbox(label=f"{desc} {output_name5}", value=output_description5),
        gr.Textbox(label=f"{desc} {output_name6}", value=output_description6),
        gr.Textbox(label=f"{sd_eff} {output_name1}", value=output_side_effects1),
        gr.Textbox(label=f"{sd_eff} {output_name2}", value=output_side_effects2),
        gr.Textbox(label=f"{sd_eff} {output_name3}", value=output_side_effects3),
        gr.Textbox(label=f"{sd_eff} {output_name4}", value=output_side_effects4),
        gr.Textbox(label=f"{sd_eff} {output_name5}", value=output_side_effects5),
        gr.Textbox(label=f"{sd_eff} {output_name6}", value=output_side_effects6),
    )


with gr.Blocks(title="Pill Search") as demo:
    gr.Markdown("# Compute the similarity between pills.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Upload a pill image.")
            image_input = gr.Image()
            image_button = gr.Button("Compute similarity")
        with gr.Column(scale=4):
            gr.Markdown("Most similar images:")
            with gr.Row():
                with gr.Column(scale=1):
                    # image_output1 = gr.Image()
                    image_output1 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc1 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects1 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output2 = gr.Image()
                    image_output2 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc2 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects2 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output3 = gr.Image()
                    image_output3 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc3 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects3 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
        with gr.Column(scale=4):
            gr.Markdown("Do not confuse with the following pills:")
            with gr.Row():
                with gr.Column(scale=1):
                    # image_output4 = gr.Image()
                    image_output4 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc4 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects4 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output5 = gr.Image()
                    image_output5 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc5 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects5 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output6 = gr.Image()
                    image_output6 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc6 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects6 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )

    image_button.click(
        search_from_image,
        inputs=image_input,
        outputs=[
            image_output1,
            image_output2,
            image_output3,
            image_output4,
            image_output5,
            image_output6,
            desc1,
            desc2,
            desc3,
            desc4,
            desc5,
            desc6,
            side_effects1,
            side_effects2,
            side_effects3,
            side_effects4,
            side_effects5,
            side_effects6,
        ],
    )

demo.launch()
