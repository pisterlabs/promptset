import streamlit as st
import clip
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import skimage.io as io
from tqdm import tqdm
import cv2

def main():
    st.set_page_config(
        layout="wide", page_title="q-CLIP", initial_sidebar_state="expanded"
    )
    dataset = st.sidebar.selectbox("Choose a dataset", ["ImageNetV2", "Coco", "DeepDrive"])
    top_k = st.sidebar.slider("How many images do you want to search for?", 1, 10, 10)
    st.sidebar.markdown("---------------------")
    st.sidebar.markdown(
        f"Test out [CLIP](https://openai.com/blog/clip) from OpenAI on various datasets."
    )
    model, transform = load_model()

    # standardize directory/file names...
    if dataset == "ImageNetV2":
        image_features = torch.load("./imagenetv2/imagenetv2_matched_freq_features.pt")
        with open("./imagenetv2/imagenetv2_matched_freq_filenames.txt", "rb") as f:
            fnames = pickle.load(f)
    elif dataset == "Coco":
        image_features = torch.load("./coco/coco_features.pt")
        with open("./coco/coco_filenames.txt", "rb") as f:
            fnames = pickle.load(f)
    elif dataset == "DeepDrive":
        image_features = torch.load("./bdd/bdd_features.pt")
        with open("./bdd/bdd_filenames.txt", "rb") as f:
            fnames = pickle.load(f)

    run_app(dataset, image_features, fnames, model, transform, top_k)


def run_app(dataset, image_features, fnames, model, transform, top_k=10, device="cpu"):
    text_input = st.text_area(
        "Insert text query here. [Command + Enter] to search.",
        value="This is an example query. Replace it with something more interesting!"
    )

    sim = torch.zeros(image_features.shape[0])
    with torch.no_grad():
        text_tok = clip.tokenize([text_input]).to(device)
        text_features = model.encode_text(text_tok)
        sim = cosine_similarity(text_features, image_features)
    im_vals, im_inds = torch.topk(sim, k=top_k)


    values, images, inds = im_vals.numpy()[:, None].T, [], []
    for i in im_inds:
        if dataset == "ImageNetV2":
            f = "./imagenetv2/imagenetv2-matched-frequency-format-val/" + "/".join(fnames[i].split('/')[1:])
            inds.append(int((f.split('/'))[3]))
        elif dataset == "Coco":
            f = "http://images.cocodataset.org/train2014/" + fnames[i].split('/')[2]
            inds.append(i)
        elif dataset == "DeepDrive":
            f = './bdd/' + '/'.join(fnames[i].split('/')[2:])
            inds.append(i)
        images.append(io.imread(f))

    fig = px.histogram(sim.cpu().numpy(), title='Similarity Distribution')
    st.plotly_chart(fig, use_container_width=True)
    st.write('Most Similar to Least Similar')
    plot_pics(images, values, top_k)
    # plot_plt(dataset, values, images, inds)
    plot_pca(image_features[im_inds])

def plot_pics(images, values, top_k):
    values = values.flatten()
    images = [cv2.resize(im, (360, 360)) for im in images]

    c1, c2, c3, c4, c5 = st.beta_columns(5)
    with c1:
        st.image(images[0], caption=f"Similarity score: {values[0]}")
        if top_k > 5: st.image(images[5], caption=f"Similarity score: {values[5]}")
    with c2:
        if top_k > 1: st.image(images[1], caption=f"Similarity score: {values[1]}")
        if top_k > 6: st.image(images[6], caption=f"Similarity score: {values[6]}")
    with c3:
        if top_k > 2: st.image(images[2], caption=f"Similarity score: {values[2]}")
        if top_k > 7: st.image(images[7], caption=f"Similarity score: {values[7]}")
    with c4:
        if top_k > 3: st.image(images[3], caption=f"Similarity score: {values[3]}")
        if top_k > 8: st.image(images[8], caption=f"Similarity score: {values[8]}")
    with c5:
        if top_k > 4: st.image(images[4], caption=f"Similarity score: {values[4]}")
        if top_k > 9: st.image(images[9], caption=f"Similarity score: {values[9]}")

def plot_plt(dataset, values, images, inds):

    if dataset == "ImageNetV2":
        with open("./imagenetv2/imagenetv2_matched_freq_classes.txt", "rb") as f:
            classes = pickle.load(f)
    elif dataset == "Coco":
        with open("./coco/coco_all_captions.txt", "rb") as f:
            classes = pickle.load(f)
        classes = [descrip[0] for descrip in classes]

    count = values.shape[1]
    plt.figure(figsize=(20, 14))
    plt.imshow(values, vmin=0.0, vmax=1.0)
    # plt.xticks(range(count), [classes[ind] for ind in inds], fontsize = 10)
    plt.yticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(count):
        for y in range(1):
            plt.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([0.5, -1.6])

    plt.title("Cosine similarity between text and image features", size=20)
    with st.beta_container():
        st.pyplot(plt)


def plot_pca(x):
    sc = StandardScaler()
    x_norm = sc.fit_transform(x)
    pca = PCA(n_components=3)
    components = pca.fit_transform(x_norm)

    var = pca.explained_variance_ratio_.sum()
    fig = px.scatter_3d(components, x=0, y=1, z=2, title=f'Total Explained Variance: {var}',
    labels={'0':'PC1', '1':'PC2', '2':'PC3'})

    st.plotly_chart(fig, use_container_width=True)


@st.cache(allow_output_mutation=True)
def load_model():
    model, transform = clip.load("ViT-B/32", "cpu")
    return model, transform


if __name__ == "__main__":
    main()
