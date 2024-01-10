from utils import Time
from langchain.llms import huggingface_hub # REPLACE IF NOT USING HUGGINGFACE

llm = huggingface_hub.HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                     model_kwargs={"temperature":1.2})


# bedtimes = {
#     "awake": Time(11, 0),
#     "tired": Time(10, 0)
# }


playlists = {
    "awake-sad": ["https://open.spotify.com/album/4SZko61aMnmgvNhfhgTuD3?si=I0FJLldhT3ConRwn1S8nZA", "https://open.spotify.com/album/2ua5bFkZLZl1lIgKWtYZIz?si=c38c73fa1aa1405d"],
    "awake-neutral": ["https://open.spotify.com/album/2Ek1q2haOnxVqhvVKqMvJe?si=cb03ea674a47404a"],
    "awake-happy": ["https://open.spotify.com/album/0rwbMKjNkp4ehQTwf9V2Jk?si=oiAhTa0pRyO9Pvn9mvfS4A", "https://open.spotify.com/album/18NOKLkZETa4sWwLMIm0UZ?si=d6a8c16d63204f80"],
    "tired-sad": ["https://open.spotify.com/album/4SZko61aMnmgvNhfhgTuD3?si=I0FJLldhT3ConRwn1S8nZA"],
    "tired-neutral": ["https://open.spotify.com/playlist/37i9dQZF1EIhpG2qPPV1Lr?si=6b6bef2a363044e6"],
    "tired-happy": ["https://open.spotify.com/playlist/37i9dQZF1EIhpG2qPPV1Lr?si=6b6bef2a363044e6"]
}

labels = ["awake-sad", "awake-neutral", "awake-happy", "tired-sad", "tired-neutral", "tired-happy"]