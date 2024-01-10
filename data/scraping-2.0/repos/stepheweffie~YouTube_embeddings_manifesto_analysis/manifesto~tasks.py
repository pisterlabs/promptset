from manifesto_openai_embeddings import openai_pdf_embeddings
import d6tflow
import pandas as pd


# The first two embeddings tasks use OpenAI embeddings-ada-002 model

class SingleVideoEmbeddingsTask(d6tflow.tasks.TaskPickle):
    def run(self):
        # Currently subprocess must be called on the file prior; single_video_openai_embeddings.py
        video_embeddings = pd.read_pickle('data/single_video_openai_embeddings.pkl')
        df = pd.DataFrame()
        print(video_embeddings)
        data = {'video_embeddings': df}
        self.save(data)


class PDFEmbeddingsTask(d6tflow.tasks.TaskPickle):
    def run(self):
        pdf_embeddings = openai_pdf_embeddings()
        df = pd.DataFrame(pdf_embeddings)
        data = {'pdf_embeddings': df}
        self.save(data)


flow = d6tflow.Workflow()
flow.run(SingleVideoEmbeddingsTask)