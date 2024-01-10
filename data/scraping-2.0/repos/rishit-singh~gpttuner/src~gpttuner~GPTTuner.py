import openai

class TuningJob:
    def __init__(self, _id: str, trainingFileID: str):
        self.ID: str = _id
        self.TrainingFileID: str = trainingFileID
    def ToDict(self):
        return 

class GPTTuner:
    def FetchFile(self, fileId: str) -> dict:   
        file: dict = None

        try:
            file = openai.File.retrieve(fileId)
        except:
            pass

        return file

    def ListFiles(self, maxItems: int = 10) -> dict:
        return openai.File.list(limit=maxItems)
    
    def ListJobs(self, maxItems: int = 10) -> dict:
        return openai.FineTuningJob.list(limit=maxItems)

    def FileExists(self, filePath: str) -> bool:
        return (len(openai.File.find_matching_files(filePath, open(filePath, "rb").read(), "fine-tune")) > 0)

    def CreateTrainingFile(self, filePath: str) -> dict:
        response: dict = {}

        if (self.FileExists(filePath)):
            raise Exception(f"File {filePath} already exists.")

        try:
            response = openai.File.create(file=open(filePath, "rb"))
        except Exception as e:
            print(e)

        return response

    def CreateJob(self, trainingFile: str, id: bool = False) -> dict[str, str]:
        response: dict = openai.FineTuningJob.create(model=self.Model, training_file=trainingFile)

        return response

    def CancelJob(self, jobID: str) -> bool:
        response: dict = openai.FineTuningJob.cancel(jobID)

        return response

    def __init__(self, apiKey: str, model: str = "gpt3.5-turbo"):
        self.Model: str = model
        self.APIKey: str = apiKey
        openai.api_key = self.APIKey
