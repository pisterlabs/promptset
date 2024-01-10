from langflow import CustomComponent
from langchain.schema import Document
from VideoAnalysisTool import extract_metadata, extract_frames

class VideoAnalysisTool(CustomComponent):
    display_name = "Video Analysis Tool"
    description = "This component is responsible for video analysis tasks such as metadata and frame extraction."
    
    def build_config(self) -> dict:
        options = ["Extract Metadata", "Extract Frames"]
        return {
            "task_type": {
                "options": options,
                "value": options[0],
                "display_name": "Task Type"
            },
            "video_path": {
                "display_name": "Video Path",
                "type": "str"
            }
        }

    def build(self, video_path: str, task_type: str) -> Document:
        if task_type == "Extract Metadata":
            metadata = extract_metadata(video_path)  # Replace with actual function
            self.repr_value = f"Extracted Metadata: {metadata}"
            return Document(page_content=metadata)
        
        elif task_type == "Extract Frames":
            frames = extract_frames(video_path)  # Replace with actual function
            self.repr_value = f"Extracted Frames: {frames}"
            return Document(page_content=frames)
