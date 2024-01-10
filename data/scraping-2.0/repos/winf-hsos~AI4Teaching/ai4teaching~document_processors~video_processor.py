from ai4teaching import DocumentProcessor
from ai4teaching import EmbeddingModel
from ai4teaching import LargeLanguageModel
from ai4teaching.utils import log
from pytube import YouTube
import json
import math

class VideoProcessor(DocumentProcessor):
    def __init__(self, document, processed_documents_path, embedding_model: EmbeddingModel, llm: LargeLanguageModel):
        log("Initializing VideoProcessor")

        # Get the title of the YouTube video
        yt = YouTube(document["document_uri"])
        document["title"] = yt.title

        super().__init__(document, processed_documents_path, embedding_model)
        self.llm = llm
        self.llm.set_model("gpt-4-1106-preview")

    def process(self):
        log(f"Processing video document from >{self.document['document_uri']}<", type="info")

        # Download the audio from the video
        self._extract_audio_from_youtube()

        # Transcribe the audio
        self._transcribe_audio()

        # Split the transcript into overlapping segments
        self._create_transript_segments(segment_length=120, overlap_length=30)

        # Create the processed document file with chunks
        self._create_document_chunks(previous_step_name=DocumentProcessor.STEP_CREATE_TRANSCRIPT_SEGMENTS)

        # Embed the chunks
        self._embed_document_chunks(previous_step_name=DocumentProcessor.STEP_CREATE_DOCUMENT_CHUNKS)

        # Summarize the chunks
        #self._summarize_document_chunks()

        self.document["processing_outputs"] = self.step_ouput_files

        log(f"✔ Done processing video document from >{self.document['document_uri']}<", type="success")

        return self.document

    def _extract_audio_from_youtube(self):
        processing_required = self._prepare_and_check_if_processing_step_required(DocumentProcessor.STEP_EXTRACT_AUDIO_FROM_YOUTUBE)

        if not processing_required:
            return

        yt = YouTube(self.document["document_uri"])
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(filename=self.step_ouput_files[DocumentProcessor.STEP_EXTRACT_AUDIO_FROM_YOUTUBE])

    def _transcribe_audio(self):
        processing_required = self._prepare_and_check_if_processing_step_required(DocumentProcessor.STEP_TRANSCRIBE_AUDIO)

        if not processing_required:
            return

        from openai import OpenAI
        client = OpenAI()
        audio_file = open(self.step_ouput_files[DocumentProcessor.STEP_EXTRACT_AUDIO_FROM_YOUTUBE], "rb")
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="verbose_json"
            )

        # Initialize summary JSON with mandatory fields
        transcript_document = self._get_mandatory_document_data()

        # Add additional fields into existing JSON
        transcript_document["text"] = response.text
        transcript_document["metadata"] = {
            "language" : response.language,
            "duration" : response.duration
        }
        transcript_segments = []
        transcript_document["content"] = transcript_segments

        for s in response.segments:
            segment = {}
            segment["metadata"] = {
                "segment_id" : s["id"],
                "start" : s["start"],
                "end" : s["end"]
            }
            segment["text"] = s["text"]
            transcript_segments.append(segment)
       
        # Save transcript to file
        self._save_json_file_for_step(DocumentProcessor.STEP_TRANSCRIBE_AUDIO, transcript_document)
        
    def _create_transript_segments(self, segment_length=60, overlap_length=20):
        processing_required = self._prepare_and_check_if_processing_step_required(DocumentProcessor.STEP_CREATE_TRANSCRIPT_SEGMENTS)

        if not processing_required:
            return

        # Read the transcript file
        transcript_json = self._load_json_file_for_step(DocumentProcessor.STEP_TRANSCRIBE_AUDIO)

        # Initialize summary JSON with mandatory fields
        transcript_segments_document = self._get_mandatory_document_data()

        # Add additional fields into existing JSON
        transcript_segments_document["text"] = transcript_json["text"]
        transcript_segments_document["metadata"] = transcript_json["metadata"]

        segments = []
        transcript_segments_document["content"] = segments

        # First segment
        start = 0
        end = segment_length
        segment = self._extract_segment_by_timestamp(transcript_json, start, end)
        if segment is not None:
            segments.append(segment)

        while segment is not None:
            start = end - overlap_length
            end = start + segment_length
            
            segment = self._extract_segment_by_timestamp(transcript_json, start, end)
            if segment is not None:
                segments.append(segment)

        # Save output to file
        self._save_json_file_for_step(DocumentProcessor.STEP_CREATE_TRANSCRIPT_SEGMENTS, transcript_segments_document)
        
    def _extract_segment_by_timestamp(self, transcript_json, from_time, to_time):
        extracted_segment_start = math.inf
        extracted_segment_end = 0

        extracted_segments_text = ""
        for segment in transcript_json["content"]:
            start_time = segment["metadata"]["start"]
            end_time = segment["metadata"]["end"]
            if start_time <= to_time and end_time >= from_time:
                extracted_segments_text += segment["text"]
                if start_time < extracted_segment_start:
                    extracted_segment_start = start_time
                if end_time > extracted_segment_end:
                    extracted_segment_end = end_time

        return { 
            "text": extracted_segments_text, 
            "metadata" : { 
                "start": extracted_segment_start, 
                "end": extracted_segment_end, 
                "youtube_id" : self.document["youtube_id"] if "youtube_id" in self.document else None 
                } 
            } if extracted_segments_text != "" else None

    def _create_document_chunks(self, previous_step_name=None):
        processing_required = self._prepare_and_check_if_processing_step_required(DocumentProcessor.STEP_CREATE_DOCUMENT_CHUNKS, previous_step_name=previous_step_name)

        if not processing_required:
            return
    
        # Read the transcript segments file
        transcript_segments = self._load_json_file_for_step(DocumentProcessor.STEP_CREATE_TRANSCRIPT_SEGMENTS)
        
        # Initialize summary JSON with mandatory fields
        chunks_document = self._get_mandatory_document_data()
        
        # Add additional fields into existing JSON
        chunks_document["text"] = transcript_segments["text"]
        chunks_document["metadata"] = transcript_segments["metadata"] if "metadata" in transcript_segments else {}
        chunks_document["chunks"] = []
        
        for i, segment in enumerate(transcript_segments["content"]):
            chunk = { 
                "chunk_id" : f"{chunks_document['id']}_{i}",
                "metadata" : segment["metadata"] if "metadata" in segment else {},
                "content" : segment["text"]
            }

            chunks_document["chunks"].append(chunk)
        
        # Save output file
        self._save_json_file_for_step(DocumentProcessor.STEP_CREATE_DOCUMENT_CHUNKS, chunks_document)

    def _summarize_document_chunks(self):
        processing_required = self._prepare_and_check_if_processing_step_required(DocumentProcessor.STEP_SUMMARIZE_DOCUMENT_CHUNKS)

        if not processing_required:
            return

        # Load chunks file
        embedded_chunks = self._load_json_file_for_step(DocumentProcessor.STEP_CREATE_DOCUMENT_CHUNKS)

        # Initialize summary JSON with mandatory data
        summary_document = self._get_mandatory_document_data()

        chunks = [chunk for chunk in embedded_chunks["chunks"]]
        
        # Summarize each chunk
        summary_document["content"] = [{ "summary" : self.llm.summarize(c["content"]), "chunk_id" : c["chunk_id"] } for c in chunks]
        
        # Write to file
        self._save_json_file_for_step(DocumentProcessor.STEP_SUMMARIZE_DOCUMENT_CHUNKS, summary_document)
        