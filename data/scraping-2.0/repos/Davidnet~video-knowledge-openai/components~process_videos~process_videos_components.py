"""
Module that contains the process videos components.
Author: David Cardozo <david.cardozo@me.com>
"""
from kfp import dsl
from kfp.dsl import Artifact, Input, Output

# pylint: disable=import-outside-toplevel


@dsl.component(
    target_image="us-central1-docker.pkg.dev/mlops-explorations/yt-whisper-images/transcribe:3.1",
    base_image="davidnet/python-ffmpeg:1.0",
    packages_to_install=["pytube==12.1.3", "openai==0.27.4", "tenacity==8.2.2"],
)
def process_videos(video_urls: Input[Artifact], transcriptions: Output[Artifact]):
    """
    Component that processes the videos.
    :param video_urls: Artifact that contains urls of the videos to process.
    """
    import json
    import os
    import shutil
    import subprocess as sp
    import tempfile
    from pathlib import Path

    import openai
    from pytube import YouTube
    from pytube.exceptions import RegexMatchError
    from tenacity import (
        retry,
        stop_after_attempt,  # for exponential backoff
        wait_random_exponential,
    )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _transcribe_with_backoff(**kwargs):
        return openai.Audio.transcribe(**kwargs)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt_keywords = "Kubeflow pipelines kustomize Kubernetes MLOps UI, Katib Tuning API CLI changelogs CNCF Katib fine-grained log correlation hyperparameter Tune API Docker scripts SDK Katib RBAC Postgres pagination HPA Pytorch Elastic Job Spec. PodGroup Paddlepaddle coscheduling KFP v2 2.0.0-alpha.7 Sub-DAG visualization PodDefaults TensorBoard PodDefaults S3 K8s Istio KNative Kustomize Cert Mgr DEX Argo Tekton Oidc-authservice"

    stage_dir = Path(tempfile.mkdtemp())
    with open(video_urls.path, "r", encoding="utf-8") as f:
        urls = f.readlines()

    urls = {url.strip() for url in urls}
    for video_url in urls:
        print(f"Processing video: {video_url}")
        try:
            yt_handler = YouTube(video_url)
        except RegexMatchError:
            print(f"Invalid url: {video_url}")
            continue
        try:
            audio_stream = yt_handler.streams.filter(
                only_audio=True, file_extension="mp4"
            ).first()
        except Exception:
            print(f"Could not find audio stream for video: {video_url}")
            continue
        if audio_stream is None:
            print(f"No audio stream found for video: {video_url}")
            continue
        video_description = yt_handler.description
        video_title = yt_handler.title
        video_url = yt_handler.watch_url
        video_id = yt_handler.video_id
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = "video.mp4"
            try:
                audio_stream.download(output_path=temp_dir, filename=filename)
            except Exception:
                print(f"Could not download video: {video_url}")
                continue
            filename_path = Path(temp_dir) / filename
            tmp_path = Path(temp_dir)
            audio_chunks = tmp_path / "chunks"
            audio_chunks.mkdir()
            print(f"Splitting audio file for video: {video_url}")
            sp.check_call(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    filename_path.resolve(),
                    "-f",
                    "segment",
                    "-segment_time",
                    "60",
                    f"{audio_chunks.resolve()}/%d.mp3",
                ]
            )
            json_file_path = Path(temp_dir) / f"{video_title}.jsonl"
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                for audio_file in audio_chunks.glob("*.mp3"):
                    print(
                        f"Transcribing audio file: {audio_file.name} for video: {video_url}"
                    )
                    chunk_number = int(audio_file.stem)
                    with open(audio_file, "rb") as f:
                        prompt = f"{video_title} {video_description}"
                        prompt += " the folowing keywords are used " + prompt_keywords
                        transcript = _transcribe_with_backoff(
                            model="whisper-1", file=f, prompt=prompt, language="en"
                        )
                        transcript_text = transcript["text"]
                        json_str = json.dumps(
                            {
                                "title": video_title,
                                "url": video_url,
                                "text": transcript_text,
                                "start": chunk_number * 60,
                                "end": max((chunk_number + 1) * 60, yt_handler.length),
                                "video_id": f"{video_id}?t={chunk_number * 60}",
                            }
                        )
                        json_file.write(json_str + "\n")
            shutil.copy(json_file_path, stage_dir / f"{video_id}.jsonl")
    shutil.make_archive(transcriptions.path, "gztar", root_dir=stage_dir)
    transcriptions.uri = transcriptions.uri + ".tar.gz"
