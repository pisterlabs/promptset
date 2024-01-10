import os
import json
import openai
import yt_dlp
import pyrallis
import whisperx
import subprocess

from pathlib import Path
from typing import NamedTuple, Optional
from transformers import pipeline
from dataclasses import dataclass, asdict
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils import read_text
from data.data_config import DataConfig, ScraperConfig
import jinja2 as j2

openai.api_key = os.getenv("OPENAI_API_KEY")


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    whisperx_large = "large-v2"


class Sentiment(NamedTuple):
    positive = "Positive"
    neutral = "Neutral"
    negative = "Negative"


@dataclass
class TextSegment:
    text: str
    start: float
    end: float
    sentiment: Optional[str] = None


class Whisper:
    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size

    def transcribe(self, audio_path: Path) -> list[dict]:
        # Load pre-trained ASR model
        transcriber = pipeline("automatic-speech-recognition", model=self.model_name)
        transcription = transcriber(
            str(audio_path), return_timestamps=True, chunk_length_s=self.batch_size
        )
        text_segments = [
            asdict(
                TextSegment(
                    text=seg["text"],
                    start=seg["timestamp"][0],
                    end=seg["timestamp"][1],
                )
            )
            for seg in transcription["chunks"]
        ]
        return text_segments


class WhisperX(Whisper):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        device: str,
        compute_type: str = "float16",
    ):
        self.compute_type = compute_type
        self.device = device
        super().__init__(model_name=model_name, batch_size=batch_size)

    def transcribe(self, audio_path: Path) -> list[dict]:
        model = whisperx.load_model(
            self.model_name, self.device, compute_type=self.compute_type
        )
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=self.batch_size)
        text_segments = [
            asdict(TextSegment(text=seg["text"], start=seg["start"], end=seg["end"]))
            for seg in result["segments"]
        ]
        return text_segments


def scrape_videos(
    cfg: ScraperConfig, action: str, dataset_dir: Path, video_prefix: str = "video"
):
    def filter_videos(info_dict):
        duration = info_dict.get("duration")
        lang = info_dict.get("language")
        if duration and (
            duration < cfg.min_vid_duration or duration > cfg.max_vid_duration
        ):
            return "The video is either too short or too long"
        if not lang == "en":
            return "This video is not in English"

    prompt = cfg.prefix_prompt + action
    ydl_opts = {
        "restrictfilenames": cfg.restrict_filenames,
        "match_filter": filter_videos,
        "format": cfg.ext,
        "noplaylist": cfg.no_playlist,
        "quiet": cfg.quiet_mode,
        "writeautomaticsub": cfg.write_auto_subs,
        "writeinfojson": cfg.write_info_json,
        "ignoreerrors": True,
        "outtmpl": {
            "default": f"{dataset_dir / action / video_prefix}/%(title)s.%(ext)s"
        },
    }

    max_num_urls = cfg.max_num_url
    url = cfg.urls
    if url is None:
        url = f"{cfg.extractor}{max_num_urls}:{prompt}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error = ydl.download(url)
        print(error)


def extract_audio(
    vid_path: Path, cache: bool, prefix: str = "audio", ext: str = "wav"
) -> Path:
    audio_dir = vid_path.parents[1] / prefix
    audio_dir.mkdir(exist_ok=True)
    filepath = audio_dir / f"{vid_path.stem}.{ext}"
    if cache is True and filepath.exists():
        print(f"skip audio-extractor, use local {filepath.name}")
    else:
        with VideoFileClip(str(vid_path)) as clip:
            clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(
    audio_path: Path, batch_size: int, cache: bool, prefix: str = "text"
) -> Path:
    text_dir = audio_path.parents[1] / prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    if cache is True and filepath.exists():
        print(f"skip transcriber, use local {filepath.name}")
    else:
        s2t_model = WhisperX(
            model_name=ASRModelZoo.whisperx_large,
            batch_size=batch_size,
            device="cuda",
        )
        transcription = s2t_model.transcribe(audio_path)
        with open(filepath, "w") as fp:
            json.dump(transcription, fp)
    return filepath


def prepare_prompt(
    text_path: Path, system_template_path: Path, user_template_path: Path
) -> tuple[str, str]:
    data = read_text(text_path)
    text_segments = [segment["text"] for segment in data]
    txt = ""
    for i, seg in enumerate(text_segments):
        txt += f"{i + 1}.{seg}\n"

    templates_dir = system_template_path.parent
    environment = j2.Environment(loader=j2.FileSystemLoader(templates_dir))
    system_template = environment.get_template(system_template_path.name)
    user_template = environment.get_template(user_template_path.name)
    sentences = {"sentences": txt}
    system_prompt = system_template.render()
    user_prompt = user_template.render(sentences)
    return system_prompt, user_prompt


def write_gpt_response(
    system_prompt: str, user_prompt: str, output_path: Path, cache: bool
):
    if cache is True and output_path.exists():
        print(f"skip ChatGPT, use local {output_path.name}")
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        with open(output_path, "w") as fp:
            json.dump(response, fp)


def get_gpt_sentences(gpt_path: Path) -> list[str]:
    response = read_text(gpt_path)
    sentences = response["choices"][0]["message"]["content"].split("\n")
    return sentences


def parse_annotations(results_path: Path, vid_name: str) -> list[list]:
    annotations = read_text(results_path)
    vid_names = {x["data"].get("name"): i for i, x in enumerate(annotations)}
    vid_idx = vid_names.get(vid_name)
    vid_annotations = None if vid_idx is None else annotations[vid_idx]
    new_segments = []
    if vid_annotations is not None:
        results = vid_annotations["annotations"][0]["result"]
        for res in results:
            label = res["value"]["labels"][0]
            text = res["value"]["text"]
            new_segments.append([label, text])
    return new_segments


def calculate_word_durations(
    old_segments, old_time_stamps
) -> tuple[list, list, list, list]:
    word_durations = []
    word_durations_plus_jump = []
    bools = []
    jumps_alone = []
    for i, segment in enumerate(old_segments):
        start_time, end_time = old_time_stamps[i]
        if i > 0:
            _, prev_end = old_time_stamps[i - 1]
            jump = start_time - prev_end
        else:
            jump = 0
        words = segment.split()
        word_duration = (end_time - start_time) / len(words)

        word_durations_plus_jump.extend(
            [
                word_duration + jump if i == 0 else word_duration
                for i in range(len(words))
            ]
        )
        jumps_alone.extend([jump for i in range(len(words))])
        if i % 2 == 0:
            bools.extend([True for i in range(len(words))])
        else:
            bools.extend([False for i in range(len(words))])
        word_durations.extend(
            [word_duration if i == 0 else word_duration for i in range(len(words))]
        )
    return word_durations, word_durations_plus_jump, bools, jumps_alone


def calculate_new_time_stamps(
    old_segments: list[str], old_time_stamps: list[tuple], new_segments: list[list]
) -> list[tuple]:
    (
        word_durations,
        word_durations_plus_jump,
        bools,
        jumps_alone,
    ) = calculate_word_durations(old_segments, old_time_stamps)
    new_time_stamps = []
    current_word = 0  # Initialize current_word index
    current_start = old_time_stamps[0][0]
    for label, text in new_segments:
        words = text.split()
        if all(bools[current_word : current_word + len(words)]) or not any(
            bools[current_word : current_word + len(words)]
        ):
            segment_duration = sum(
                word_durations[current_word : current_word + len(words)]
            )
            new_time_stamps.append((current_start, current_start + segment_duration))

        else:
            segment_duration = sum(
                word_durations_plus_jump[current_word : current_word + len(words)]
            )
            new_time_stamps.append((current_start, current_start + segment_duration))
            current_start = current_start + jumps_alone[current_word]

        current_word += len(words)  # Increment by word count
        current_start += segment_duration
    return new_time_stamps


def accumulate_text_by_interpolation(
    text_path: Path, new_segments: list[list]
) -> list[TextSegment]:
    text_data = read_text(text_path)
    old_segments = [segment["text"] for segment in text_data]
    old_timestamps = [(segment["start"], segment["end"]) for segment in text_data]
    new_timestamps = calculate_new_time_stamps(
        old_segments, old_timestamps, new_segments
    )
    chunks = []
    for segment, timestamp in zip(new_segments, new_timestamps):
        chunks.append(
            TextSegment(
                text=segment[1],
                start=timestamp[0],
                end=timestamp[1],
                sentiment=segment[0],
            )
        )
    return chunks


def accumulate_text_by_sentiment(
    text_path: Path, sentiments: list[str]
) -> list[TextSegment]:
    data = read_text(text_path)
    text_segments = [segment["text"] for segment in data]
    samples = []
    end = None
    text_paragraph = text_segments[0]
    accumulated_sentiments = [sentiments[0]]
    start = data[0]["timestamp"][0]
    for i in range(1, len(text_segments)):
        curr_segment = text_segments[i]
        curr_sentiment = sentiments[i]
        prev_sentiment = sentiments[i - 1]

        if curr_sentiment == prev_sentiment or curr_sentiment == Sentiment.neutral:
            text_paragraph += curr_segment
            accumulated_sentiments.append(curr_sentiment)
            end = data[i]["timestamp"][-1]
            if end is None:
                end = data[i]["timestamp"][0]

        else:
            sentiment = Sentiment.positive
            if Sentiment.negative in accumulated_sentiments:
                sentiment = Sentiment.negative

            samples.append(
                TextSegment(
                    text=text_paragraph, start=start, end=end, sentiment=sentiment
                )
            )
            start = data[i]["timestamp"][0]
            end = data[i]["timestamp"][-1]
            text_paragraph = text_segments[i]
            accumulated_sentiments = [sentiments[i]]

    if Sentiment.positive in accumulated_sentiments:
        sentiment = Sentiment.positive
    elif Sentiment.negative in accumulated_sentiments:
        sentiment = Sentiment.negative
    else:
        sentiment = Sentiment.neutral
        print(f"all sentences are {sentiment}")

    samples.append(
        TextSegment(text=text_paragraph, start=start, end=end, sentiment=sentiment)
    )
    return samples


def cut_video_by_text_chunks(
    vid_path: Path,
    chunks: list[TextSegment],
    video_output_dir: Path,
    text_output_dir: Path,
    cache: bool,
):
    vid_segment_dir = video_output_dir / vid_path.stem
    text_segment_dir = text_output_dir / vid_path.stem
    if cache is True and vid_segment_dir.exists() and text_segment_dir.exists():
        print(f"skip cutting video chunks, use existing chunks in {vid_segment_dir}")
    else:
        vid_segment_dir.mkdir(exist_ok=True, parents=True)
        text_segment_dir.mkdir(exist_ok=True, parents=True)
        with VideoFileClip(str(vid_path)) as vid:
            for sentence in chunks:
                start, end = sentence.start, sentence.end
                sub_vid = vid.subclip(start, end)
                segment_name = f"{vid_path.stem}_{start:.{1}f}_{end:.{1}f}"
                vid_segment_path = vid_segment_dir / f"{segment_name}.mp4"
                text_segment_path = text_segment_dir / f"{segment_name}.json"

                sub_vid.write_videofile(
                    str(vid_segment_path),
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                )
                with open(text_segment_path, "w") as fp:
                    json.dump(asdict(sentence), fp)
            sub_vid.close()
        vid.close()


def run_alphapose_on_videos(root_dir: Path, output_dir: Path, vid_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg_path = (
        root_dir
        / "configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml"
    )
    ckpt = root_dir / "pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth"

    for i, vid_path in enumerate(vid_dir.rglob("*.mp4")):
        vid_output_dir = output_dir / vid_path.stem
        vid_output_dir.mkdir(exist_ok=True)
        subprocess.run(
            f"{root_dir / 'scripts/inference.sh'} {cfg_path} {ckpt} {vid_path} {vid_output_dir}",
            shell=True,
        )


@pyrallis.wrap()
def main(cfg: DataConfig):
    dataset_dir = cfg.dataset_dir
    actions = cfg.actions
    # scrape new videos or use local videos otherwise
    if cfg.scraper.run is True:
        for action in actions:
            print(f"{action}:")
            scrape_videos(cfg=cfg.scraper, action=action, dataset_dir=dataset_dir)

    out_gpt_dir = cfg.output_dir / "gpt"
    out_gpt_dir.mkdir(exist_ok=True, parents=True)
    video_output_dir = cfg.output_dir / "video"
    text_output_dir = cfg.output_dir / "text"

    files = dataset_dir.rglob("*.mp4")
    if len(cfg.filenames) > 0:
        files = [dataset_dir / "video" / name for name in cfg.filenames]

    for vid_path in files:
        # extract audio and transcription from videos
        audio_path = extract_audio(vid_path, cache=cfg.audio_extractor.use_cache)
        text_path = transcribe_speech(
            audio_path,
            cfg.transcriber.chunk_length_s,
            cache=cfg.transcriber.use_cache,
            prefix="text",
        )
        system_prompt, user_prompt = prepare_prompt(
            text_path,
            system_template_path=cfg.templates.system_prompt_path,
            user_template_path=cfg.templates.user_prompt_path,
        )

        if cfg.sentence_segments.use_manual_annotations:
            new_segments = parse_annotations(
                cfg.sentence_segments.manual_results_path, vid_path.stem
            )
        else:
            gpt_path = out_gpt_dir / text_path.name
            # OPENAI GPT API Call
            write_gpt_response(
                system_prompt,
                user_prompt,
                gpt_path,
                cache=cfg.sentence_segments.use_cache,
            )
            sentences = get_gpt_sentences(gpt_path)
            new_segments = [sentence.split(": ") for sentence in sentences]
        chunks = accumulate_text_by_interpolation(text_path, new_segments)
        # segment videos by GPT outputs
        cut_video_by_text_chunks(
            vid_path,
            chunks,
            video_output_dir,
            text_output_dir,
            cache=cfg.video_cutter.use_cache,
        )
    # run alphapose
    out_pose_dir = cfg.output_dir / "pose"
    run_alphapose_on_videos(
        root_dir=cfg.alphapose.root_dir,
        output_dir=out_pose_dir,
        vid_dir=video_output_dir,
    )


if __name__ == "__main__":
    main()
