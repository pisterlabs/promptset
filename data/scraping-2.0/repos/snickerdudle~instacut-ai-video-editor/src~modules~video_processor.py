"""Contains the video download, frame splitting and sampling logic."""

import logging
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import imutils
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from src.modules.transcript_processor import (
    TranscriptProcessor,
    secondsToMinutes,
)
from src.utils.file_utils import BaseConfig, FileUtils
from src.utils.io_utils import tqdm
from src.utils.llm_utils import OpenAIChat
from src.utils.prompts.prompts import Prompt, PromptCollection

PathObj = Union[str, Path]
# A VideoObj is either a Video object or a URL.
VideoObj = Union["Video", str]


class VideoCaptureError(Exception):
    """Exception raised when a video cannot be captured."""

    pass


class Frame:
    """Class for a frame."""

    def __init__(
        self,
        image: Any,
        parent: Optional[Any] = None,
        timestamp: Optional[float] = None,
        order: Optional[int] = None,
        is_thumb: Optional[bool] = False,
    ):
        self.image = image
        self.parent = parent
        self.timestamp = timestamp
        self.order = order
        self.is_thumb = is_thumb

    def saveToDir(self, path: PathObj, name: Optional[str] = None) -> None:
        """Save the frame to a directory.

        Args:
            path (PathObj): The path to save the frame to, directory.
        """
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        name = f"{self.order:06d}.jpg" if name is None else name
        cv2.imwrite(str(path / name), self.image[..., ::-1])

        # Also save the thumbnail
        if not self.is_thumb:
            self.toThumb(128).saveToDir(path / "thumb")

        # Also save the preview
        if not self.is_thumb:
            self.toThumb(320).saveToDir(path / "preview")

    def __repr__(self) -> str:
        """Get the string representation of the frame."""
        s = "Frame"
        if self.order is not None:
            s += f" {self.order:,}"
        if self.timestamp is not None:
            s += f" ({secondsToMinutes(self.timestamp)})"
        return f"<{s}>"

    def toThumb(self, new_width: Optional[int] = 128) -> Any:
        """Get the thumbnail of the frame."""
        resized_image = imutils.resize(self.image, width=new_width)
        return self.__class__(
            resized_image,
            parent=self.parent,
            timestamp=self.timestamp,
            order=self.order,
            is_thumb=True,
        )


@dataclass
class VideoMetadata:
    """Class for the metadata of a video."""

    name: Optional[str] = None
    length: Optional[float] = None  # in seconds
    url: Optional[str] = None
    video_path: Optional[PathObj] = None
    framerate: Optional[int] = None
    resolution: Optional[int] = None
    thumbnail_url: Optional[str] = None


class Video:
    """Class for a video."""

    def __init__(
        self,
        metadata: VideoMetadata,
        output_dir: PathObj,
        frames: Optional[List[Frame]] = None,
    ):
        self.metadata = metadata
        self.output_dir = Path(output_dir) / self.unix_name(metadata.name)
        self.frames = frames or []

    @classmethod
    def from_file(cls, path: PathObj) -> "Video":
        """Create a video from a file."""
        raise NotImplementedError

    @classmethod
    def from_url(
        cls, url: str, output_dir: Optional[PathObj] = None
    ) -> "Video":
        """Create a video from a URL."""
        yt = YouTube(url)
        metadata = VideoMetadata(name=yt.title, length=yt.length, url=url)
        return cls(
            metadata=metadata,
            output_dir=Path(output_dir),
            frames=[],
        )
        raise NotImplementedError

    @classmethod
    def unix_name(cls, name: Optional[str] = None) -> str:
        """Convert string to unix-friendly name."""
        if name is None:
            # Generate uuid4 name.
            name = uuid.uuid4()
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
        name = name[:32]
        return name

    def get_unix_name(self) -> str:
        """Get the name of this video."""
        return self.unix_name(self.metadata.name)

    @classmethod
    def convertAllToVideo(
        cls,
        video: Union[VideoObj, List[VideoObj]],
        output_dir: Optional[PathObj] = None,
    ) -> Union["Video", List["Video"]]:
        """Make sure that all inputs are Video objects.

        Args:
            video (Union[VideoObj, List[VideoObj]]): The video(s) to convert.
            output_path (Optional[PathObj], optional): The output path. Defaults to None.
        Returns:
            List[Video]: The converted video(s)."""

        def convert(video):
            if isinstance(video, str):
                # The current item is a URL.
                return cls.from_url(video, output_dir=output_dir)
            else:
                # The current item is a Video object.
                return video

        video = (
            [convert(v) for v in video]
            if isinstance(video, list)
            else convert(video)
        )

        return video

    def download(
        self, path: Optional[PathObj] = None, custom_name: Optional[str] = None
    ) -> None:
        """Download the video to a file."""
        if self.metadata.url is None:
            raise ValueError("The video does not have a URL.")
        yt = YouTube(self.metadata.url)

        if path is not None:
            path = Path(path)
            if path.is_file():
                logging.warning(
                    "The path is a file, not a directory. Using the parent directory instead."
                )
                path = path.parent
        else:
            path = FileUtils.getVideoFileOutputDir(self)

        # TODO: Add custom download options.
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        filename = custom_name or stream.default_filename
        stream.download(output_path=str(path), filename=filename)
        self.metadata.video_path = path / filename

    def procureFile(self) -> PathObj:
        """Ensure a video file exists, either by downloading it or using an existing file.

        Returns:
            PathObj: The path to the video file.
        """
        if self.metadata.video_path is None:
            self.download()
        else:
            if not self.metadata.video_path.exists():
                self.download()
        return self.metadata.video_path

    def sample(
        self,
        sampling_policy: "SamplingPolicyBaseClass",
        save_frames: Optional[bool] = False,
    ) -> Tuple[List[Frame], PathObj]:
        """Sample the frames of the video.

        Args:
            sampling_policy (SamplingPolicyBaseClass): The sampling policy to use.
            save_frames (Optional[bool], optional): Whether to save the frames to a file. Defaults to False.

        Returns:
            List[Frame]: The sampled frames.
        """
        return sampling_policy.sample(self, save_frames=save_frames)

    def getTranscript(self, use_timestamps=False) -> str:
        """Get the transcript of the video."""
        url = self.metadata.url
        url = re.sub(r".*/watch\?v=", "", url)
        transcript = YouTubeTranscriptApi.get_transcript(url)
        transcript_processor = TranscriptProcessor(transcript)
        if use_timestamps:
            # Convert the transcript to a string with timestamps.
            new_transcript = transcript_processor.trimIntervals(n=48)
            transcript_text = (
                transcript_processor.generateTimestampedTextTranscript(
                    new_transcript
                )
            )
        else:
            transcript_text = transcript_processor.generateJustTextTranscript()

        return transcript_text


class SamplingPolicyBaseClass:
    """Class for sampling the frames of a video."""

    def getSamplingIndeces(self, video_capture: cv2.VideoCapture) -> List[int]:
        """Get the indices of the frames to sample."""
        raise NotImplementedError

    def sample(
        self,
        video: Union[Video, cv2.VideoCapture],
        save_frames: Optional[bool] = False,
    ) -> Tuple[List[Frame], PathObj]:
        """Sample frames from the video.

        Args:
            video (Video): The video to sample.
            save_frames (Optional[bool], optional): Whether to save the frames to a file. Defaults to False.
        Returns:
            Tuple[List[Frame], PathObj]: The sampled frames and the path to the frames directory.
        """
        if isinstance(video, Video):
            # Check to see if the video frames already exist.
            frames_dir = FileUtils.getVideoFramesOutputDir(video, self)
            if frames_dir.exists() and len(list(frames_dir.iterdir())) > 0:
                # The frames already exist, so no need to regenerate.
                logging.info(
                    f"The frames for {video.metadata.name} already exist. Skipping."
                )
                return None, frames_dir

            video_file_path = video.procureFile()
            video_capture = cv2.VideoCapture(str(video_file_path))
        else:
            video_capture = video
            frames_dir = None
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        indeces = self.getSamplingIndeces(video_capture)
        output_frames = []

        for frame_index in tqdm(indeces, desc="Sampling frames", unit="frame"):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, image = video_capture.read()
            if not success:
                raise VideoCaptureError("The video capture failed.")

            output_frames.append(
                Frame(
                    image=image[..., ::-1],
                    order=frame_index,
                    parent=video if isinstance(video, Video) else None,
                    timestamp=float(frame_index) / fps,
                )
            )
        if save_frames:
            self.save_frames(video, output_frames)
        video_capture.release()
        return output_frames, frames_dir

    def save_frames(self, video, frames):
        """Save the frames to separate files. Use the video's output directory.

        Args:
            video (Video): The video.
            frames (List[Frame]): The frames to save.

        Returns:
            None
        """
        FileUtils.saveVideoFrames(video, self, frames)

    @property
    def name(self) -> str:
        """Get the Unix name of the sampling policy."""
        # Get the name of the class
        name = self.__class__.__name__
        # Remove the "SamplingPolicy" part
        name = name.replace("SamplingPolicy", "")
        # Convert to lowercase
        name = name.lower()
        value = str(getattr(self, "value", ""))
        if value:
            name += f"_{value}"
        return name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


class UniformFramesSamplingPolicy(SamplingPolicyBaseClass):
    """Class for uniformly sampling the frames of a video."""

    default_value = 1

    def __init__(self, value: Optional[Union[str, int]] = None) -> None:
        # Sample every n frames. n=1 means sample every frame. n=2 means sample
        # every other frame, etc.
        self.value = int(value) if value is not None else self.default_value
        if self.value < 1:
            raise ValueError("The sampling interval must be at least 1.")

    def getSamplingIndeces(self, video_capture: cv2.VideoCapture) -> List[int]:
        """Sample every nth frame of the video, starting at frame index 0.

        So, if self.value is 4, the following will be sampled:
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
        ^       ^       ^         ^           ^

        Args:
            video_capture (VideoCapture): The VideoCapture cv2 object.
        Returns:
            List[Frame]: The sampled frames.
        """
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        output_indeces = [_ for _ in range(0, int(total_frames), self.value)]
        return output_indeces


class UniformSecondsSamplingPolicy(SamplingPolicyBaseClass):
    """Class for uniformly sampling the seconds of a video."""

    default_value = 1.0

    def __init__(self, value: Optional[Union[str, float]] = None) -> None:
        # Sample every n seconds. n=1 means sample every second. n=2 means
        # sample every 2 seconds, etc.
        self.value = float(value) if value is not None else self.default_value
        if self.value < 0:
            raise ValueError("The sampling interval must be at least 0.")

    def getSamplingIndeces(self, video_capture: cv2.VideoCapture) -> List[int]:
        """Sample every nth second of the video, starting at frame index 0.

        So, if self.value is 0.1, and fps is 30, then the following will be
        sampled (every 0.1 * 30 = 3 frames):
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
        ^     ^     ^     ^       ^        ^        ^

        Args:
            video_capture (VideoCapture): The VideoCapture cv2 object.
        Returns:
            List[Frame]: The sampled frames.
        """
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        interval = int(round(fps * self.value))
        output_indeces = [_ for _ in range(0, int(total_frames), interval)]
        return output_indeces


class RandomSamplingPolicy(SamplingPolicyBaseClass):
    """Class for random sampling the frames of a video."""

    default_value = 0.5

    def __init__(
        self,
        value: Optional[Union[float, str]] = None,
        total_number: Optional[int] = None,
    ) -> None:
        self.value = float(value) if value is not None else self.default_value
        if self.value < 0 or self.value > 1:
            raise ValueError(
                "The sampling probability must be between 0 and 1."
            )
        self.total_number = total_number

    def getSamplingIndeces(self, video_capture: cv2.VideoCapture) -> List[int]:
        """Sample the frames of a video randomly, returning a set final number."""
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.total_number is not None:
            frames_to_sample = self.total_number
        else:
            frames_to_sample = int(round(total_frames * self.value))
        # Randomly sample frames_to_sample frames from the video.
        output_indeces = random.sample(
            [_ for _ in range(0, int(total_frames))], frames_to_sample
        )
        output_indeces = sorted(output_indeces)
        return output_indeces


class VideoProcessorConfig(BaseConfig):
    """Class for the configuration of a video processing job."""

    def __init__(
        self,
        sampling_policy: Union[SamplingPolicyBaseClass, str],
        output_dir: PathObj,
        prompt: Optional[Union[Prompt, str]] = None,
        prompt_id: Optional[str] = None,
        model: Optional[str] = None,
        video: Optional[Union[VideoObj, List[VideoObj]]] = None,
        **kwargs,
    ) -> None:
        self.sampling_policy = self._retrieve_sampling_policy(sampling_policy)
        self.output_dir = Path(output_dir).resolve()
        self.prompt = prompt or PromptCollection()[prompt_id]
        self.model = model
        self.video = video
        super().__init__(**kwargs)

    def _retrieve_sampling_policy(self, sampling_policy: str):
        """Retrieve the sampling policy."""
        # List possible sampling policies.

        if isinstance(sampling_policy, SamplingPolicyBaseClass):
            return sampling_policy

        options = {
            # random, random_0.25, random 0.25, random0.25, r, r_0.25, r 0.25, r0.25
            r"^(?:random|r)[_\s]?([\d\.]*)$": RandomSamplingPolicy,
            # random_1/4, random 1/4, random1/4, r_1/4, r 1/4, r1/4
            r"^(?:random|r)[_\s]?(\d+/\d+)$": RandomSamplingPolicy,
            # uniform, uniform_2, uniform 2, uniform2, u, u_2, u 2, u2
            r"^(?:uniform|u)[_\s]?([\d]*)$": UniformFramesSamplingPolicy,
            # seconds, seconds_0.1, seconds 0.1, seconds0.1, s, s_0.1, s 0.1, s0.1
            r"^(?:second|seconds|s)[_\s]?([\d\.]*)$": UniformSecondsSamplingPolicy,
            # seconds_1/40, seconds 1/40, seconds1/40, s_1/40, s 1/40, s1/40
            r"^(?:second|seconds|s)[_\s]?(\d+/\d+)$": UniformSecondsSamplingPolicy,
        }

        def resolve_value(value: Optional[str] = None):
            """Resolve the value."""
            if value is None:
                return value
            if "/" in value:
                numerator, denominator = value.split("/")
                return float(numerator) / float(denominator)
            return value

        for pattern, policy in options.items():
            if re.match(pattern, sampling_policy):
                value = re.match(pattern, sampling_policy).group(1)
                value = resolve_value(value)
                if not value:
                    return policy()
                else:
                    return policy(value)

        # If no match, raise an error.
        raise ValueError(
            f'The sampling policy "{sampling_policy}" is not valid.'
        )


class SummaryConfig(BaseConfig):
    def __init__(
        self,
        save_transcript: Optional[bool] = False,
        use_timestamps: Optional[bool] = True,
        **kwargs,
    ):
        self.save_transcript = save_transcript
        self.use_timestamps = use_timestamps
        super().__init__(**kwargs)


class FrameSamplerConfig(BaseConfig):
    def __init__(self, save_frames: Optional[bool] = False, **kwargs):
        self.save_frames = save_frames
        super().__init__(**kwargs)


class VideoProcessor:
    """Class for processing a video or a set of videos."""

    def __init__(
        self,
        config: VideoProcessorConfig,
    ) -> None:
        self.config = config
        self.file_processor = FileUtils(output_dir=self.config.output_dir)

    def summarize(
        self,
        video: Optional[Union[VideoObj, List[VideoObj]]] = None,
        model: Optional[str] = None,
        use_timestamps: bool = False,
        save_transcript: bool = False,
        custom_chat: Optional[OpenAIChat] = None,
    ) -> int:
        """Summarize the video(s).

        Args:
            video (Union[VideoObj, List[VideoObj]]): The video(s) to summarize.
            model (Optional[str], optional): The model to use for summarization. Defaults to None.
            use_timestamps (bool, optional): Whether to include timestamps in the summary. Defaults to False.
            save_transcript (bool, optional): Whether to save the transcript of the video. Defaults to False.
            custom_chat (Optional[OpenAIChat], optional): A custom chat to use for summarization. Defaults to None.

        Returns:
            int: The number of videos summarized.
        """
        # Resolve config inputs
        sc = SummaryConfig.from_parent_config(self.config)

        video = video or self.config.video
        model = model or self.config.model
        use_timestamps = use_timestamps or sc.use_timestamps
        save_transcript = save_transcript or sc.save_transcript

        if not isinstance(video, list):
            video = [video]
        videos = Video.convertAllToVideo(
            video, output_dir=Path(self.config.output_dir)
        )
        for video in tqdm(videos, desc="Summarizing videos", unit="video"):
            print('Performing summarization on "{}"'.format(video))
            name = video.get_unix_name()
            # Check if exists already to not burn through API calls.
            if (
                self.file_processor.getVideoSummaryOutputDir(video)
                / "summary.txt"
            ).exists():
                print(f'File "{name}" summary already exists. Skipping')
                continue
            transcript = video.getTranscript(use_timestamps=use_timestamps)
            if save_transcript:
                self.file_processor.saveVideoTranscript(
                    video=video, transcript=transcript
                )
            chat = (
                (
                    OpenAIChat(self.config.prompt)
                    if model is None
                    else OpenAIChat(self.config.prompt, model=model)
                )
                if custom_chat is None
                else custom_chat
            )
            summary = chat.processMessage(transcript)
            self.file_processor.saveVideoSummary(video=video, summary=summary)
        return len(videos)

    def sampleFrames(
        self,
        video: Union[VideoObj, List[VideoObj]],
        save_frames: Optional[bool] = False,
    ) -> Union[List[Frame], List[List[Frame]]]:
        """Sample the frames of a video or a set of videos.

        Args:
            video (Union[VideoObj, List[VideoObj]]): The video(s) to sample.
            save_frames (Optional[bool], optional): Whether to save the sampled frames. Defaults to False.
        Returns:
            Union[List[Frame], List[List[Frame]]]: The sampled frames.
        """
        # Resolve config inputs
        fsc = FrameSamplerConfig.from_parent_config(self.config)

        video = video or self.config.video
        save_frames = save_frames or fsc.save_frames

        # Validate video
        videos = Video.convertAllToVideo(
            video, output_dir=Path(self.config.output_dir)
        )
        if isinstance(videos, Video):
            return [
                videos.sample(
                    self.config.sampling_policy, save_frames=save_frames
                )
            ]
        else:
            output_samples = []
            for video in videos:
                # Frames and directory where the frames are stored
                output_samples.append(
                    video.sample(
                        self.config.sampling_policy, save_frames=save_frames
                    )
                )
            return output_samples
