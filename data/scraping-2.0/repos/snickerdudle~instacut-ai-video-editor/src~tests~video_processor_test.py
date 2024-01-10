# Tests for the VideoProcessor code.

import tempfile
import unittest
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

import src.modules.video_processor as vp_module
from src.modules.video_processor import (
    RandomSamplingPolicy,
    UniformFramesSamplingPolicy,
    UniformSecondsSamplingPolicy,
    Video,
    VideoMetadata,
    VideoProcessor,
    VideoProcessorConfig,
)
from src.utils.llm_utils import OpenAIChat
from src.utils.prompts.prompts import PromptCollection

prompts = PromptCollection()
video_summarization_prompt_2 = prompts["2"]

TRANSCRIPT_CONTENTS = "This is a transcript."
DUMMY_FPS = 25
DUMMY_RESOLUTION = 720
DUMMY_LENGTH = 5  # seconds

PathObj = Union[str, Path]


class DummyVideo(Video):
    def getTranscript(self, *args, **kwargs) -> str:
        return TRANSCRIPT_CONTENTS

    @classmethod
    def createDummyVideo(
        cls, name: str, output_dir: PathObj, *args, **kwargs
    ) -> "DummyVideo":
        # Create metadata, then create the video.
        metadata = VideoMetadata(name=name)
        return cls(metadata, output_dir=output_dir, *args, **kwargs)

    @classmethod
    def createDummyVideoFile(
        cls,
        output_dir: PathObj,
        name: Optional[str] = "dummy_video.mp4",
    ) -> Path:
        """Creates a dummy video file.

        Args:
            name (str): The name of the video file.
            output_dir (PathObj): The output directory.

        Returns:
            Path: The path to the dummy video file.
        """
        size = DUMMY_RESOLUTION, DUMMY_RESOLUTION * 16 // 9, 3
        duration = DUMMY_LENGTH  # seconds
        fps = DUMMY_FPS
        out = cv2.VideoWriter(
            str(Path(output_dir) / name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (size[1], size[0]),
            True,
        )
        for _ in range(fps * duration):
            data = np.random.randint(0, 256, size, dtype="uint8")
            out.write(data)
        out.release()
        return Path(output_dir) / name


class DummyOpenAIChat(OpenAIChat):
    def processMessage(self, message: str) -> str:
        messages = self.getMessageHistory()
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": message})
        return message

    @classmethod
    def createDummyChat(cls, *args, **kwargs) -> "DummyOpenAIChat":
        return cls([], *args, **kwargs)


class TestVideoProcessorConfig(unittest.TestCase):
    def setUp(self):
        self.config = VideoProcessorConfig(
            "random_0.5", "/path/to/output", "test prompt"
        )

    def sampling_policy_test_helper(self, test_cases, policy_class):
        for policy_str, policy_value in test_cases:
            with self.subTest(
                f"{policy_class.__name__} with input being {policy_str}",
                policy_str=policy_str,
            ):
                policy = self.config._retrieve_sampling_policy(policy_str)
                self.assertIsInstance(policy, policy_class)
                if policy_value is not None:
                    self.assertEqual(
                        policy_value,
                        policy.value,
                    )
                else:
                    self.assertEqual(
                        policy.value,
                        policy.default_value,
                    )

    def test_retrieve_sampling_policy_random(self):
        test_cases = [
            (RandomSamplingPolicy(), None),
            ("random", None),
            ("random_0.25", 0.25),
            ("random 0.25", 0.25),
            ("random0.25", 0.25),
            ("r", None),
            ("r_0.25", 0.25),
            ("r 0.25", 0.25),
            ("r0.25", 0.25),
            ("random_1/4", 1 / 4),
            ("random 1/4", 1 / 4),
            ("random1/4", 1 / 4),
            ("r_1/4", 1 / 4),
            ("r 1/4", 1 / 4),
            ("r1/4", 1 / 4),
        ]
        self.sampling_policy_test_helper(test_cases, RandomSamplingPolicy)

    def test_retrieve_sampling_policy_uniform(self):
        test_cases = [
            (UniformFramesSamplingPolicy(), None),
            ("uniform", None),
            ("uniform_12", 12),
            ("uniform 12", 12),
            ("uniform12", 12),
            ("u", None),
            ("u_12", 12),
            ("u 12", 12),
            ("u12", 12),
        ]
        self.sampling_policy_test_helper(
            test_cases, UniformFramesSamplingPolicy
        )

    def test_retrieve_sampling_policy_uniform(self):
        test_cases = [
            (UniformSecondsSamplingPolicy(), None),
            ("second", None),
            ("second_0.1", 0.1),
            ("second 0.1", 0.1),
            ("second0.1", 0.1),
            ("seconds", None),
            ("seconds_0.1", 0.1),
            ("seconds 0.1", 0.1),
            ("seconds0.1", 0.1),
            ("s", None),
            ("s_0.1", 0.1),
            ("s 0.1", 0.1),
            ("s0.1", 0.1),
            ("second_1/10", 1 / 10),
            ("second 1/10", 1 / 10),
            ("second1/10", 1 / 10),
            ("seconds_1/10", 1 / 10),
            ("seconds 1/10", 1 / 10),
            ("seconds1/10", 1 / 10),
            ("s_1/10", 1 / 10),
            ("s 1/10", 1 / 10),
            ("s1/10", 1 / 10),
        ]
        self.sampling_policy_test_helper(
            test_cases, UniformSecondsSamplingPolicy
        )


class TestVideoProcessor(unittest.TestCase):
    def test_summarize(self):
        """Test the summarize method."""
        # Create a dummy processed directory using tempdir.
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a dummy chat.
            chat = DummyOpenAIChat.createDummyChat()
            # Create a dummy video processor.
            vp_config = VideoProcessorConfig(
                sampling_policy="u10",
                output_dir=tempdir,
                prompt=video_summarization_prompt_2,
            )
            vp = VideoProcessor(vp_config)
            # Create a dummy video.
            input_video = DummyVideo.createDummyVideo(
                "test_for_summarize", tempdir
            )
            # Summarize the video.
            vp.summarize(input_video, custom_chat=chat, save_transcript=True)
            # Check that the processed directory was created.
            tempdir = Path(tempdir)
            self.assertTrue(tempdir.is_dir())
            self.assertTrue((tempdir / "test_for_summarize").is_dir())
            self.assertTrue(
                (tempdir / "test_for_summarize" / "transcript").is_dir()
            )
            self.assertTrue(
                (tempdir / "test_for_summarize" / "summary").is_dir()
            )

            # Check that the transcript file was created.
            output_file = (
                tempdir
                / "test_for_summarize"
                / "transcript"
                / "transcript.txt"
            )
            self.assertTrue(output_file.exists())

            # Check that the summary file was created.
            output_file = (
                tempdir / "test_for_summarize" / "summary" / "summary.txt"
            )
            self.assertTrue(output_file.exists())
            # Check that the output file contains the summary (same as transcript for this test).
            with open(output_file, "r") as f:
                self.assertEqual(f.read(), TRANSCRIPT_CONTENTS)

    def test_summarize_2(self):
        """Test the summarize method with prompt_id."""
        # Create a dummy processed directory using tempdir.
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a dummy chat.
            chat = DummyOpenAIChat.createDummyChat()
            # Create a dummy video processor.
            vp_config = VideoProcessorConfig(
                sampling_policy="u10",
                output_dir=tempdir,
                prompt_id="2",
            )
            vp = VideoProcessor(vp_config)
            # Create a dummy video.
            input_video = DummyVideo.createDummyVideo(
                "test_for_summarize", tempdir
            )
            # Summarize the video.
            vp.summarize(input_video, custom_chat=chat, save_transcript=True)
            # Check that the processed directory was created.
            tempdir = Path(tempdir)
            self.assertTrue(tempdir.is_dir())
            self.assertTrue((tempdir / "test_for_summarize").is_dir())
            self.assertTrue(
                (tempdir / "test_for_summarize" / "transcript").is_dir()
            )
            self.assertTrue(
                (tempdir / "test_for_summarize" / "summary").is_dir()
            )

            # Check that the transcript file was created.
            output_file = (
                tempdir
                / "test_for_summarize"
                / "transcript"
                / "transcript.txt"
            )
            self.assertTrue(output_file.exists())

            # Check that the summary file was created.
            output_file = (
                tempdir / "test_for_summarize" / "summary" / "summary.txt"
            )
            self.assertTrue(output_file.exists())
            # Check that the output file contains the summary (same as transcript for this test).
            with open(output_file, "r") as f:
                self.assertEqual(f.read(), TRANSCRIPT_CONTENTS)


class TestSamplingPolicy(unittest.TestCase):
    def create_dummy_video(self, tempdir):
        video_file_path = DummyVideo.createDummyVideoFile(tempdir)
        return video_file_path

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.video_file_path = str(self.create_dummy_video(self.tempdir.name))
        video_capture = cv2.VideoCapture(self.video_file_path)
        self.assertEqual(video_capture.get(cv2.CAP_PROP_FPS), DUMMY_FPS)
        self.assertEqual(
            video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
            DUMMY_FPS * DUMMY_LENGTH,
        )
        video_capture.release()

    def sampling_policy_test_helper(
        self, policy_input, expected_indeces, sampling_policy_class
    ):
        """Helper function for testing sampling policies."""
        video_capture = cv2.VideoCapture(self.video_file_path)
        sampling_policy = sampling_policy_class(policy_input)
        indeces = sampling_policy.getSamplingIndeces(video_capture)
        self.assertEqual(
            indeces,
            expected_indeces,
        )
        frames, _ = sampling_policy.sample(video_capture)
        self.assertEqual(len(frames), len(expected_indeces))
        for frame, idx in zip(frames, expected_indeces):
            self.assertEqual(
                frame.image.shape,
                (DUMMY_RESOLUTION, DUMMY_RESOLUTION * 16 // 9, 3),
            )
            self.assertEqual(frame.order, idx)
        video_capture.release()

    def test_uniform_seconds_sampling_policy_1(self):
        """Test the UniformSecondsSamplingPolicy class."""
        self.sampling_policy_test_helper(
            1,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, DUMMY_FPS * 1)],
            UniformSecondsSamplingPolicy,
        )

    def test_uniform_seconds_sampling_policy_2(self):
        """Test the UniformSecondsSamplingPolicy class."""
        self.sampling_policy_test_helper(
            0.5,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, DUMMY_FPS // 2)],
            UniformSecondsSamplingPolicy,
        )

    def test_uniform_seconds_sampling_policy_3(self):
        """Test the UniformSecondsSamplingPolicy class."""
        self.sampling_policy_test_helper(
            0.1,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, DUMMY_FPS // 10)],
            UniformSecondsSamplingPolicy,
        )

    def test_uniform_frames_sampling_policy_1(self):
        """Test the UniformFramesSamplingPolicy class."""
        self.sampling_policy_test_helper(
            1,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, 1)],
            UniformFramesSamplingPolicy,
        )

    def test_uniform_frames_sampling_policy_2(self):
        """Test the UniformFramesSamplingPolicy class."""
        self.sampling_policy_test_helper(
            10,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, 10)],
            UniformFramesSamplingPolicy,
        )

    def test_uniform_frames_sampling_policy_3(self):
        """Test the UniformFramesSamplingPolicy class."""
        self.sampling_policy_test_helper(
            100,
            [_ for _ in range(0, DUMMY_FPS * DUMMY_LENGTH, 100)],
            UniformFramesSamplingPolicy,
        )
