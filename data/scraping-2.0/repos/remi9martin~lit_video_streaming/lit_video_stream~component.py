import imp
import pickle
from typing import List

import cv2
import lightning as L
from lightning.app.storage import Path
from PIL import Image

from lit_video_stream.feature_extractors.open_ai import OpenAIClip
from lit_video_stream.stream_processors.no_stream_processor import NoStreamProcessor


class LitVideoStream(L.LightningWork):
    def __init__(
        self,
        feature_extractor=None,
        stream_processor=None,
        num_batch_frames=-1,
        process_every_n_frame=1,
        prog_bar=None,
        length_limit=None,
        **kwargs,
    ):
        """Downloads a video from a URL and extracts features using any custom model. Includes support for feature
        extraction in real-time.

        Arguments:

            feature_extractor: A LightningFlow that extracts features in its run method (NOT YET SUPPORTED)
            stream_processor: A function to extract streams from a video (NOT YET SUPPORTED)
            num_batch_frames: How many frames to use for every "batch" of features being extracted. -1 Waits for the full video
                to download before processing it. If memory constrained on the machine, use smaller batch sizes.
            process_every_n_frame: process every "n" frames. if process_every_n_frame = 0, don't skip frames (ie: process every frame),
                if = 1, then skip every 1 frame, if 2 then process every 2 frames, and so on.
            prog_bar: A class that implements 2 methods: update and reset.
            length_limit: limit how long videos can be
        """
        super().__init__(**kwargs)

        # we use Open AI clip by default
        self._feature_extractor = (
            feature_extractor if feature_extractor is not None else OpenAIClip()
        )

        # by default, we just return the input
        self._stream_processor = (
            stream_processor if stream_processor is not None else NoStreamProcessor()
        )

        self.length_limit = length_limit
        self.process_every_n_frame = process_every_n_frame
        if self.process_every_n_frame < 1:
            raise SystemError(
                f"process_every_n_frame cannot be < 1, you passed in {self.process_every_n_frame}"
            )

        class NoPBAR:
            def update(self, *args):
                pass

            def reset(self, *args):
                pass

        self._prog_bar = prog_bar if prog_bar is not None else NoPBAR()

        # nothing mod infinity is ever zero... it means, the whole video will process at once when it's downloaded.
        if num_batch_frames == -1:
            num_batch_frames = float("inf")
        self.num_batch_frames = num_batch_frames
        self.features_path = None

    def run(self, video_urls: List):
        """Downloads a set of videos and processes them in real-time.

        Arguments:
            video_urls: a list of video URLs

        Return:
            a list with the matching features
        """
        features = {}

        # TODO: parallelize each video processing
        for video_url in video_urls:
            frame_features, fps = self._get_features(video_url)
            features[video_url] = {
                "frame_features": frame_features,
                "num_skipped_frames": self.process_every_n_frame,
                "fps": fps,
            }

        with open("features.p", "wb") as fp:
            pickle.dump(features, fp)

        # Use lightning.app.storage.Path to create a reference to the features
        # When running in the cloud on multiple machines, by simply passing this reference to another work,
        # it triggers automatically a transfer.
        self.features_path = Path("features.p")

    def _get_features(self, video_url):
        # give the user a chance to split streams
        stream_url = self._stream_processor.run(video_url)

        # use cv2 to load video details for downloading
        capture = cv2.VideoCapture(stream_url)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        # total_frames = math.ceil(total_frames / self.process_every_n_frame)

        # apply video length limit
        if self.length_limit and (duration > self.length_limit):
            m = f"""
            Video length is limited to {self.length_limit} seconds.
            """
            raise ValueError(m)

        # allow prog bar to reset
        self._prog_bar.reset(total_frames)

        # do actual download and online extraction
        current_frame = 0
        unprocessed_frames = []
        features = []
        while capture.isOpened():
            # update the progress
            self._prog_bar.update(current_frame)

            # get the frame
            ret, frame = capture.read()

            # add a new frame to process
            if ret:
                unprocessed_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break

            # process a batch of frames if requested by the user
            # if user said num_batch_frames = -1 then num_batch_frames is +inf which will never be 0
            if len(unprocessed_frames) % self.num_batch_frames == 0:

                # process the frames and clear the frame cache
                features.append(self._feature_extractor.run(unprocessed_frames))
                unprocessed_frames = []

            # advance frame
            current_frame += self.process_every_n_frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # process any leftover frames
        features.append(self._feature_extractor.run(unprocessed_frames))
        unprocessed_frames = []
        return features, fps
