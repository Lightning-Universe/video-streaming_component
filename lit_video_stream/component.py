import lightning as L
import cv2
from PIL import Image
import math
from lit_video_stream.feature_extractors.open_ai import OpenAIClip
from lit_video_stream.stream_processors.no_stream_processor import NoStreamProcessor


class LitVideoStream(L.LightningWork):
    def __init__(
        self, 
        feature_extractor=None, 
        stream_processor=None, 
        num_batch_frames=-1,
        process_every_n_frame=1,
        prog_bar_fx=None,
        length_limit=None
    ):
        """Downloads a video from a URL and extracts features using any custom model.
        Includes support for feature extraction in real-time.

        Arguments:

            feature_extractor: A LightningFlow that extracts features in its run method (NOT YET SUPPORTED)
            stream_processor: A function to extract streams from a video (NOT YET SUPPORTED)
            num_batch_frames: How many frames to use for every "batch" of features being extracted. -1 Waits for the full video
                to download before processing it. If memory constrained on the machine, use smaller batch sizes.
            process_every_n_frame: process every "n" frames. if skip_frames = 0, don't skip frames (ie: process every frame), 
                if = 1, then skip every 1 frame, if 2 then process every 2 frames, and so on.
            prog_bar_fx: function called with every new frame to update the progress bar.
            length_limit: limit how long videos can be
        """
        super().__init__()

        # we use Open AI clip by default
        self._feature_extractor = feature_extractor if feature_extractor is not None else OpenAIClip()

        # by default, we just return the input
        self._stream_processor = stream_processor if stream_processor is not None else NoStreamProcessor()

        self.length_limit = length_limit
        self.process_every_n_frame = process_every_n_frame
        if self.process_every_n_frame < 1:
            raise SystemError(f'process_every_n_frame cannot be < 1, you passed in {self.process_every_n_frame}')

        self._prog_bar_fx = prog_bar_fx if not None else lambda *x: x

        # nothing mod infinity is ever zero... it means, the whole video will process at once when it's downloaded.
        if num_batch_frames == -1:
            num_batch_frames = float('inf')
        self.num_batch_frames = num_batch_frames
        self.features = []

    def download(self, video_url):
        """Downloads a video and process in real-time"""
        self.run('download', video_url)

    def run(self, action, *args, **kwargs):
        if action == 'download':
            self._download(*args, **kwargs)

    def _download(self, video_url):
        # give the user a chance to split streams
        stream_url = self._stream_processor.run(video_url)

        # use cv2 to load video details for downloading
        capture = cv2.VideoCapture(stream_url)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = total_frames/fps

        total_frames = math.ceil(total_frames / self.process_every_n_frame)

        # apply video length limit
        if self.length_limit and (duration > self.length_limit):
            m = f"""
            Video length is limited to {self.length_limit} seconds.
            """
            raise ValueError(m)

        # do actual download and online extraction
        current_frame = 0
        self.features = []
        unprocessed_frames = []
        features = []
        while capture.isOpened():
            # update the progress
            self._prog_bar_fx(current_frame, total_frames)

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

        self.features = L.storage.Payload(features)
