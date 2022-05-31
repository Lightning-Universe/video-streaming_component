import lightning as L
from lit_video_stream import LitVideoStream
from lit_video_stream.feature_extractors import OpenAIClip
from lit_video_stream.stream_processors import YouTubeStreamProcessor
from tqdm import tqdm


class PBar:
    def __init__(self) -> None:
        self._prog_bar = None

    def update(self, current_frame):
        self._prog_bar.update(1)

    def reset(self, total_frames):
        if self._prog_bar is not None:
            self._prog_bar.close()
        self._prog_bar = tqdm(total=total_frames)

class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.lit_video_stream = LitVideoStream(
            feature_extractor=OpenAIClip(batch_size=256),
            stream_processor=YouTubeStreamProcessor(),
            prog_bar=PBar(),
            process_every_n_frame=30,
            num_batch_frames=256,
        )

    def run(self):
        one_hour = 'https://www.youtube.com/watch?v=rru2passumI'
        one_min = 'https://www.youtube.com/watch?v=8SQL4knuDXU'
        self.lit_video_stream.download(video_urls=[one_min, one_min])


app = L.LightningApp(LitApp())
