import lightning as L
from lit_video_stream import LitVideoStream
from lit_video_stream.feature_extractors import OpenAIClip
from lit_video_stream.stream_processors import YouTubeStreamProcessor
from tqdm import tqdm


class PBar:
    def __init__(self) -> None:
        self._prog_bar = None

    def cli_prog_bar(self, current_frame, total_frames):
        if self._prog_bar is None:
            self._prog_bar = tqdm(total=total_frames)
        self._prog_bar.update(1)

class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self._pbar = PBar()
        self.lit_video_stream = LitVideoStream(
            feature_extractor=OpenAIClip(batch_size=256),
            stream_processor=YouTubeStreamProcessor(),
            prog_bar_fx=self._pbar.cli_prog_bar,
            process_every_n_frame=10,
            num_batch_frames=256
        )

    def run(self):
        self.lit_video_stream.download(video_url='https://www.youtube.com/watch?v=8SQL4knuDXU')


app = L.LightningApp(LitApp())

if __name__ == '__main__':

    class PBar:
        def __init__(self) -> None:
            self._prog_bar = None

        def _fake_progress_bar(self, current_frame, total_frames):
            if self._prog_bar is None:
                self._prog_bar = tqdm(total=total_frames)
            self._prog_bar.update(1)
    
    pb = PBar()
    lit_video_stream = LitVideoStream(
        feature_extractor=OpenAIClip(batch_size=256),
        stream_processor=YouTubeStreamProcessor(),
        prog_bar_fx=pb._fake_progress_bar,
        process_every_n_frame=30,
        num_batch_frames=256
    )

    lit_video_stream.download(video_url='https://www.youtube.com/watch?v=8SQL4knuDXU')