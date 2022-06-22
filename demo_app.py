import os
import torch

import lightning as L
from lightning_app.storage import Drive

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
        

class ProxyWork(L.LightningWork):
    
    def __init__(self) -> None:
        super().__init__()
        self.features_drive = Drive("lit://features")
        
    def run(self, features_path):
        self.features_drive.get(features_path)
        
        if os.path.exists(features_path):
            print("Do something cool with the features!")
            features = torch.load(features_path)
            print(features)


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
        self.proxy_work = ProxyWork()

    def run(self):
        one_min = 'https://www.youtube.com/watch?v=8SQL4knuDXU'
        self.lit_video_stream.download(video_urls=[one_min])
        self.proxy_work.run(self.lit_video_stream.features_path)
        self._exit("Application End!")


app = L.LightningApp(LitApp())
