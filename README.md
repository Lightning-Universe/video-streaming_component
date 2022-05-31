# lit_video_stream component
Extract features from any video while streaming from any internet URL.

- Supports passing in arbitrary feature extractor models.
- Enables custom stream processors.
- Any accelerator (GPU/TPU/IPU) (single and multi-device)

## Supported Feature extractors
- any vision model from Open AI

## Supported stream processors
- YouTube
- Any video from a URL

## To run lit_video_stream
First, install lit_video_stream (warning: this app has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/lit_video_stream
```

Once the app is installed, use it in an app:

```python
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
```

## make your own progress bar
To pass in any kind of progress bar (including streaming to a web UI), pass in a class that implements update and reset

```python
from tqdm import tqdm

class TQDMProgressBar:
    def __init__(self) -> None:
        self._prog_bar = None

    def update(self, current_frame):
        self._prog_bar.update(1)

    def reset(self, total_frames):
        if self._prog_bar is not None:
            self._prog_bar.close()
        self._prog_bar = tqdm(total=total_frames)
```

```python
import requests

class StreamingProgressBar:
    def update(self, current_frame):
        r = requests.post('http://your/url', json={"current_frame": current_frame})

    def reset(self, total_frames):
        r = requests.post('http://your/url', json={"total_frames": total_frames})
```


## TODO:
[ ] Multi-node
