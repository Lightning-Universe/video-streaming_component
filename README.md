# lit_video_stream component
Extract features from any video while streaming from any video web link.

- Supports passing in arbitrary feature extractor models.
- Enables custom stream processors.
- Any accelerator (GPU/TPU/IPU) (single device).

## Supported Feature extractors
- any vision model from Open AI

## Supported stream processors
- YouTube
- Any video from a URL

## Install this component
```bash
lightning install component lightning/LAI-lit-video-streaming
```

## Use the component
Here's an example of using this component in an app

```python
import lightning as L
from lit_video_stream import LitVideoStream
from lit_video_stream.feature_extractors import OpenAIClip
from lit_video_stream.stream_processors import YouTubeStreamProcessor

class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.lit_video_stream = LitVideoStream(
            feature_extractor=OpenAIClip(batch_size=256),
            stream_processor=YouTubeStreamProcessor(),
            process_every_n_frame=30,
            num_batch_frames=256,
        )

    def run(self):
        one_min = 'https://www.youtube.com/watch?v=8SQL4knuDXU'
        self.lit_video_stream.download(video_urls=[one_min, one_min])
        if len(self.lit_video_stream.features) > 0:
            print('do something with the features')


app = L.LightningApp(LitApp())
```

## Add a progress bar
To track the progress of processing, implement a class that overrides "update" and "reset"

CLI progress bar
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

For a web server
```python
import requests

class StreamingProgressBar:
    def update(self, current_frame):
        r = requests.post('http://your/url', json={"current_frame": current_frame})

    def reset(self, total_frames):
        r = requests.post('http://your/url', json={"total_frames": total_frames})
```

and pass it in:
```python
self.lit_video_stream = LitVideoStream(
    feature_extractor=OpenAIClip(batch_size=256),
    stream_processor=YouTubeStreamProcessor(),
    process_every_n_frame=30,
    num_batch_frames=256,
    prog_bar=TQDMProgressBar()
)
```

## Add your own feature extractor
To pass in your own feature extractor, simply implement a class that overrides "extract_features"
For example, this feature extractor uses Open AI + PyTorch Lightning to accelerate feature extraction

```python
import clip as openai_clip
import torch
import pytorch_lightning as pl


class LightningInferenceModel(pl.LightningModule):
    def __init__(self, model, preprocess) -> None:
        super().__init__()
        self.model = model
        self.preprocess = preprocess

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_features = self.model.encode_image(batch)
        batch_features /= batch_features.norm(dim=-1, keepdim=True)

        return batch_features


class OpenAIClip:
    def __init__(self, model_type='ViT-B/32', batch_size=256, feature_dim=512, num_workers=1):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.num_workers = num_workers

        model, preprocess = openai_clip.load(model_type)
        self.predictor = LightningInferenceModel(model, preprocess)

        # PyTorch Lightning does not yet support distributed inference
        # when it does, use this one:    self.trainer = pl.Trainer(accelerator='auto')
        self.trainer = pl.Trainer(accelerator='auto', devices=1)

    def run(self, frames):
        # PIL images -> torch.Tensor
        batch = torch.stack([self.predictor.preprocess(frame) for frame in frames])

        # dataset
        batch_size = min(len(batch), self.batch_size)
        dl = torch.utils.data.DataLoader(batch, batch_size=batch_size, num_workers=self.num_workers)

        # ⚡ accelerated inference with PyTorch Lightning ⚡
        batch = self.trainer.predict(self.predictor, dataloaders=dl)

        # results
        batch = torch.cat(batch)
        return batch
```

## Add a stream processor
Stream processors allow you to process videos more efficiently. To add your own, simply pass in an object 
that implements "run".

Here's an example that creates a stream processor for YouTube

```python
from pytube import YouTube

class YouTubeStreamProcessor:
    def run(self, video_url):
        yt = YouTube(video_url)
        streams = yt.streams.filter(adaptive=True, subtype='mp4', resolution='360p', only_video=True)
        return streams[0].url
```

## TODO:
[ ] Multi-node
