# lit_video_stream component

This ⚡ [Lightning component](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init component lit_video_stream
```

## To run lit_video_stream

First, install lit_video_stream (warning: this app has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/lit_video_stream
```

Once the app is installed, use it in an app:

```python
from lit_video_stream import TemplateComponent
import lightning as L


class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.lit_video_stream = TemplateComponent()

    def run(self):
        print(
            "this is a simple Lightning app to verify your component is working as expected"
        )
        self.lit_video_stream.run()


app = L.LightningApp(LitApp())
```
