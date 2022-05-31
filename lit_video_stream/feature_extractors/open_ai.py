import clip as openai_clip
import math
import torch


class OpenAIClip:
    def __init__(self, model_type='ViT-B/32', batch_size=256, feature_dim=512):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.model = None
        self.preprocess = None

    def run(self, frames):
        batches = math.ceil(len(frames) / self.batch_size)
        batch_size = min(len(frames), self.batch_size)
        video_features = torch.empty([0, self.feature_dim], dtype=torch.float16)

        # load the model only once
        if self.model is None:
            self.model, self.preprocess = openai_clip.load(self.model_type)

        for i in range(batches):
            batch_frames = frames[i * batch_size : (i + 1) * batch_size]
            batch_preprocessed = torch.stack(
                [self.preprocess(frame) for frame in batch_frames]
            )
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            video_features = torch.cat((video_features, batch_features))

        return video_features