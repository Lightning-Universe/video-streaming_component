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
    def __init__(self, model_type='ViT-B/32', batch_size=256, feature_dim=512):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.feature_dim = feature_dim

        model, preprocess = openai_clip.load(model_type)
        self.predictor = LightningInferenceModel(model, preprocess)
        self.trainer = pl.Trainer(accelerator='auto', devices=1)

    def run(self, frames):
        # PIL images -> torch.Tensor
        batch = torch.stack([self.predictor.preprocess(frame) for frame in frames])

        # dataset
        batch_size = min(len(batch), self.batch_size)
        dl = torch.utils.data.DataLoader(batch, batch_size=batch_size, num_workers=8)

        # ⚡ accelerated inference with PyTorch Lightning ⚡
        batch = self.trainer.predict(self.predictor, dataloaders=dl)

        # results
        batch = torch.cat(batch)
        return batch