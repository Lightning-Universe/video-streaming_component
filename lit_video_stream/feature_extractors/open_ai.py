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
    
    def search(self, search_query: str, results_count:int, video_features):
        video_frame_features = video_features['frame_features']
        fps = video_features['fps']
        num_skipped_frames = video_features['num_skipped_frames']

        with torch.no_grad():
            text_features = self.predictor.model.encode_text(openai_clip.tokenize(search_query))
            text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = 100.0 * torch.cat(video_frame_features) @ text_features.T
        _, best_photo_idx = similarities.topk(results_count, dim=0)

        # frames numbers
        search_results = best_photo_idx.cpu().numpy().tolist()

        frames_result = [result for sub_list in search_results for result in sub_list]

        results_in_ms = [
            round(
                frame * num_skipped_frames / fps * 1000
            )
            for frame in frames_result
        ]

        return results_in_ms
