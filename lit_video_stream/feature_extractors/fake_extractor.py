"""Feature extractor used for debugging."""
import torch
from torchvision import transforms


class FakeFeatureExtractor:
    def run(self, frame_batch):
        tensor_transform = transforms.PILToTensor()
        frame_batch = [tensor_transform(x) for x in frame_batch]
        frame_batch = torch.stack(frame_batch)
        reduce_dims = list(range(frame_batch.dim()))[1:]
        frame_batch = frame_batch.sum(reduce_dims)
        return frame_batch
