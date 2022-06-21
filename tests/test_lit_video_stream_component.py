import os
from lit_video_stream.component import LitVideoStream


def test_placeholder_component():
    # TODO: improve testing
    video_stream_component = LitVideoStream()
    assert str(video_stream_component.features_path) == os.path.join(os.getcwd(), *[".storage", "features.pt"])
