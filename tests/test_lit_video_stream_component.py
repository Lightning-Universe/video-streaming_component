r"""
To test a lightning component:

1. Init the component.
2. call .run()
"""
from lit_video_stream.component import LitVideoStream


def test_placeholder_component():
    # TODO: improve testing
    video_stream_component = LitVideoStream()
    video_stream_component.run()
