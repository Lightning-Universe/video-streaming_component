"""
To test a lightning component:

1. Init the component.
2. call .run()
"""
from lit_video_stream import LitVideoStream


def test_placeholder_component():
    messenger = LitVideoStream()
    messenger.run()
    assert messenger.value == 1
