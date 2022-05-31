from pytube import YouTube


class YouTubeStreamProcessor:
    def run(self, video_url):
        yt = YouTube(video_url)
        streams = yt.streams.filter(adaptive=True, subtype='mp4', resolution='360p', only_video=True)
        return streams[0].url
