from pytube import YouTube, extract


class YouTubeStreamProcessor:
    def run(self, video_url):
        yt = YouTube(video_url)
        streams = yt.streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)
        return streams[0].url

    def embed_link(self, video_url, time):
        video_id = extract.video_id(video_url)
        return f"https://www.youtube.com/embed/{video_id}?start={time}"
