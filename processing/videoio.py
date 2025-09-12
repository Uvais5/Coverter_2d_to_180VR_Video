import os
from moviepy.editor import VideoFileClip, ImageSequenceClip


def probe_fps(video_path: str) -> float:
    """Get FPS of video."""
    clip = VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    return fps


def extract_frames(in_path: str, out_dir: str, size: int = 1024) -> None:
    """Extract frames from video and save as PNGs (fast)."""
    os.makedirs(out_dir, exist_ok=True)
    clip = VideoFileClip(in_path)

    # Scale shortest side to `size`
    w, h = clip.size
    scale = size / min(w, h)
    clip_resized = clip.resize(scale)

    # Write frames directly (MUCH faster than loop + PIL)
    clip_resized.write_images_sequence(os.path.join(out_dir, "f_%06d.png"))

    clip.close()
    clip_resized.close()


def extract_audio(in_path: str, out_path: str) -> None:
    """Extract audio from video (fast)."""
    clip = VideoFileClip(in_path)
    if clip.audio:
        clip.audio.write_audiofile(out_path, codec="aac", fps=44100)
    clip.close()


def encode_video_from_frames(frames_dir: str, fps: float, out_path: str) -> None:
    """Encode video from extracted frames."""
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", fps=fps, audio=False)
    clip.close()


def mux_audio(video_path: str, audio_path: str, out_path: str) -> None:
    """Mux video + audio together."""
    video_clip = VideoFileClip(video_path)
    audio_clip = VideoFileClip(audio_path).audio
    final = video_clip.set_audio(audio_clip)
    final.write_videofile(out_path, codec="libx264", audio_codec="aac")
    video_clip.close()
    audio_clip.close()
    final.close()
