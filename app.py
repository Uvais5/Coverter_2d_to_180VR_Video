import gradio as gr
import tempfile
import os
from processing.pipeline_small import VR180Pipeline
from moviepy.editor import VideoFileClip


def add_audio_to_video(input_video, vr_video_path):
    """Add original audio back to the processed VR video."""
    video_clip = VideoFileClip(vr_video_path)
    original_audio = VideoFileClip(input_video).audio
    video_clip = video_clip.set_audio(original_audio)
    out_path = vr_video_path.replace(".mp4", "_with_audio.mp4")
    video_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
    return out_path


def convert(in_video):
    if in_video is None:
        return None

    # Gradio sometimes passes a dict with 'name' or a path string
    if isinstance(in_video, dict):
        in_path = in_video.get("name") or in_video.get("tmp_path") or in_video.get("file")
    else:
        in_path = in_video

    if not in_path or not os.path.exists(in_path):
        return None

    # Create temporary working directory
    work = tempfile.mkdtemp(prefix="vr180_")

    # Initialize pipeline using GPU if available
    pipeline = VR180Pipeline(work, device="cuda")

    # Run VR180 processing
    vr_video = pipeline.run(in_path)

    # Add audio back
    final_video = add_audio_to_video(in_path, vr_video)

    return final_video


# === Gradio UI ===
iface = gr.Interface(
    fn=convert,
    inputs=gr.Video(label="Upload 2D Clip"),
    outputs=gr.File(label="Download VR180 Video"),
    title="2D → VR180 Converter",
    description="Upload a short 2D video and convert it into VR180 format for VR headsets."
)

if __name__ == "__main__":
    print("✅ Launching Gradio app...")
    iface.launch( server_port=7860, debug=True,share =True)
