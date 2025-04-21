from pipeline.video_inference import run_video_inference

video_path = "data/two-rings.mp4"
output_path = "output/results/tracked_video.mp4"

run_video_inference(video_path, output_path)
