from pipeline.video_inference import run_video_inference

video_path = "data/ring-on-hand.mp4"           # Update to your video
output_path = "output/results/test_out.mp4"  # Output location

run_video_inference(video_path, output_path)
