from pipeline.video_inference import run_video_inference

video_path = "data/two-rings.mp4"           # Update to your video
output_path = "output/results/test_out_2.mp4"  # Output location

run_video_inference(video_path, output_path)
