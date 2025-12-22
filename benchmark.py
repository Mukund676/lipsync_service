import requests
import time
import sys
import os

# REPLACE THIS with your actual Modal deployment URL
# You get this when you run `modal deploy app.py`
API_URL = "https://mukund676--lipsync-service-fastapi-entrypoint.modal.run"

def run_benchmark(video_path, audio_path):
    print(f"--- Starting Benchmark ---")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")

    # 1. Start Inference (Cold Start + Queue Time)
    start_time = time.time()
    
    files = {
        'video': open(video_path, 'rb'),
        'audio': open(audio_path, 'rb')
    }
    
    print("Uploading files and triggering inference...")
    response = requests.post(f"{API_URL}/generate", files=files)
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    data = response.json()
    job_id = data['job_id']
    call_id = data['call_id']
    print(f"Job started. ID: {job_id}")

    # 2. Poll for Completion
    status = "processing"
    while status == "processing":
        time.sleep(2) # Polling interval
        r = requests.get(f"{API_URL}/status/{call_id}")
        res_data = r.json()
        
        # Modal returns the function result directly when complete
        if "status" in res_data:
             status = res_data["status"]
        else:
             # If the function returned a dict directly (completed)
             status = "completed"

        sys.stdout.write(".")
        sys.stdout.flush()

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\nDone! Status: {status}")
    print(f"Total Runtime: {duration:.2f} seconds")

    if status == "completed":
        print("Downloading result...")
        # 3. Download Video
        down_r = requests.get(f"{API_URL}/download/{job_id}")
        if down_r.status_code == 200:
            output_filename = f"output_{job_id}.mp4"
            with open(output_filename, 'wb') as f:
                f.write(down_r.content)
            print(f"Saved to {output_filename}")
        else:
            print("Error downloading file.")

if __name__ == "__main__":
    # Ensure you have these dummy files locally to run the test
    if not os.path.exists("test_video.mp4") or not os.path.exists("test_audio.wav"):
        print("Error: Please provide 'test_video.mp4' and 'test_audio.wav' in this directory.")
    else:
        run_benchmark("test_video.mp4", "test_audio.wav")