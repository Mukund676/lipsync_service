import os
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import modal

# --- Configuration ---
MINUTES = 60
GPU_CONFIG = "A10G" 
vol = modal.Volume.from_name("lipsync-data", create_if_missing=True)

# --- Image Definition ---
def download_models():
    """
    Downloads weights and symlinks them to cover all hardcoded naming variations in MuseTalk.
    """
    import os
    import urllib.request
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download
    
    # 1. Define Paths
    base_path = "/root/MuseTalk/models"
    musetalk_path = f"{base_path}/musetalk"
    dwpose_path = f"{base_path}/dwpose"
    face_path = f"{base_path}/face-parse-bisent"
    vae_path = f"{base_path}/sd-vae-ft-mse"
    whisper_path = f"{base_path}/whisper"
    
    for p in [musetalk_path, dwpose_path, face_path, vae_path, whisper_path]:
        os.makedirs(p, exist_ok=True)

    # 2. Download DWPose
    print("Downloading DWPose...")
    hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.pth", local_dir=dwpose_path)
    
    # 3. Download Face Parsing
    print("Downloading Face Parsing...")
    hf_hub_download(repo_id="ManyOtherFunctions/face-parse-bisent", filename="79999_iter.pth", local_dir=face_path)
    hf_hub_download(repo_id="ManyOtherFunctions/face-parse-bisent", filename="resnet18-5c106cde.pth", local_dir=face_path)

    # 4. Download SD-VAE
    print("Downloading SD-VAE...")
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=vae_path)
    if not os.path.exists(f"{base_path}/sd-vae"):
        os.symlink(vae_path, f"{base_path}/sd-vae")

    # 5. Download Whisper
    print("Downloading Whisper...")
    snapshot_download(repo_id="openai/whisper-tiny", local_dir=whisper_path)
    url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
    if not os.path.exists(f"{whisper_path}/tiny.pt"):
        urllib.request.urlretrieve(url, f"{whisper_path}/tiny.pt")

    # 6. Download MuseTalk Main Weights
    print("Downloading MuseTalk Weights...")
    hf_hub_download(repo_id="TMElyralab/MuseTalk", subfolder="musetalk", filename="musetalk.json", local_dir=base_path)
    hf_hub_download(repo_id="TMElyralab/MuseTalk", subfolder="musetalk", filename="pytorch_model.bin", local_dir=base_path)

    if os.path.exists(f"{musetalk_path}/musetalk.json") and not os.path.exists(f"{musetalk_path}/config.json"):
        shutil.copy(f"{musetalk_path}/musetalk.json", f"{musetalk_path}/config.json")
    if os.path.exists(f"{musetalk_path}/pytorch_model.bin") and not os.path.exists(f"{musetalk_path}/unet.pth"):
        shutil.copy(f"{musetalk_path}/pytorch_model.bin", f"{musetalk_path}/unet.pth")
    if not os.path.exists(f"{base_path}/musetalkV15"):
        os.symlink(musetalk_path, f"{base_path}/musetalkV15")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.0.1", 
        "torchvision==0.15.2", 
        "torchaudio==2.0.2",
        "diffusers==0.24.0",
        "transformers==4.36.2",
        "huggingface_hub",
        "ffmpeg-python",
        "opencv-python",
        "scipy",
        "tqdm",
        "pydub",
        "librosa",
        "openmim",
        "pyyaml"
    )
    .run_commands(
        "mim install mmengine",
        "mim install 'mmcv==2.1.0'",
        "mim install 'mmdet>=3.1.0'",
        "mim install 'mmpose>=1.1.0'",
        "git clone https://github.com/TMElyralab/MuseTalk.git /root/MuseTalk",
    )
    .run_commands("pip install -r /root/MuseTalk/requirements.txt")
    .run_function(download_models)
)

app = modal.App("lipsync-service", image=image)
web_app = FastAPI()

@app.function(
    gpu=GPU_CONFIG, 
    timeout=10 * MINUTES,
    volumes={"/data": vol}
)
def run_inference_task(job_id: str, video_filename: str, audio_filename: str):
    import sys
    import yaml
    import shutil

    vol.reload()
    sys.path.append("/root/MuseTalk")
    
    base_dir = Path("/data")
    job_dir = base_dir / job_id
    video_path = job_dir / video_filename
    audio_path = job_dir / audio_filename
    
    # 1. Preprocessing
    video_25fps_path = job_dir / "input_25fps.mp4"
    print(f"Converting {video_filename} to 25fps...")
    subprocess.run([
        "ffmpeg", "-y", 
        "-i", str(video_path), 
        "-r", "25", 
        str(video_25fps_path)
    ], check=True)
    
    # 2. Prepare Config (Fresh Dictionary)
    job_config_path = job_dir / "config.yaml"
    config_data = {
        "task_0": {
            "video_path": str(video_25fps_path),
            "audio_path": str(audio_path),
            "bbox_shift": 0
        }
    }
    with open(job_config_path, "w") as f:
        yaml.dump(config_data, f)
        
    # 3. Run Inference
    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", str(job_config_path),
        "--result_dir", str(job_dir) 
    ]
    
    print(f"Starting inference for Job {job_id}...")
    try:
        subprocess.run(cmd, cwd="/root/MuseTalk", check=True)
        
        # MuseTalk output is hidden inside a 'v15' subfolder
        # We search recursively using rglob
        generated_files = list(job_dir.rglob("*.mp4"))
        
        output_files = []
        for f in generated_files:
            # Exclude inputs and temp files
            if f.name == video_filename or f.name == "input_25fps.mp4":
                continue
            if "temp" in f.name:
                continue
            output_files.append(f)
            
        if output_files:
            output_path = job_dir / "output.mp4"
            # Get the most recently modified file (the result)
            output_files.sort(key=os.path.getmtime, reverse=True)
            
            # Move the file from 'v15' folder to the main job folder
            print(f"Found output video: {output_files[0]}")
            shutil.move(str(output_files[0]), str(output_path))
            vol.commit()
            return {"status": "completed", "path": str(output_path)}
        else:
            # Debugging info in case it fails again
            all_files = [str(f) for f in job_dir.rglob("*")]
            raise Exception(f"No output video found. Files in dir: {all_files}")

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        return {"status": "failed", "error": str(e)}

@web_app.post("/generate")
async def generate(video: UploadFile = File(...), audio: UploadFile = File(...)):
    import uuid
    job_id = str(uuid.uuid4())
    job_dir = Path("/data") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = job_dir / video.filename
    audio_path = job_dir / audio.filename
    
    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
        
    vol.commit() 
    call = run_inference_task.spawn(job_id, video.filename, audio.filename)
    return {"job_id": job_id, "call_id": call.object_id}

@web_app.get("/status/{call_id}")
async def status(call_id: str):
    from modal.functions import FunctionCall
    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
        return result
    except TimeoutError:
        return {"status": "processing"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@web_app.get("/download/{job_id}")
async def download(job_id: str):
    output_path = Path("/data") / job_id / "output.mp4"
    vol.reload()
    if output_path.exists():
        return FileResponse(output_path, media_type="video/mp4", filename=f"lipsync_{job_id}.mp4")
    raise HTTPException(status_code=404, detail="Video not found")

@app.function(volumes={"/data": vol})
@modal.asgi_app()
def fastapi_entrypoint():
    return web_app