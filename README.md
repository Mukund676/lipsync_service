# Serverless LipSync Service

An end-to-end pipeline that generates lip-synced videos from an input video and audio file. This project deploys **MuseTalk** on **Modal** serverless GPU infrastructure, wrapped in a **FastAPI** interface.

## 🚀 Features
* **Serverless Infrastructure:** Auto-scaling GPU inference using Modal.
* **Model:** MuseTalk (Real-time, high-fidelity lip synchronization).
* **Optimization:** Automatic 25fps frame-rate conversion and audio pre-processing.
* **Architecture:** Asynchronous "Manager-Worker" pattern to handle long-running inference tasks without HTTP timeouts.

## 🛠️ Tech Stack
* **Infrastructure:** Modal (Python SDK)
* **Model:** MuseTalk (PyTorch/Diffusers)
* **API:** FastAPI
* **Hardware:** NVIDIA A10G (Remote)

## 📦 Setup & Deployment

### 1. Prerequisites
* Python 3.10+
* Modal Account
* `pip install modal requests`

### 2. Setup Modal
Authenticate your local environment with Modal:
```bash
modal setup
```

### 3. Deploy the Service
Deploy the inference server to the cloud. This triggers a remote build that downloads the model weights (~10GB) and caches them.
```bash
modal deploy app.py
```

_Note: The first deployment may take 5-10 minutes. Subsequent deployments are faster due to caching._

## 🏃 usage

### Running the Benchmark Script

I have provided a client-side script to test latency and download the result automatically.
1. Place a `test_video.mp4` and `test_audio.wav` in the root directory.

2. Update the `API_URL` in `benchmark.py` with your deployed Modal URL.

3. Run
```bash
python benchmark.py
```

**Direct API Usage (cURL)**

You can also trigger the inference manually with the REST API.

```bash
curl -X POST "https://YOUR-MODAL-URL/generate" \
  -F "video=@test_video.mp4" \
  -F "audio=@test_audio.wav"
```

## 📊 Evaluation Strategy & Metrics

To objectively measure the performance of this lip-sync service, I propose the following metrics based on the SyncNet framework:

1. **LSE-D (Lip Sync Error - Distance):** 
    * Measures the distance between visual lip embeddings and audio embeddings.
    * **Target:** Lower is better (Ideal < 7.0).
2. **LSE-C (Lip Sync Error - Confidence):**
    * Measures the confidence that the audio and video belong together.
    * **Target:** Higher is better (Ideal > 6.0).
3. **Visual Quality (FID)**
    * Fréchet Inception Distance measures the realism of generated frames against real video frames.
    * _Note:_ Since MuseTalk outputs 256x256 crops, adding a Face Restoration step (like GFPGAN) is recommended for production use cases to improve FID.