# Serverless LipSync Service

An end-to-end pipeline that generates lip-synced videos from an input video and audio file. This project deploys **MuseTalk** on **Modal** serverless GPU infrastructure, wrapped in a **FastAPI** interface.

## 🚀 Features
* **Serverless Infrastructure:** Auto-scaling GPU inference using Modal.
* **Model:** MuseTalk (Real-time, high-fidelity lip synchronization).
* **Optimization:** Automatic 25fps frame-rate conversion and audio pre-processing.
* **Architecture:** Asynchronous "Manager-Worker" pattern to handle long-running inference tasks without HTTP timeouts.

## ⚖️ Model Trade Study
To select the best open-source model, I evaluated three state-of-the-art options based on visual quality, inference speed, and realism:

| Model | Pros | Cons | Verdict |
| :--- | :--- | :--- | :--- |
| **Wav2Lip** | Very fast inference; robust lip-sync accuracy. | Low resolution outputs (96x96); struggles with HD faces; strictly modifies lips without broader facial context. | ❌ Rejected (Low fidelity) |
| **SadTalker** | Generates full head motion from audio; good for static images. | Slower inference; expression can feel robotic ("uncanny valley"); often drifts from the original video identity. | ❌ Rejected (Unnatural motion) |
| **MuseTalk** | **High Fidelity (256x256 latent diffusion);** Real-time capable (30fps+ on A10G); modifying upper-face expressions for realism. | Requires higher VRAM (12GB+); sensitive to bounding box shifts. | ✅ **Selected** |

**Why MuseTalk?**
It strikes the best balance for a production service. Unlike Wav2Lip, it generates sharp, high-res mouth regions suitable for modern video standards. Unlike SadTalker, it preserves the original video's head pose and personality, only modifying the necessary facial region.

## 🛠️ Tech Stack & Hardware
* **Infrastructure:** Modal (Python SDK)
* **Framework:** PyTorch / Diffusers / FastAPI
* **GPU Hardware:** NVIDIA A10G (24GB VRAM)

### Hardware Selection Logic
I chose the **NVIDIA A10G** over the T4 or A100 for this specific workload:
* **A10G (Selected):** The "Sweet Spot." It provides 24GB VRAM (necessary to hold MuseTalk, VAE, and Whisper models simultaneously without offloading) and offers fast fp16 inference.
* **NVIDIA T4:** While cheaper, the T4 (16GB) is significantly slower for diffusion-based models and risks OOM (Out of Memory) errors during high-res generation.
* **NVIDIA A100:** While faster, the cost per hour is ~3-4x higher than A10G. The inference speed gain for this specific model (which is already near real-time on A10G) does not justify the extra cost.


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