# 🚀 GeoVision AI - Setup & Deployment Guide

## Local Setup

### 1. Prerequisites
- Python 3.8+
- GPU (recommended): CUDA 12.x, cuDNN 8.x
- 8GB+ RAM
- 5GB+ disk space

### 2. Installation

**Clone Repository:**
```bash
git clone https://github.com/YOUR_USERNAME/GeoVision-AI.git
cd GeoVision-AI
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Download Models

Download trained checkpoints and place in `checkpoints/`:
- `hybrid_best.pt` (97.01% accuracy)
- `segformer_best.pt` (88.89% IoU)
- `changeformer_best.pt` (86.34% IoU)

[Download from GitHub Releases](https://github.com/YOUR_USERNAME/GeoVision-AI/releases)

### 4. Run Web App

```bash
streamlit run app.py
```

This opens http://localhost:8501

---

## Cloud Deployment

### Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create new app
4. Connect GitHub repo
5. Select `app.py`
6. Deploy!

**Free tier:** Perfect for demo

### Docker (Production)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t geovision-ai .
docker run -p 8501:8501 geovision-ai
```

### AWS SageMaker

1. Upload notebook to SageMaker
2. Use `ml.p3.2xlarge` instance (GPU)
3. Run inference through SageMaker endpoint

### Google Cloud Run

```bash
gcloud run deploy geovision-ai \
  --source . \
  --memory 2Gi \
  --allow-unauthenticated
```

---

## Usage Examples

### Python API

```python
from unified_pipeline import UnifiedGeoVisionPipeline

pipeline = UnifiedGeoVisionPipeline(device)

# Classify
class_name, conf = pipeline.classify(image)

# Segment
mask, flood% = pipeline.segment(image)

# Detect changes
change_mask, change% = pipeline.detect_change(before, after)

# All at once
results = pipeline.infer_all(before, after)
```

### Command Line

```bash
# Classify image
python classify.py --image image.jpg

# Segment image
python segment.py --image image.jpg --output mask.png

# Detect changes
python detect_changes.py --before before.jpg --after after.jpg
```

### Web App

Simply open the Streamlit app and use the UI!

---

## Performance Benchmarks

### Classification (Hybrid Model)
- Input: 128×128 RGB
- Output: 10-class prediction
- Accuracy: 97.01%
- Time: ~50ms (GPU), ~500ms (CPU)

### Segmentation (SegFormer)
- Input: 256×256 RGB
- Output: 256×256 binary mask
- IoU: 88.89%
- Time: ~30ms (GPU), ~300ms (CPU)

### Change Detection (ChangeFormer)
- Input: 2×256×256 RGB images
- Output: 256×256 binary change mask
- IoU: 86.34%
- Time: ~25ms (GPU), ~250ms (CPU)

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size or image size in config

### Issue: Models not loading
**Solution:** Download checkpoints and place in `checkpoints/` folder

### Issue: Streamlit app slow
**Solution:** Use GPU machine (check `torch.cuda.is_available()`)

### Issue: Incorrect predictions
**Solution:** Ensure images are 8-bit RGB (0-255) or float (0-1)

---

## Contributing

See CONTRIBUTING.md for guidelines

---

## Support

- 📧 Email: geovision@ai.dev
- 💬 Discussions: GitHub Discussions tab
- 🐛 Bugs: GitHub Issues tab

---

Made with ❤️ by GeoVision AI
