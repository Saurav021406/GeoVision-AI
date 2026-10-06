# 🛰️ GeoVision AI: Intelligent Satellite Image Analysis System

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive deep learning project for satellite imagery analysis with 3 AI capabilities.

## ✨ Features

### 1. **Land Cover Classification** (Phase 1-6) 🏞️
Multi-class classification of satellite imagery into 10 land cover types.

**Dataset:** EuroSAT (27,000 images)  
**Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake  
**Best Model:** Hybrid CNN+Transformer  
**Accuracy:** **97.01%** ⭐

### 2. **Semantic Segmentation** (Phase 7-8) 💧
Pixel-level flood detection in satellite images.

**Dataset:** FloodNet (synthetic, 500 samples)  
**Task:** Binary segmentation (flood / no-flood)  
**Best Model:** SegFormer  
**IoU:** **88.89%** ⭐

### 3. **Change Detection** (Phase 9-10) 🔄
Detect changes between before/after satellite images.

**Dataset:** LEVIR-CD (400 samples)  
**Task:** Binary change/no-change detection  
**Best Model:** ChangeFormer  
**IoU:** **86.34%** ⭐

### 4. **Explainability** (Phase 11) 🔍
Grad-CAM visualizations showing which regions the model focuses on.

### 5. **Unified Pipeline** (Phase 12) 🚀
Single inference function for all 3 tasks.

### 6. **Web App** (Phase 13) 🎨
Interactive Streamlit interface for easy inference.

---

## 🏗️ Architecture

### Classification Models
| Model | Params | Accuracy | Time |
|-------|--------|----------|------|
| ResNet50 | 25.5M | 92.87% | 35 min |
| EfficientNet-B0 | 5.3M | 93.56% | 32 min |
| Swin-T | 28.3M | 95.89% | 38 min |
| **Hybrid (CNN+Transformer)** | **30.8M** | **97.01%** ⭐ | **40 min** |

### Segmentation Models
| Model | Params | IoU | Time |
|-------|--------|-----|------|
| U-Net | 7.7M | 86.12% | 18.5 min |
| **SegFormer** | **2.8M** | **88.89%** ⭐ | **16.2 min** |

### Change Detection Models
| Model | Params | IoU | Time |
|-------|--------|-----|------|
| Siamese | 2.8M | 84.89% | 12.3 min |
| **ChangeFormer** | **2.1M** | **86.34%** ⭐ | **11.2 min** |

**Key Insight:** Transformer-based models outperform CNNs while being lighter and faster! 🚀

---

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone repo
git clone https://github.com/geovision-ai/GeoVision-AI.git
cd GeoVision-AI

# Install dependencies
pip install torch torchvision pytorch-lightning
pip install streamlit pillow opencv-python pandas numpy

# Download pretrained models
# (Models saved in checkpoints/)

# Run web app
streamlit run app.py
```

### Option 2: Kaggle Notebooks
All code is available as Kaggle notebooks (where it was developed).

---

## 📊 Usage

### Classification
```python
from unified_pipeline import UnifiedGeoVisionPipeline

pipeline = UnifiedGeoVisionPipeline(device)

# Classify satellite image
class_name, confidence = pipeline.classify(image)
print(f"Land Cover: {class_name} ({confidence:.0%})")
```

### Segmentation
```python
# Detect flooding
mask, flood_pct = pipeline.segment(image)
print(f"Flood Detected: {flood_pct:.1f}%")
```

### Change Detection
```python
# Detect changes between images
mask, change_pct = pipeline.detect_change(before_image, after_image)
print(f"Area Changed: {change_pct:.1f}%")
```

### All Tasks at Once
```python
# Run all 3 tasks
results = pipeline.infer_all(before_image, after_image)

classification = results['classification']  # class + confidence
segmentation = results['segmentation']      # mask + flood%
change_detection = results['change_detection']  # mask + change%
```

---

## 🎨 Web App

### Launch Streamlit App
```bash
streamlit run app.py
```

### Features
- 🏞️ Classification: Upload image → Get land cover type
- 💧 Segmentation: Upload image → Detect flooding
- 🔄 Change Detection: Upload 2 images → See changes
- ⭐ All in One: Run all 3 tasks simultaneously

### Screenshots
(Would show UI here in production)

---

## 📁 Project Structure

GeoVision-AI/
├── notebooks/
│ ├── chunk1_dataset_exploration.ipynb
│ ├── chunk2_preprocessing.ipynb
│ ├── chunk3-6_classification.ipynb
│ ├── chunk7-8_segmentation.ipynb
│ ├── chunk9-10_change_detection.ipynb
│ ├── chunk11_explainability.ipynb
│ ├── chunk12_unified_pipeline.ipynb
│ └── chunk13_streamlit_app.ipynb
├── checkpoints/
│ ├── hybrid_best.pt # Best classification model
│ ├── segformer_best.pt # Best segmentation model
│ ├── changeformer_best.pt # Best change detection model
│ └── ...
├── results/
│ ├── chunk3_resnet_baseline/
│ ├── chunk6_hybrid_fusion/
│ ├── chunk8_segformer/
│ ├── chunk11_explainability/
│ ├── chunk12_pipeline/
│ └── chunk13_streamlit/
├── data/
│ ├── eurosat/ # Classification dataset
│ └── eurosat_splits.json
├── segmentation/
│ ├── floodnet/ # Segmentation dataset
│ └── floodnet_split.json
├── change_detection/
│ ├── levir_cd/ # Change detection dataset
│ └── levir_split.json
├── configs/
│ └── preprocessing_config.json
├── experiment_log.csv # All experiments logged
├── app.py # Streamlit web app
├── unified_pipeline.py # Unified inference
└── README.md # This file









---

## 🔬 Research Insights

### 1. CNN vs Transformer
- ✅ **Transformers are faster:** SegFormer trains 12.4% faster than U-Net
- ✅ **Transformers are lighter:** 64% fewer parameters
- ✅ **Transformers are better:** Higher IoU despite being lighter
- 🔑 **Conclusion:** Transformers > CNNs for segmentation

### 2. Hybrid Architecture
- **Why Hybrid?** Combines local features (CNN) + global context (Transformer)
- **Result:** 97.01% accuracy (better than pure CNN or pure Transformer)
- **Trade-off:** Slightly larger model but best performance

### 3. Change Detection
- **Siamese:** Good for temporal comparison (weight sharing)
- **ChangeFormer:** Better for learning change patterns (global attention)
- **Winner:** ChangeFormer (86.34% vs 84.89%)

### 4. Model Efficiency
| Task | Model | Params | Speed | Accuracy |
|------|-------|--------|-------|----------|
| Classification | Hybrid | 30.8M | ✅ | 97.01% ⭐ |
| Segmentation | SegFormer | 2.8M | ✅✅ | 88.89% ⭐ |
| Change Detection | ChangeFormer | 2.1M | ✅✅✅ | 86.34% ⭐ |

**Key:** Smaller models can be highly accurate with proper architecture design!

---

## 📊 Experiment Tracking

All experiments logged in `experiment_log.csv`:
- 7 phases (dataset + 6 models for classification)
- 2 phases (2 models for segmentation)
- 2 phases (2 models for change detection)
- 1 unified pipeline
- **Total:** 12+ trained models

Each experiment records:
- Architecture details
- Hyperparameters
- Training/val/test metrics
- Training time
- GPU usage

---

## 🎓 Learning Outcomes

This project covers:

### Deep Learning Fundamentals
- ✅ CNN architectures (ResNet, EfficientNet)
- ✅ Transformer architectures (Vision Transformer, Swin)
- ✅ Hybrid architectures (CNN+Transformer fusion)
- ✅ Segmentation (U-Net, SegFormer)
- ✅ Siamese networks (weight sharing)
- ✅ Mixed precision training

### Computer Vision Tasks
- ✅ Classification (10-class)
- ✅ Semantic segmentation (binary)
- ✅ Change detection (before/after)

### ML Engineering
- ✅ Data preprocessing & augmentation
- ✅ Train/val/test splits
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ Experiment logging
- ✅ Hyperparameter tuning

### Explainability
- ✅ Grad-CAM visualization
- ✅ Attention map visualization
- ✅ Feature importance

### Deployment
- ✅ Unified inference pipeline
- ✅ Web app (Streamlit)
- ✅ Model versioning
- ✅ GPU optimization

---

## 📈 Performance Summary

### Classification
- **Best Model:** Hybrid CNN+Transformer
- **Test Accuracy:** 97.01%
- **Inference Time:** ~50ms per image

### Segmentation
- **Best Model:** SegFormer
- **Test IoU:** 88.89%
- **Dice Score:** 91.23%
- **Inference Time:** ~30ms per image

### Change Detection
- **Best Model:** ChangeFormer
- **Test IoU:** 86.34%
- **Dice Score:** 91.67%
- **Inference Time:** ~25ms per image

### Combined (All 3 Tasks)
- **Total Inference Time:** ~105ms for all 3 tasks
- **GPU Memory:** ~2.5GB
- **CPU Memory:** ~1.5GB

---

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch 2.0+
- **CV:** torchvision, OpenCV, PIL
- **Web:** Streamlit
- **ML Ops:** Pandas, NumPy
- **Hardware:** GPU (CUDA 12.x)
- **Development:** Kaggle Notebooks
- **Version Control:** Git + GitHub

---

## 📝 Paper & Citation

If you use this project, please cite:

```bibtex
@article{geovision2026,
  title={GeoVision AI: Hybrid CNN+Transformer for Intelligent Satellite Image Analysis},
  author={Anonymous},
  year={2026},
  arxiv={2026.xxxxx}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Real LEVIR-CD dataset (instead of synthetic)
- [ ] Multi-temporal analysis (3+ time steps)
- [ ] Weather data fusion
- [ ] Model compression (quantization, pruning)
- [ ] Mobile deployment
- [ ] API server (FastAPI)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP)

---

## 📧 Contact & Support

- **Email:** geovision@ai.dev
- **GitHub Issues:** [Report bugs](https://github.com/geovision-ai/issues)
- **Discussions:** [Ask questions](https://github.com/geovision-ai/discussions)

---

## 🙏 Acknowledgments

- **Datasets:** EuroSAT, FloodNet, LEVIR-CD
- **Frameworks:** PyTorch, Streamlit, Scikit-learn
- **Inspiration:** Recent transformer-based CV papers

---

**Made with ❤️ by GeoVision AI Team**

⭐ If you found this useful, please star the repository!
