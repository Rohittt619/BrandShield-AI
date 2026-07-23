<div align="center">

# 🛡️ BrandShield-AI — Multimodal Counterfeit Logo Detection & Brand Protection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-green.svg)](https://opencv.org/)
[![Google Gemini API](https://img.shields.io/badge/Google--Gemini-1.5--Pro-blue.svg)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**BrandShield-AI** is an enterprise-grade AI system designed for intellectual property protection, counterfeit logo detection, and brand integrity enforcement. Powered by **OpenCV Feature Matching (ORB/Canny Edges)**, **MobileNetV2 Deep Learning**, and **Google Gemini 1.5 Pro Multimodal Vision API**, it identifies fake logos, altered trademarks, and counterfeit merchandise in real time, generating executive forensic verification certificates (PDF).

</div>

---

# 🧠 System Architecture

```text
Uploaded Logo Image
        │
        ▼
OpenCV Pre-processing & Canny Edge Density
        │
        ▼
ORB Feature Keypoint Extraction (500 Keypoints)
        │
        ▼
Google Gemini 1.5 Pro Vision Multimodal Inspection
        │
        ▼
Real-Time Authenticity Scoring & Threat Classification
        │
        ▼
Visual Edge & Feature Heatmap Generation
        │
        ▼
ReportLab PDF Forensic Certificate Export
```

---

# ✨ Enterprise Features

- 🔬 **Hybrid Computer Vision & Multimodal AI**: Combines OpenCV structural edge analysis, ORB feature keypoints, and Google Gemini 1.5 Pro Multimodal Vision.
- 🎨 **Visual Forensic Heatmaps**: Displays side-by-side comparisons of Uploaded Image, ORB Keypoint Maps, and Canny Edge Heatmap Overlays.
- 🏢 **Multi-Brand Support**: Built-in verification profiles for major global brands (**Nike, Adidas, Apple, Starbucks, Gucci, Louis Vuitton, Rolex, Puma, Samsung**).
- 📜 **Executive PDF Forensic Verification Certificates**: Generates downloadable ReportLab PDF audit certificates complete with authenticity scores and threat classifications.
- 📊 **Brand Threat Database**: Interactive history portal tracking counterfeit incidents and risk trends.

---

# 🛠️ Directory Layout

```text
BrandShield-AI/
├── app.py                             # Main Streamlit Enterprise Web Portal
├── requirements.txt                   # Production Dependencies
├── README.md                          # Production GitHub Documentation
│
├── src/
│   └── brandshield_ai/
│       ├── __init__.py
│       ├── detector_engine.py         # Hybrid Multimodal Engine (OpenCV + Gemini Vision)
│       └── forensic_report.py         # ReportLab PDF Verification Generator
│
└── scripts/
    ├── train_classifier.py            # Deep Learning Model Training Script
    ├── evaluate_classifier.py         # Evaluation & Confusion Matrix
    └── inference.py                   # CLI Model Inference
```

---

# ⚙️ Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Rohittt619/BrandShield-AI.git
cd BrandShield-AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### 6. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

# 👨‍💻 Author

## Rohit Rathod

🎓 **B.Tech (Data Science)**  
💼 **Aspiring Data Analyst | Data Engineer | Data Scientist | AI Specialist**

- **GitHub**: [https://github.com/Rohittt619](https://github.com/Rohittt619)
- **LinkedIn**: [https://www.linkedin.com/in/rohit-rathod-19442a228/](https://www.linkedin.com/in/rohit-rathod-19442a228/)
- **Portfolio**: [https://rohittt619.github.io/](https://rohittt619.github.io/)

---

# 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
