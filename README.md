 ###VeritasAI: Ultimate Deepfake & Misinformation Detection Platform
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.95+-emerald.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/transformers)
[![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-orange.svg?logo=opencv&logoColor=white)](https://opencv.org)




VeritasAI is an advanced, enterprise-grade media authentication backend designed to combat deepfakes, voice cloning, and online misinformation. By combining deep learning classifiers, digital image forensics, computer vision metadata analysis, and real-time news indexing, VeritasAI calculates a comprehensive Authenticity Trust Score for images, video, and audio.
---
## Key Features
* **Multi-Format Deepfake Classification**: Automatic routing and analysis for Images (JPG, PNG, WebP), Videos (MP4, AVI, MOV), and Audio (MP3, WAV) using state-of-the-art vision and audio Transformers.
* **Live Face & Eye Detection (OpenCV)**: Analyzes facial geometry, counts eyes, checks reflections, and tracks multi-face swaps dynamically, outputting exact pixel bounding boxes for front-end rendering.
* **Live Web Fact-Checking (DDG)**: Queries text claims against real-time web results, looks for negative semantic keywords (e.g. debunked, hoax), and generates dynamic verification status reports.
* **Forensic Heatmaps (Error Level Analysis)**: Performs compression differential analysis (ELA) on uploaded images to highlight digitally altered areas (copy-paste, face-swaps).
* **Automated PDF Reports & QR Verification**: Automatically prints details to a professional report containing a verification QR code pointing back to the authentication page.
* **Dynamic Analytics**: Keeps live counts of total scans, verified real assets, and detected manipulations for the frontend dashboard.
---
## Technology Stack
* **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous, Type-safe endpoints)
* **ML Libraries**: [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/transformers)
* **Computer Vision**: [OpenCV](https://opencv.org/) (Haar Cascades for face/eye detection), [Pillow](https://python-pillow.org/)
* **Document Generation**: [ReportLab](https://www.reportlab.com/) (PDF engine), [Qrcode](https://pypi.org/project/qrcode/)
* **Web UI Interface**: [Gradio](https://gradio.app/) (Mounted directly inside FastAPI)
---
