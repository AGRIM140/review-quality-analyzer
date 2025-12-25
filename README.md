# AI Product Review Quality Analyzer

## Features
- BERT-based fake review detection
- Attention visualization (token-level)
- GPU-optimized inference (FP16 + TorchScript)
- Dockerized deployment
- AWS EC2 GPU ready

## Tech Stack
Python, PyTorch, BERT, Streamlit, Docker, AWS

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training/inference)
- Docker (for containerized deployment)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python 02_bert_train.py
   ```
   This will:
   - Load data from `data/review.csv`
   - Train a BERT model for fake review detection
   - Save model to `models/bert_fake/`
   - Save tokenizer to `models/bert_fake/tokenizer/`

3. **Optional: Create TorchScript traced model (for faster inference):**
   ```bash
   python 04_trace_model.py
   ```
   This creates `models/bert_fake/bert_traced.pt` for optimized inference.

## Run Locally

```bash
streamlit run 01_app.py
```

The app will be available at `http://localhost:8501`

## Docker Deployment

### GPU Deployment (Recommended)
```bash
docker build -t review-analyzer .
docker run --gpus all -p 8501:8501 review-analyzer
```

### CPU Deployment
For CPU-only deployment, modify the Dockerfile to use a CPU base image:
```dockerfile
FROM python:3.12-slim
```
Then build and run:
```bash
docker build -t review-analyzer .
docker run -p 8501:8501 review-analyzer
```

## Project Structure
```
product-review-analyzer/
├── 01_app.py              # Streamlit web application
├── 02_bert_train.py       # Model training script
├── 03_attention.py        # Attention extraction utilities
├── 04_trace_model.py      # TorchScript model tracing
├── attention_utils.py     # Attention visualization helper
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container definition
├── data/
│   └── review.csv        # Training dataset
└── models/
    └── bert_fake/        # Trained model and tokenizer
```

## Notes
- The model is trained on a small dataset (50 samples) for demonstration
- For production use, train on a larger, more diverse dataset
- GPU acceleration significantly improves inference speed
- The fake review detection uses a simple heuristic: 5-star reviews with <40 characters are labeled as fake
