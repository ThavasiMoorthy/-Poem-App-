# ✍️ கவிதை வழிகாட்டி (Tamil Poem Generator)

A standalone Tamil poem generation application built with FastAPI and a custom GPT-based Transformer model.

## 🚀 Installation & Local Setup

To run this application locally, follow these steps:

### 1. Prerequisites
- Python 3.9 or higher
- Git LFS (for downloading the model weights)

### 2. Clone the Repository
```bash
git clone https://github.com/ThavasiMoorthy/-Poem-App-.git
cd -Poem-App-
```

### 3. Install Dependencies
We recommend using a virtual environment. The following command installs **PyTorch (CPU version)** and all other required libraries:

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` is configured to download the lightweight CPU version of PyTorch to save space and memory.*

### 4. Run the Application
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

## 🌐 Deployment (Render.com)

1. Connect your GitHub repository to **Render.com**.
2. Select **Web Service**.
3. Use the following settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Starter or higher (due to PyTorch memory requirements).

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **Model**: Custom Transformer (RoPE, SiLU)
- **AI Engine**: PyTorch & SentencePiece
- **Frontend**: React (Tailwind CSS + Glassmorphism)

---
Developed for **Thamizh Vazhikaati** Project.