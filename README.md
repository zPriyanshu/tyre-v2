# 🛞 Tyre OCR Specification Extractor

A modern, high-performance, open-source Python software suite that extracts key specifications (Brand, Size, Load Index, Speed Rating, DOT manufacture dates, and all raw text markings) from photos of tyre sidewalls. 

Powered by **Qwen-VL** (Alibaba's Vision-Language Model) running locally on your machine.

---

## 🌟 Features
- **Accurate Spec Parsing**: Extracts Brand, Model, Dimensions (e.g. `225/45ZR17`), Load/Speed ratings (e.g. `94W`), DOT codes, and calculates the tyre manufacturing date (e.g. Week 18, 2019).
- **Strict Structured Outputs**: Utilizes advanced visual prompting to ensure the model outputs valid JSON representation, backed by regex-based cleaning and fallback parsing.
- **Two Execution Interfaces**:
  1. **CLI (Command Line Interface)**: Fast, machine-readable extraction tool suitable for shell scripting and automation pipelines.
  2. **Premium Web GUI (Streamlit)**: Sleek, glassmorphism dark-themed dashboard showing a side-by-side comparison, special markings chips, raw JSON inspections, and downloadable CSV/JSON report exports.
- **Multi-Hardware Support**: Auto-detects and accelerates execution using Apple Silicon MPS (Metal Performance Shaders), NVIDIA CUDA, or Standard CPU.

---

## 🛠️ Installation & Setup

We recommend using Python 3.10 or 3.11.

### 1. Clone or Open the Repository
Ensure you are in the project folder:
```bash
cd /Users/Priyanshu/Documents/Projects/Tyre
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*Note: On your first model run, Hugging Face will automatically download the vision language weights (~5.5GB for `Qwen2-VL-2B-Instruct`). This will be cached locally on your drive so subsequent runs load instantly.*

---

## 🚀 Usage Guide

We have included a high-resolution test image in the repository under `images/sample_tyre.png` representing a Michelin tyre sidewall with embossed specs.

### Option A: Command Line Interface (CLI)

Run the CLI utility to parse the sample image.

**Pretty table output (human-readable):**
```bash
python src/cli.py --image images/sample_tyre.png --output pretty
```

**Raw JSON output (machine-readable):**
```bash
python src/cli.py --image images/sample_tyre.png --output json
```

**Custom parameters support:**
```bash
# Save output to a JSON file
python src/cli.py --image images/sample_tyre.png --save output.json

# Use a different model size (e.g., Qwen2.5-VL-3B-Instruct)
python src/cli.py --image images/sample_tyre.png --model Qwen/Qwen2.5-VL-3B-Instruct
```

---

### Option B: Interactive Web Interface

Launch the Streamlit web dashboard:
```bash
streamlit run src/app.py
```

This will spin up a local development server and open `http://localhost:8501` in your browser.
- Select your target model size from the sidebar.
- Drag and drop your tyre image or select `images/sample_tyre.png`.
- Click **Run Extraction** to run the OCR and view the premium structured dashboard!
- Download your extracted specs as JSON or CSV directly from the button controls.

---

## 📂 Repository Layout
```
Tyre/
├── README.md               # Setup and usage documentation
├── requirements.txt        # Third-party package dependencies
├── images/
│   └── sample_tyre.png    # High-quality mockup image for verification
└── src/
    ├── __init__.py         # Package module initializer
    ├── ocr_engine.py       # Core Qwen-VL model loader & OCR parsing engine
    ├── cli.py              # CLI wrapper execution file
    └── app.py              # Streamlit Web GUI dashboard
```

---

## 🖥️ System Performance Tweaks

- **Mac users (Apple Silicon M1/M2/M3/M4)**: The engine automatically utilizes the **MPS (Metal Performance Shaders)** backend to run the model on your unified graphics cores for GPU-accelerated processing.
- **NVIDIA GPU Users**: The script automatically maps layers using `device_map="auto"` via standard `accelerate` for CUDA acceleration.
- **Low Memory / CPU**: If you run on CPU, calculations might take 15–30 seconds. To improve performance, close other RAM-heavy tasks while running inference.
