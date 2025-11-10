# 🔍 Dots.OCR - Document Text Extraction System

A comprehensive web application for extracting text and layout information from PDF documents and images using the state-of-the-art **Dots.OCR** vision language model. Features include layout detection, markdown export, and benchmark dataset generation.

## ✨ Features

- 📄 **Multi-format Support**: Process PDF files (multi-page) and images (JPG, PNG, BMP, TIFF)
- 🎯 **Layout Detection**: Automatic detection of 11 layout categories (Title, Section-header, Text, List-item, Table, Formula, Picture, Caption, Footnote, Page-header, Page-footer)
- 📝 **Markdown Export**: Convert detected content to clean markdown format
- 🖼️ **Visual Annotations**: View detected layouts with color-coded bounding boxes
- 🌐 **Multi-language**: Supports Vietnamese, English, and mixed content with automatic language detection
- 📊 **Benchmark Export**: Export results to standardized benchmark format for dataset creation and evaluation
- 🔄 **Batch Processing**: Process entire PDF documents page by page
- 💾 **Easy Export**: Export individual pages or entire documents in benchmark format

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for best performance)
- At least 16GB RAM
- ~10GB disk space for model weights

### Installation

#### 🐧 Linux GPU Server (A4000/A6000) - Automated Setup

For fresh Linux GPU servers, use our one-click installer:

```bash
# One-liner installation (recommended)
curl -fsSL https://raw.githubusercontent.com/mtue04/dotsocr-deploy/main/install.sh | bash
```

Or manual setup:
```bash
# Clone repository
git clone https://github.com/mtue04/dotsocr-deploy.git
cd dotsocr-deploy

# Make scripts executable
chmod +x auto_setup.sh run.sh

# Run automated setup (installs Python, CUDA, dependencies, etc.)
./auto_setup.sh

# Start application
./run.sh
```

The `auto_setup.sh` script will automatically:
- Update system packages
- Install Python 3, Git, and essential tools
- Install NVIDIA drivers and CUDA toolkit (if needed)
- Clone the repository
- Create virtual environment
- Install PyTorch with GPU support
- Install all dependencies
- Optionally start the application

#### 🪟 Windows - Quick Setup

```batch
# Run setup script
setup.bat

# Start application
run.bat
```

#### 🔧 Manual Installation (All Platforms)

1. **Clone or download this repository**

```bash
cd DOTSOCR
```

2. **Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Note**: Installing `flash-attn` may take some time and requires a compatible CUDA setup. If you encounter issues:
- For CPU-only: Remove `flash-attn` from requirements.txt and change `attn_implementation="eager"` in app.py
- For incompatible CUDA: Try `pip install flash-attn --no-build-isolation`

4. **Run the application**

```bash
python app.py
```

The application will:
- Download the Dots.OCR model (~8GB) on first run
- Start a Gradio web interface
- Open automatically in your browser at `http://localhost:7860`

## 📖 Usage Guide

### Basic Workflow

1. **Upload Document**
   - Click "Upload Image or PDF" button
   - Select your file (PDF or image)
   - Preview appears automatically

2. **Process Document**
   - Adjust settings if needed (Advanced Settings accordion)
   - Click "🚀 Process Document" button
   - Wait for processing to complete (may take 30s-2min per page)

3. **View Results**
   - **🖼️ Processed Image**: View layout detection with bounding boxes
   - **📝 Extracted Content**: Read extracted text in markdown format
   - **📋 Layout JSON**: Inspect detailed layout structure

4. **Navigate Pages** (for PDFs)
   - Use "◀ Previous" and "Next ▶" buttons to switch pages
   - Results are cached after processing

### Export Benchmark Format

The benchmark format is standardized for dataset creation and model evaluation:

1. **Export Current Page**
   - Open "📦 Export Benchmark Format" accordion
   - Click "💾 Export Current Page"
   - Files saved to `benchmark_dataset/` folder

2. **Export All Pages**
   - Click "💾 Export All Pages"
   - All processed pages exported at once

**Output files:**
```
benchmark_dataset/
├── document_page001_20240106_143022.json
├── document_page001_20240106_143022.png
├── document_page002_20240106_143022.json
├── document_page002_20240106_143022.png
└── ...
```

### Advanced Settings

- **Max Tokens** (1000-32000): Maximum output length. Increase for complex documents
- **Min/Max Pixels**: Control image resolution. Default values work well for most cases
- Lower resolution = faster but less accurate
- Higher resolution = slower but more detailed

## 📊 Benchmark Format Structure

Each exported JSON contains:

```json
[
  {
    "layout_dets": [
      {
        "category_type": "title",
        "poly": [x1, y1, x2, y1, x2, y2, x1, y2],
        "ignore": false,
        "order": 0,
        "text": "Document title",
        "line_with_spans": [
          {
            "category_type": "text_line",
            "poly": [...],
            "text": "Line text"
          }
        ],
        "attributes": {
          "text_language": "text_vietnamese",
          "text_background": "white",
          "text_rotate": "normal"
        }
      }
    ],
    "extra": {
      "relation": []
    },
    "page_info": {
      "page_attribute": {
        "data_source": "dots.ocr",
        "language": "vietnamese",
        "layout": "article",
        "special_issue": []
      },
      "page_no": 1,
      "height": 3226,
      "width": 2596,
      "image_path": "path/to/image.png"
    }
  }
]
```

### Field Descriptions

**layout_dets**: Array of detected layout elements
- `category_type`: Element type (title, text, table, etc.)
- `poly`: 8-coordinate polygon [x1,y1, x2,y1, x2,y2, x1,y2]
- `ignore`: Whether to skip during evaluation
- `order`: Reading sequence (0-based)
- `text`: Extracted text content
- `line_with_spans`: Text split into lines
- `attributes`:
  - `text_language`: text_vietnamese / text_english / text_vi_en_mixed
  - `text_background`: Background color
  - `text_rotate`: Text rotation (normal/90/180/270)

**page_info**: Document metadata
- `page_no`: Page number (1-based)
- `height/width`: Image dimensions in pixels
- `image_path`: Path to corresponding image file
- `page_attribute`:
  - `data_source`: Origin of document
  - `language`: Primary language
  - `layout`: Document type (article/form/presentation)
  - `special_issue`: Quality issues (noisy/blur/watermark)

## 🏷️ Layout Categories

| Category | Description | Example |
|----------|-------------|---------|
| Title | Document main title | "Annual Report 2024" |
| Section-header | Section headings | "Introduction", "Methods" |
| Text | Body paragraphs | Regular text content |
| List-item | Bulleted/numbered items | "• Item 1" |
| Table | Tabular data | HTML formatted tables |
| Formula | Mathematical equations | LaTeX formatted |
| Picture | Images/figures | (no text) |
| Caption | Image/table captions | "Figure 1: Results" |
| Footnote | Footer notes | References |
| Page-header | Header text | Page titles |
| Page-footer | Footer text | Page numbers |

## 🔧 Troubleshooting

### Model Download Issues

**Problem**: Model download fails or is very slow

**Solutions**:
- Check internet connection
- Use HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- Pre-download model: `huggingface-cli download rednote-hilab/dots.ocr`

### GPU Memory Issues

**Problem**: CUDA out of memory error

**Solutions**:
- Reduce Max Pixels in Advanced Settings
- Process one page at a time instead of entire PDF
- Close other GPU-intensive applications
- Use CPU mode (slower): Set `device_map="cpu"` in app.py

### Flash Attention Errors

**Problem**: flash-attn installation or runtime errors

**Solutions**:
- Change to eager attention in app.py:
  ```python
  attn_implementation="eager",  # Instead of "flash_attention_2"
  ```
- Or install compatible version: `pip install flash-attn==2.5.0`

### Processing Too Slow

**Problem**: Each page takes >5 minutes

**Solutions**:
- Ensure GPU is being used: Check "Model loaded successfully!" message
- Reduce image resolution via Min/Max Pixels
- Check GPU utilization: `nvidia-smi`

### Export Fails

**Problem**: Benchmark export shows error

**Solutions**:
- Ensure document is processed first
- Check disk space in working directory
- Verify `benchmark_dataset/` folder permissions

## 📁 Project Structure

```
DOTSOCR/
├── app.py                      # Main Gradio application
├── benchmark_converter.py       # Format conversion utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Downloaded model weights (auto-created)
│   └── dots-ocr-local/
└── benchmark_dataset/          # Exported benchmark files (auto-created)
```

## 🔬 Technical Details

### Model Information

- **Model**: rednote-hilab/dots.ocr
- **Type**: Vision Language Model (VLM)
- **Base**: Qwen-VL architecture
- **Size**: ~8GB
- **Precision**: bfloat16
- **Context**: Up to 32K tokens

### Performance

| Hardware | Speed (per page) | Notes |
|----------|------------------|-------|
| RTX 4090 | ~15s | Recommended |
| RTX 3080 | ~30s | Good |
| RTX 2080 | ~45s | Adequate |
| CPU Only | ~5-10min | Not recommended |

### Coordinate System

- Origin: Top-left corner (0, 0)
- X-axis: Left to right
- Y-axis: Top to bottom
- Units: Pixels
- Format: [x1, y1, x2, y2] for bbox
- Format: [x1,y1, x2,y1, x2,y2, x1,y2] for polygon

### Scale Handling

The application automatically handles resolution scaling:
1. Original image loaded
2. Resized for model inference (if needed)
3. Bounding boxes scaled back to original size
4. Export uses original dimensions

## 🌟 Use Cases

### 1. Dataset Creation
Create benchmark datasets for training/evaluating document parsing models:
- Process existing PDFs
- Export standardized format
- Use for model training

### 2. Document Digitization
Convert physical documents to structured digital format:
- Scan documents to PDF/image
- Process with Dots.OCR
- Export to markdown for editing

### 3. Content Extraction
Extract specific content types:
- Tables from reports
- Formulas from papers
- Headers for indexing

### 4. Quality Assurance
Verify OCR quality:
- Visual bbox inspection
- Compare with ground truth
- Identify problem areas

## 📚 References

- **Dots.OCR Model**: [HuggingFace](https://huggingface.co/rednote-hilab/dots.ocr)
- **Release Blog**: [GitHub](https://github.com/rednote-hilab/dots.ocr/blob/master/assets/blog.md)
- **GitHub Repository**: [rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)
- **Qwen-VL**: [HuggingFace](https://huggingface.co/Qwen/Qwen-VL)

## 🤝 Contributing

Suggestions and improvements welcome! Areas for contribution:
- Additional export formats
- Batch processing optimizations
- UI/UX enhancements
- Documentation improvements

## 📝 License

This application uses the Dots.OCR model which has its own license. Please refer to the [model repository](https://huggingface.co/rednote-hilab/dots.ocr) for details.

## ⚠️ Known Limitations

1. **Table Recognition**: Complex tables may not parse perfectly
2. **Rotated Text**: Best results with upright text
3. **Handwriting**: Not designed for handwritten content
4. **Low Resolution**: Poor quality images may have reduced accuracy
5. **Large Documents**: Very large PDFs (>100 pages) may require batch processing

## 💡 Tips for Best Results

1. **Image Quality**: Use 150-300 DPI scans
2. **File Format**: PDF preferred over images for multi-page
3. **Language**: Works best with Vietnamese and English
4. **Layout**: Clean, structured documents work better than dense text
5. **Preprocessing**: Deskew and clean scans before processing

## 🆘 Support

For issues and questions:
1. Check this README thoroughly
2. Review [Dots.OCR documentation](https://github.com/rednote-hilab/dots.ocr)
3. Check [HuggingFace discussions](https://huggingface.co/rednote-hilab/dots.ocr/discussions)

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-06  
**Powered by**: Dots.OCR, Gradio, PyTorch, Transformers
