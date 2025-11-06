# 📁 DOTSOCR Project Structure

```
DOTSOCR/
│
├── 📄 Core Application Files
│   ├── app.py                      # Main Gradio web application
│   ├── benchmark_converter.py      # Benchmark format conversion utilities
│   └── test_example.py            # Example test script
│
├── 📋 Configuration Files
│   ├── requirements.txt            # Python dependencies
│   └── .gitignore                 # Git ignore patterns
│
├── 📖 Documentation
│   ├── README.md                   # Complete documentation (English)
│   ├── QUICKSTART.md              # Quick start guide (Vietnamese)
│   ├── CHANGELOG.md               # Version history
│   └── PROJECT_STRUCTURE.md       # This file
│
├── ⚖️ Legal
│   └── LICENSE                     # MIT License
│
├── 🚀 Windows Scripts
│   ├── setup.bat                  # Automated setup script
│   └── run.bat                    # Quick launch script
│
├── 🤖 Model Cache (auto-created)
│   └── models/
│       └── dots-ocr-local/        # Downloaded model weights (~8GB)
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer files
│           └── ...
│
├── 💾 Output Directory (auto-created)
│   └── benchmark_dataset/         # Exported benchmark files
│       ├── doc_page001_timestamp.json
│       ├── doc_page001_timestamp.png
│       └── ...
│
└── 🐍 Virtual Environment (created by setup)
    └── venv/                      # Python virtual environment
        ├── Scripts/               # Executables (Windows)
        ├── Lib/                   # Installed packages
        └── ...
```

## 📄 File Descriptions

### Core Files

**app.py** (Main Application)
- Entry point for the application
- Gradio web interface setup
- Document processing logic
- Page navigation handlers
- Export functionality
- ~900 lines, well-commented

**benchmark_converter.py** (Converter Module)
- Converts dots.ocr output to benchmark format
- Handles bbox to polygon conversion
- Language detection (Vietnamese/English/Mixed)
- Category mapping
- Line splitting for text spans
- Batch conversion support
- ~280 lines

**test_example.py** (Test Script)
- Tests benchmark conversion
- Reads file 'a' from parent directory
- Demonstrates usage of converter
- Prints detailed statistics
- ~90 lines

### Configuration

**requirements.txt**
- All Python dependencies
- PyTorch and transformers
- Gradio for web UI
- PDF and image processing libraries
- Version-pinned for stability

**.gitignore**
- Excludes model files (too large)
- Ignores virtual environment
- Skips output files
- Standard Python patterns

### Documentation

**README.md** (Primary Documentation)
- Complete feature overview
- Installation instructions
- Usage guide with examples
- Benchmark format specification
- Troubleshooting guide
- Technical details
- ~500 lines in English

**QUICKSTART.md** (Vietnamese Guide)
- Quick installation steps for Windows
- Basic usage examples
- Common error solutions
- Tips and tricks
- Checklist format
- ~200 lines in Vietnamese

**CHANGELOG.md** (Version History)
- Version 1.0.0 details
- Feature list
- Future roadmap
- Known issues
- ~100 lines

**PROJECT_STRUCTURE.md** (This File)
- Directory structure
- File descriptions
- Component overview
- Data flow diagram

### Scripts

**setup.bat** (Windows Setup)
- Checks Python installation
- Creates virtual environment
- Installs dependencies
- Handles GPU/CPU detection
- Error checking
- ~80 lines

**run.bat** (Windows Launcher)
- Activates virtual environment
- Launches application
- Error handling
- ~40 lines

## 🔄 Data Flow

```
User Input (PDF/Image)
        ↓
    app.py
        ↓
  [Load File]
        ↓
  [Process with Model]
        ↓
  [Layout Detection]
        ↓
    ┌───┴───┐
    ↓       ↓
[Display] [Export]
    ↓       ↓
Gradio UI  benchmark_converter.py
            ↓
        Benchmark JSON
```

## 🧩 Component Interactions

```
┌─────────────────────────────────────────┐
│           Gradio Web UI                 │
│  (User Interface Layer)                 │
└────────────┬────────────────────────────┘
             ↓
┌────────────┴────────────────────────────┐
│         app.py                          │
│  - File handling                        │
│  - Model inference                      │
│  - Result processing                    │
└────────────┬────────────────────────────┘
             ↓
     ┌───────┴────────┐
     ↓                ↓
┌─────────┐    ┌──────────────────┐
│ Model   │    │ benchmark_      │
│ (dots.  │    │ converter.py     │
│  ocr)   │    │ - Format conv.   │
└─────────┘    └──────────────────┘
     ↓                ↓
[Layout JSON]   [Benchmark JSON]
```

## 📊 File Size Reference

| File/Directory | Size | Notes |
|----------------|------|-------|
| app.py | ~60 KB | Main application |
| benchmark_converter.py | ~10 KB | Converter |
| requirements.txt | ~1 KB | Dependencies list |
| README.md | ~40 KB | Documentation |
| models/ | ~8 GB | Model weights (downloaded) |
| venv/ | ~2 GB | Virtual environment |
| benchmark_dataset/ | Varies | Output files |

## 🔑 Key Features by File

### app.py
- ✅ PDF/Image loading
- ✅ Multi-page navigation
- ✅ Layout detection
- ✅ Markdown conversion
- ✅ Visual annotation
- ✅ Export functionality
- ✅ RTL text support
- ✅ Coordinate scaling

### benchmark_converter.py
- ✅ Format conversion
- ✅ Language detection
- ✅ Category mapping
- ✅ Line splitting
- ✅ Batch processing
- ✅ Polygon generation

### setup.bat
- ✅ Environment setup
- ✅ Dependency installation
- ✅ GPU detection
- ✅ Error handling

### run.bat
- ✅ Quick launch
- ✅ Environment activation
- ✅ Validation checks

## 🎯 Quick Navigation

- **Setup**: Run `setup.bat`
- **Launch**: Run `run.bat` or `python app.py`
- **Test**: Run `python test_example.py`
- **Documentation**: See `README.md`
- **Quick Guide**: See `QUICKSTART.md`

## 📝 Notes

- All Python files are UTF-8 encoded
- Scripts are Windows-optimized (.bat files)
- Documentation is bilingual (English/Vietnamese)
- Code is well-commented for maintainability
- Follows PEP 8 style guidelines

## 🔄 Update History

- **v1.0.0** (2024-01-06): Initial structure created

---

For questions or issues, refer to README.md or QUICKSTART.md
