# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-01-06

### Added
- Initial release of Dots.OCR Document Processing System
- Web-based Gradio interface for document processing
- Support for PDF and image files (JPG, PNG, BMP, TIFF)
- Multi-page PDF processing with page navigation
- Layout detection with 11 categories
- Markdown export functionality
- Visual layout annotation with color-coded bounding boxes
- JSON export of layout structure
- Benchmark format export for dataset creation
- Automatic coordinate scaling for different image resolutions
- Multi-language support (Vietnamese, English, mixed)
- RTL text detection and support
- Batch export for all pages
- Individual page export option
- Advanced settings for token limits and resolution
- Comprehensive documentation (README, QUICKSTART)
- Example test script
- Error handling and user feedback

### Features
- **Layout Categories**: Title, Section-header, Text, List-item, Table, Formula, Picture, Caption, Footnote, Page-header, Page-footer
- **Export Formats**: Markdown, JSON, Benchmark JSON
- **Language Detection**: Automatic Vietnamese/English/Mixed detection
- **Resolution Handling**: Smart resize with scale factor preservation
- **Caching**: Results cached for quick page navigation

### Technical Details
- Model: rednote-hilab/dots.ocr
- Framework: Gradio 4.19+
- Deep Learning: PyTorch with Flash Attention 2
- PDF Processing: PyMuPDF (fitz)
- Image Processing: Pillow

### Documentation
- Comprehensive README with usage guide
- Quick start guide for Windows
- Troubleshooting section
- API documentation for benchmark format
- Example scripts and test cases

### Known Issues
- Flash Attention may require manual configuration on some systems
- Large PDFs (>100 pages) may need batch processing
- Complex tables may have parsing limitations

### Requirements
- Python 3.9+
- CUDA-capable GPU recommended
- 16GB+ RAM
- ~15GB disk space

---

## Future Roadmap

### [1.1.0] - Planned
- [ ] Add support for more image formats (WebP, HEIC)
- [ ] Implement batch processing queue
- [ ] Add progress bars for long operations
- [ ] Support for custom category definitions
- [ ] Export to additional formats (DOCX, HTML)

### [1.2.0] - Planned
- [ ] RESTful API endpoint
- [ ] Docker container support
- [ ] Multi-user authentication
- [ ] Database storage for processed documents
- [ ] Annotation correction interface

### [2.0.0] - Future
- [ ] Fine-tuning interface for custom models
- [ ] Advanced table structure recognition
- [ ] Handwriting support
- [ ] Multi-model ensemble
- [ ] Cloud deployment templates

---

## Version History

- **1.0.0** (2024-01-06): Initial release

---

## Contributing

We welcome contributions! Please see README.md for guidelines.

## License

MIT License - See LICENSE file for details
