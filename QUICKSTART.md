# HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG NHANH

## 🚀 Cài đặt nhanh (Windows)

### Bước 1: Cài đặt Python

Tải và cài Python 3.9+ từ: https://www.python.org/downloads/

**✅ Quan trọng**: Tick "Add Python to PATH" khi cài!

### Bước 2: Mở PowerShell trong thư mục DOTSOCR

```powershell
cd d:\S10\CSLT\DOTSOCR
```

### Bước 3: Tạo môi trường ảo

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Lưu ý**: Nếu gặp lỗi "script execution is disabled", chạy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Bước 4: Cài đặt thư viện

```powershell
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Lưu ý**: 
- Nếu không có GPU, dùng: `pip install torch torchvision torchaudio`
- Cài flash-attn có thể mất 10-30 phút

### Bước 5: Chạy ứng dụng

```powershell
python app.py
```

**Lần đầu chạy**: Sẽ tải model (~8GB), mất 10-30 phút tùy tốc độ mạng

## 📝 Sử dụng cơ bản

### 1. Truy cập Web Interface

Sau khi chạy `python app.py`, mở trình duyệt:
```
http://localhost:7860
```

### 2. Upload và xử lý file

1. Click "Upload Image or PDF"
2. Chọn file (PDF hoặc ảnh)
3. Click "🚀 Process Document"
4. Đợi xử lý (30s-2 phút mỗi trang)

### 3. Xem kết quả

- **Tab 🖼️**: Xem ảnh với khung bbox
- **Tab 📝**: Đọc nội dung markdown
- **Tab 📋**: Xem JSON chi tiết

### 4. Export benchmark format

1. Mở accordion "📦 Export Benchmark Format"
2. Click "💾 Export Current Page" hoặc "💾 Export All Pages"
3. File lưu trong thư mục `benchmark_dataset/`

## 📁 Cấu trúc thư mục sau khi chạy

```
DOTSOCR/
├── app.py                    # File chính
├── benchmark_converter.py    # Converter
├── requirements.txt          # Dependencies
├── README.md                 # Tài liệu đầy đủ
├── venv/                     # Môi trường ảo (tự tạo)
├── models/                   # Model weights (tự tải)
│   └── dots-ocr-local/
└── benchmark_dataset/        # Output files (tự tạo)
    ├── doc_page001_*.json
    ├── doc_page001_*.png
    └── ...
```

## 🔧 Xử lý lỗi thường gặp

### Lỗi: "No module named 'xxx'"
```powershell
pip install xxx
```

### Lỗi: "CUDA out of memory"
- Giảm Max Pixels trong Advanced Settings
- Xử lý từng trang thay vì cả PDF
- Hoặc dùng CPU (chậm hơn)

### Lỗi: "flash_attn not found"
Sửa trong file `app.py`, dòng ~295:
```python
attn_implementation="eager",  # Thay vì "flash_attention_2"
```

### Model tải quá chậm
Dùng mirror:
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python app.py
```

## 🎯 Test nhanh

Chạy script test:
```powershell
python test_example.py
```

Script sẽ convert file 'a' từ thư mục cha sang benchmark format.

## 📊 Format benchmark - Giải thích ngắn

Mỗi file JSON có cấu trúc:

```json
[
  {
    "layout_dets": [        // Danh sách các phần tử layout
      {
        "category_type": "title",           // Loại: title, text, table, ...
        "poly": [x1,y1, x2,y1, x2,y2, x1,y2],  // Tọa độ polygon 8 điểm
        "ignore": false,                    // Có bỏ qua khi đánh giá không
        "order": 0,                         // Thứ tự đọc (từ 0)
        "text": "Nội dung văn bản",        // Text trích xuất được
        "line_with_spans": [...],          // Chi tiết từng dòng
        "attributes": {
          "text_language": "text_vietnamese",  // Ngôn ngữ
          "text_background": "white",          // Màu nền
          "text_rotate": "normal"              // Xoay
        }
      }
    ],
    "extra": {
      "relation": []        // Quan hệ giữa các phần tử
    },
    "page_info": {          // Thông tin trang
      "page_no": 1,
      "height": 3226,
      "width": 2596,
      "image_path": "path/to/image.png",
      "page_attribute": {
        "data_source": "dots.ocr",
        "language": "vietnamese",
        "layout": "article",
        "special_issue": []
      }
    }
  }
]
```

## 💡 Tips

1. **Chất lượng ảnh tốt**: Dùng scan 150-300 DPI
2. **PDF ưu tiên**: Tốt hơn ảnh cho tài liệu nhiều trang
3. **Kiên nhẫn**: Lần đầu tải model mất thời gian
4. **GPU**: Nhanh hơn CPU rất nhiều (15s vs 5 phút/trang)
5. **Export từng trang**: Dễ quản lý hơn export toàn bộ

## 📞 Hỗ trợ

- Đọc kỹ README.md để biết chi tiết
- Check GitHub: https://github.com/rednote-hilab/dots.ocr
- HuggingFace: https://huggingface.co/rednote-hilab/dots.ocr

## ✅ Checklist cài đặt

- [ ] Python 3.9+ đã cài
- [ ] Python trong PATH (chạy `python --version` được)
- [ ] Virtual environment đã tạo và activate
- [ ] Các thư viện đã cài (`pip list` để check)
- [ ] GPU driver cập nhật (nếu dùng GPU)
- [ ] Model đã tải xong (thư mục `models/` có nội dung)
- [ ] App chạy thành công và mở được web interface

---

**Thời gian cài đặt ước tính**: 30-60 phút (tùy tốc độ mạng)

**Yêu cầu ổ cứng**: ~15GB (10GB model + 5GB dependencies)

**Yêu cầu RAM**: Tối thiểu 16GB (32GB recommended)

**GPU khuyến nghị**: RTX 3060 trở lên với 12GB+ VRAM
