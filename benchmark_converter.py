"""
Benchmark Format Converter for Dots.OCR Output
Converts layout detection results to standardized benchmark format
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from PIL import Image


def bbox_to_poly(bbox: List[float]) -> List[float]:
    """
    Convert bbox [x1, y1, x2, y2] to polygon format [x1, y1, x2, y1, x2, y2, x1, y2]
    """
    x1, y1, x2, y2 = bbox
    return [
        float(x1), float(y1),  # top-left
        float(x2), float(y1),  # top-right
        float(x2), float(y2),  # bottom-right
        float(x1), float(y2)   # bottom-left
    ]


def detect_text_language(text: str) -> str:
    """
    Detect language: text_vietnamese, text_english, or text_vi_en_mixed
    """
    if not text:
        return "text_english"
    
    # Count Vietnamese characters (with diacritics)
    vietnamese_chars = len(re.findall(
        r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]', 
        text.lower()
    ))
    
    # Count English letters
    english_chars = len(re.findall(r'[a-z]', text.lower()))
    
    total_alpha = vietnamese_chars + english_chars
    
    if total_alpha == 0:
        return "text_english"
    
    vi_ratio = vietnamese_chars / total_alpha
    
    if vi_ratio > 0.3:
        if vi_ratio < 0.7:
            return "text_vi_en_mixed"
        else:
            return "text_vietnamese"
    else:
        return "text_english"


def map_category_type(category: str) -> str:
    """
    Map dots.ocr categories to benchmark categories
    """
    category_mapping = {
        'Title': 'title',
        'Section-header': 'section_header',
        'Text': 'text',
        'List-item': 'list_item',
        'Table': 'table',
        'Formula': 'formula',
        'Picture': 'figure',
        'Caption': 'caption',
        'Footnote': 'footnote',
        'Page-header': 'page_header',
        'Page-footer': 'page_footer'
    }
    return category_mapping.get(category, category.lower())


def split_text_into_lines(text: str, bbox: List[float]) -> List[Dict]:
    """
    Split text into lines with approximate bounding boxes
    """
    if not text:
        return []
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return []
    
    x1, y1, x2, y2 = bbox
    height_per_line = (y2 - y1) / len(lines)
    
    line_with_spans = []
    for i, line_text in enumerate(lines):
        line_y1 = y1 + i * height_per_line
        line_y2 = line_y1 + height_per_line
        
        line_with_spans.append({
            "category_type": "text_line",
            "poly": bbox_to_poly([x1, line_y1, x2, line_y2]),
            "text": line_text
        })
    
    return line_with_spans


def convert_dots_to_benchmark(
    dots_output: List[Dict],
    image: Image.Image,
    page_no: int = 1,
    image_path: str = "",
    data_source: str = "dots.ocr",
    layout_type: str = "article",
    special_issues: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert dots.ocr output to benchmark format
    
    Args:
        dots_output: List of layout elements from dots.ocr
        image: PIL Image object
        page_no: Page number (1-based)
        image_path: Path to the image file
        data_source: Source of the document
        layout_type: Type of layout (article, form, presentation, etc.)
        special_issues: List of special issues (noisy, blur, watermark, etc.)
    
    Returns:
        Dictionary in benchmark format
    """
    
    # Detect overall language
    all_text = " ".join([item.get('text', '') for item in dots_output if item.get('text')])
    page_language = "vietnamese" if detect_text_language(all_text) in ["text_vietnamese", "text_vi_en_mixed"] else "english"
    
    # Convert layout detections
    layout_dets = []
    
    for order, item in enumerate(dots_output):
        category = item.get('category', 'Text')
        bbox = item.get('bbox', [])
        text = item.get('text', '')
        
        if not bbox or len(bbox) != 4:
            continue
        
        # Determine if should ignore (page headers/footers)
        ignore = category in ['Page-header', 'Page-footer']
        
        # Create layout detection entry
        layout_det = {
            "category_type": map_category_type(category),
            "poly": bbox_to_poly(bbox),
            "ignore": ignore,
            "order": order,
            "text": text,
            "line_with_spans": [],
            "attributes": {
                "text_language": detect_text_language(text) if text else "text_english",
                "text_background": "white",
                "text_rotate": "normal"
            }
        }
        
        # Add line_with_spans for text-based categories
        if text and category not in ['Picture', 'Table', 'Formula']:
            layout_det["line_with_spans"] = split_text_into_lines(text, bbox)
        
        layout_dets.append(layout_det)
    
    # Create benchmark format output
    benchmark_format = {
        "layout_dets": layout_dets,
        "extra": {
            "relation": []
        },
        "page_info": {
            "page_attribute": {
                "data_source": data_source,
                "language": page_language,
                "layout": layout_type,
                "special_issue": special_issues or []
            },
            "page_no": page_no,
            "height": image.height,
            "width": image.width,
            "image_path": image_path
        }
    }
    
    return benchmark_format


def convert_single_result(
    dots_output: List[Dict],
    image: Image.Image,
    output_path: str,
    page_no: int = 1,
    image_path: str = "",
    **kwargs
) -> str:
    """
    Convert single dots.ocr result to benchmark format and save
    
    Args:
        dots_output: Layout detection results from dots.ocr
        image: PIL Image
        output_path: Path to save the benchmark JSON
        page_no: Page number
        image_path: Path to the image file
        **kwargs: Additional parameters for convert_dots_to_benchmark
    
    Returns:
        Path to saved file
    """
    benchmark_data = convert_dots_to_benchmark(
        dots_output=dots_output,
        image=image,
        page_no=page_no,
        image_path=image_path,
        **kwargs
    )
    
    # Save as list (multi-page format)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([benchmark_data], f, indent=2, ensure_ascii=False)
    
    return output_path


def batch_convert_json_files(
    input_folder: str,
    output_folder: str,
    image_folder: Optional[str] = None
):
    """
    Batch convert dots.ocr JSON files to benchmark format
    
    Args:
        input_folder: Folder containing dots.ocr JSON outputs
        output_folder: Folder to save benchmark format JSONs
        image_folder: Folder containing corresponding images (optional)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        
        try:
            # Load dots.ocr output
            with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
                dots_data = json.load(f)
            
            # Try to find corresponding image
            image_path = ""
            image = None
            if image_folder:
                base_name = os.path.splitext(json_file)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    img_path = os.path.join(image_folder, base_name + ext)
                    if os.path.exists(img_path):
                        image = Image.open(img_path)
                        image_path = img_path
                        break
            
            # If no image found, create dummy
            if image is None:
                image = Image.new('RGB', (1000, 1000))
            
            # Convert to benchmark format
            benchmark_data = convert_dots_to_benchmark(
                dots_output=dots_data,
                image=image,
                page_no=1,
                image_path=image_path,
                data_source="dots.ocr"
            )
            
            # Save
            output_file = os.path.join(output_folder, json_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([benchmark_data], f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert dots.ocr output to benchmark format")
    parser.add_argument("--input", required=True, help="Input folder with dots.ocr JSON files")
    parser.add_argument("--output", required=True, help="Output folder for benchmark JSONs")
    parser.add_argument("--images", help="Folder containing corresponding images (optional)")
    
    args = parser.parse_args()
    
    batch_convert_json_files(args.input, args.output, args.images)
    print("\n✅ Conversion completed!")
