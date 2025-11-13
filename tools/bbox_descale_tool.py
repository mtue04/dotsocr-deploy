#!/usr/bin/env python3
"""
Tool để chuyển đổi bbox từ format đã scale về kích thước gốc (original)
Sử dụng khi bạn có JSON với bbox đã bị scale x2 và muốn về kích thước gốc
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any, Union

def descale_bbox(bbox: List[Union[int, float]], scale_factor: float = 2.0) -> List[int]:
    """
    Descale một bbox về kích thước gốc
    
    Args:
        bbox: [x1, y1, x2, y2] đã bị scale
        scale_factor: Hệ số scale (default: 2.0)
    
    Returns:
        [x1, y1, x2, y2] kích thước gốc
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox phải có 4 tọa độ, nhận được: {len(bbox)}")
    
    return [
        int(bbox[0] / scale_factor),
        int(bbox[1] / scale_factor), 
        int(bbox[2] / scale_factor),
        int(bbox[3] / scale_factor)
    ]

def process_json_data(data: Union[List, Dict], scale_factor: float = 2.0) -> Union[List, Dict]:
    """
    Xử lý JSON data để descale tất cả bbox
    
    Args:
        data: JSON data (list hoặc dict)
        scale_factor: Hệ số scale để chia
        
    Returns:
        JSON data với bbox đã được descale
    """
    if isinstance(data, list):
        # Trường hợp array của các layout items
        result = []
        for item in data:
            if isinstance(item, dict) and 'bbox' in item:
                new_item = item.copy()
                try:
                    new_item['bbox'] = descale_bbox(item['bbox'], scale_factor)
                except Exception as e:
                    print(f"Warning: Không thể descale bbox {item['bbox']}: {e}")
                    continue
                result.append(new_item)
            else:
                result.append(item)
        return result
        
    elif isinstance(data, dict):
        # Trường hợp có wrapper object
        result = data.copy()
        
        # Tìm các key có thể chứa layout data
        layout_keys = ['layout', 'results', 'annotations', 'items', 'data']
        
        for key in layout_keys:
            if key in data and isinstance(data[key], list):
                result[key] = process_json_data(data[key], scale_factor)
                break
        else:
            # Nếu không tìm thấy layout key, check xem có bbox trực tiếp không
            if 'bbox' in data:
                try:
                    result['bbox'] = descale_bbox(data['bbox'], scale_factor)
                except Exception as e:
                    print(f"Warning: Không thể descale bbox {data['bbox']}: {e}")
        
        return result
    
    else:
        return data

def descale_json_file(
    input_path: str, 
    output_path: str = None,
    scale_factor: float = 2.0,
    backup: bool = True
) -> str:
    """
    Descale bbox trong file JSON
    
    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output (None = overwrite input)
        scale_factor: Hệ số scale để chia
        backup: Có tạo backup không
        
    Returns:
        Đường dẫn file output
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File không tồn tại: {input_path}")
    
    # Đọc file JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"File JSON không hợp lệ: {e}")
    
    # Process data
    processed_data = process_json_data(data, scale_factor)
    
    # Xác định output path
    if output_path is None:
        output_path = input_path
        
        # Tạo backup nếu cần
        if backup:
            backup_path = input_path + '.backup'
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"📁 Backup created: {backup_path}")
    
    # Ghi file output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return output_path

def analyze_json_file(input_path: str) -> Dict[str, Any]:
    """
    Phân tích file JSON để hiểu cấu trúc và số lượng bbox
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    
    analysis = {
        "file_type": type(data).__name__,
        "total_items": 0,
        "bbox_count": 0,
        "categories": set(),
        "bbox_ranges": {"min_x": float('inf'), "min_y": float('inf'), 
                       "max_x": 0, "max_y": 0},
        "sample_bboxes": []
    }
    
    def analyze_item(item):
        if isinstance(item, dict) and 'bbox' in item:
            analysis["bbox_count"] += 1
            bbox = item['bbox']
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                analysis["bbox_ranges"]["min_x"] = min(analysis["bbox_ranges"]["min_x"], x1)
                analysis["bbox_ranges"]["min_y"] = min(analysis["bbox_ranges"]["min_y"], y1) 
                analysis["bbox_ranges"]["max_x"] = max(analysis["bbox_ranges"]["max_x"], x2)
                analysis["bbox_ranges"]["max_y"] = max(analysis["bbox_ranges"]["max_y"], y2)
                
                if len(analysis["sample_bboxes"]) < 5:
                    analysis["sample_bboxes"].append({
                        "bbox": bbox,
                        "category": item.get("category", "Unknown")
                    })
            
            if 'category' in item:
                analysis["categories"].add(item['category'])
    
    if isinstance(data, list):
        analysis["total_items"] = len(data)
        for item in data:
            analyze_item(item)
    elif isinstance(data, dict):
        # Tìm layout data
        layout_keys = ['layout', 'results', 'annotations', 'items', 'data']
        for key in layout_keys:
            if key in data and isinstance(data[key], list):
                analysis["total_items"] = len(data[key])
                for item in data[key]:
                    analyze_item(item)
                break
        else:
            analyze_item(data)
            analysis["total_items"] = 1
    
    analysis["categories"] = list(analysis["categories"])
    return analysis

def main():
    parser = argparse.ArgumentParser(
        description="Tool để descale bbox trong JSON từ kích thước đã scale về gốc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Descale với scale factor 2.0 (mặc định)
  python bbox_descale_tool.py input.json
  
  # Descale với scale factor tùy chỉnh
  python bbox_descale_tool.py input.json --scale 1.5
  
  # Descale và lưu file mới
  python bbox_descale_tool.py input.json --output output.json
  
  # Phân tích file trước khi descale
  python bbox_descale_tool.py input.json --analyze
  
  # Descale không tạo backup
  python bbox_descale_tool.py input.json --no-backup
        """
    )
    
    parser.add_argument('input_file', help='Đường dẫn file JSON input')
    parser.add_argument('--output', '-o', help='Đường dẫn file JSON output (default: overwrite input)')
    parser.add_argument('--scale', '-s', type=float, default=2.0, help='Scale factor để chia (default: 2.0)')
    parser.add_argument('--no-backup', action='store_true', help='Không tạo backup file')
    parser.add_argument('--analyze', '-a', action='store_true', help='Chỉ phân tích file, không descale')
    
    args = parser.parse_args()
    
    try:
        if args.analyze:
            print(f"🔍 Analyzing file: {args.input_file}")
            analysis = analyze_json_file(args.input_file)
            
            if "error" in analysis:
                print(f"❌ Error: {analysis['error']}")
                return 1
            
            print(f"\n📊 Analysis Results:")
            print(f"  File type: {analysis['file_type']}")
            print(f"  Total items: {analysis['total_items']}")
            print(f"  Bbox count: {analysis['bbox_count']}")
            print(f"  Categories: {', '.join(analysis['categories'])}")
            
            if analysis['bbox_count'] > 0:
                ranges = analysis['bbox_ranges']
                print(f"  Bbox ranges:")
                print(f"    X: {ranges['min_x']} → {ranges['max_x']}")
                print(f"    Y: {ranges['min_y']} → {ranges['max_y']}")
                
                print(f"\n📝 Sample bboxes:")
                for sample in analysis['sample_bboxes']:
                    print(f"    {sample['category']}: {sample['bbox']}")
                    
                print(f"\n💡 Suggested descale preview (÷{args.scale}):")
                for sample in analysis['sample_bboxes'][:2]:
                    original = sample['bbox']
                    descaled = descale_bbox(original, args.scale)
                    print(f"    {original} → {descaled}")
            
            return 0
        
        print(f"🔧 Descaling bbox in: {args.input_file}")
        print(f"📏 Scale factor: ÷{args.scale}")
        
        output_path = descale_json_file(
            input_path=args.input_file,
            output_path=args.output,
            scale_factor=args.scale,
            backup=not args.no_backup
        )
        
        print(f"✅ Descaled successfully!")
        print(f"📄 Output file: {output_path}")
        
        # Quick verification
        analysis_before = analyze_json_file(args.input_file)
        analysis_after = analyze_json_file(output_path)
        
        if analysis_before.get('bbox_count') == analysis_after.get('bbox_count'):
            print(f"✅ Verification: {analysis_after['bbox_count']} bboxes processed")
        else:
            print(f"⚠️  Warning: bbox count mismatch")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())