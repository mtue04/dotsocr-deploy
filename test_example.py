"""
Example script to test benchmark conversion
Convert file 'a' from parent directory to benchmark format
"""

import json
import sys
import os
from PIL import Image

# Add parent directory to path to import benchmark_converter
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_converter import convert_dots_to_benchmark


def test_conversion():
    """Test conversion with sample data"""
    
    print("=" * 60)
    print("Testing Benchmark Format Conversion")
    print("=" * 60)
    
    # Check if file 'a' exists in parent directory
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(parent_dir, 'a')
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        print("Please ensure file 'a' exists in parent directory")
        return
    
    print(f"\n📁 Reading file: {input_file}")
    
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        dots_data = json.load(f)
    
    print(f"✅ Loaded {len(dots_data)} layout elements")
    
    # Determine image size from bbox
    max_x = max_y = 0
    for item in dots_data:
        if 'bbox' in item:
            bbox = item['bbox']
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    
    width = int(max_x) + 100
    height = int(max_y) + 100
    
    print(f"📐 Image size: {width}x{height} pixels")
    
    # Create dummy image
    image = Image.new('RGB', (width, height), 'white')
    
    # Convert to benchmark format
    print("\n🔄 Converting to benchmark format...")
    
    benchmark_data = convert_dots_to_benchmark(
        dots_output=dots_data,
        image=image,
        page_no=1,
        image_path="sample_page.png",
        data_source="thanhnien.vn",
        layout_type="article",
        special_issues=[]
    )
    
    # Save output
    output_file = "example_benchmark_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([benchmark_data], f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Conversion completed!")
    print(f"💾 Saved to: {output_file}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"Total elements: {len(benchmark_data['layout_dets'])}")
    print(f"Page size: {benchmark_data['page_info']['width']}x{benchmark_data['page_info']['height']}")
    print(f"Language: {benchmark_data['page_info']['page_attribute']['language']}")
    
    # Category breakdown
    print("\n📋 Category Breakdown:")
    categories = {}
    for det in benchmark_data['layout_dets']:
        cat = det['category_type']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat:20s}: {count:3d}")
    
    # Show first element
    print("\n" + "=" * 60)
    print("📄 FIRST ELEMENT PREVIEW")
    print("=" * 60)
    if benchmark_data['layout_dets']:
        first_det = benchmark_data['layout_dets'][0]
        print(f"Category: {first_det['category_type']}")
        print(f"Order: {first_det['order']}")
        print(f"Polygon: {first_det['poly'][:4]}... (8 coords total)")
        print(f"Language: {first_det['attributes']['text_language']}")
        print(f"Ignore: {first_det['ignore']}")
        
        text_preview = first_det['text']
        if len(text_preview) > 100:
            text_preview = text_preview[:100] + "..."
        print(f"Text: {text_preview}")
        
        if first_det['line_with_spans']:
            print(f"Lines: {len(first_det['line_with_spans'])} line(s)")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_conversion()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
