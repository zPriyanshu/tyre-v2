#!/usr/bin/env python
import argparse
import json
import sys
import os

# Adjust path to import ocr_engine if run directly or as package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine


def main():
    parser = argparse.ArgumentParser(
        description="Tyre OCR Specification Extractor using Qwen-VL models."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the tyre image file (local path)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID to use (default: Qwen/Qwen2-VL-2B-Instruct)."
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use Apple Silicon MLX framework optimization if available."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="json",
        choices=["json", "pretty"],
        help="Output format: 'json' (raw machine-readable) or 'pretty' (human-readable table)."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the JSON output (e.g. output.json)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stencil",
        choices=["stencil", "tyre_size", "tube_size", "flap_size"],
        help="Extraction mode (default: stencil). Only used for VLM models, not YOLO."
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Tyre OCR Extractor...", file=sys.stderr)
    print(f"Image: {args.image}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Mode:  {args.mode}", file=sys.stderr)
    print(f"Framework: {'MLX' if args.mlx else 'PyTorch/Transformers'}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    try:
        # Load engine and run extraction
        engine = TyreOCREngine(model_id=args.model, use_mlx=args.mlx)
        result = engine.extract_tyre_info(args.image, mode=args.mode)
        
        # Save to file if specified
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.save}", file=sys.stderr)

        # Output formatting
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            if result.get("is_detection"):
                # Pretty print object detection results
                print("\n" + "=" * 80)
                print("         YOLO OBJECT DETECTION RESULTS (WITH TrOCR)")
                print("=" * 80)
                print(f" Model:            {result.get('model_id')}")
                print(f" Annotated Image:  {result.get('annotated_image_path')}")
                print("-" * 80)
                
                detections = result.get("detections", [])
                if detections:
                    print(f" {'Class':<14} | {'Confidence':<10} | {'Bounding Box (xyxy)':<22} | {'Extracted Text (TrOCR)'}")
                    print("-" * 80)
                    for det in detections:
                        cls = det.get("class")
                        conf = f"{det.get('confidence'):.2f}"
                        box = [round(x, 1) for x in det.get("box", [])]
                        text = det.get("text", "")
                        print(f" {cls:<14} | {conf:<10} | {str(box):<22} | {text}")
                else:
                    print(" No objects detected.")
                print("=" * 80 + "\n")
            else:
                # Mode-aware pretty print
                mode = result.get("_mode", args.mode)
                debug_msg = result.get("debugMessage", "")
                
                print("\n" + "=" * 55)
                
                if mode == "stencil":
                    print("         STENCIL NUMBER EXTRACTION")
                    print("=" * 55)
                    print(f" Stencil Number:   {result.get('stencilNumber') or 'N/A'}")
                elif mode == "tyre_size":
                    print("         TYRE SIZE & PATTERN EXTRACTION")
                    print("=" * 55)
                    print(f" Tyre Size:        {result.get('tyreSize') or 'N/A'}")
                    print(f" Pattern Name:     {result.get('patternName') or 'N/A'}")
                elif mode == "tube_size":
                    print("         TUBE SIZE EXTRACTION")
                    print("=" * 55)
                    print(f" Tube Size:        {result.get('tubeSize') or 'N/A'}")
                elif mode == "flap_size":
                    print("         FLAP SIZE EXTRACTION")
                    print("=" * 55)
                    print(f" Flap Size:        {result.get('flapSize') or 'N/A'}")
                
                if debug_msg:
                    print("-" * 55)
                    print(f" Debug: {debug_msg}")
                
                print("=" * 55 + "\n")

    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
