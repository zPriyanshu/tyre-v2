#!/usr/bin/env python
"""
Batch processing pipeline for Tyre OCR.

Processes all images in a directory, runs quality check → model inference → validation,
and outputs results as CSV and/or JSON for review.

Usage:
    python src/batch.py --input-dir ./images/test_batch/ --mode stencil --mlx --output-csv results.csv
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def discover_images(input_dir):
    """Find all supported image files in a directory (non-recursive)."""
    images = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            images.append(os.path.join(input_dir, fname))
    return images


def extract_primary_value(result, mode):
    """Pull the main extracted value from a result dict based on mode."""
    key_map = {
        "stencil": "stencilNumber",
        "tyre_size": "tyreSize",
        "tube_size": "tubeSize",
        "flap_size": "flapSize",
    }
    key = key_map.get(mode, "")
    return result.get(key) or ""


def build_csv_row(filename, result, mode, elapsed):
    """Build a flat dict suitable for CSV output."""
    quality = result.get("_quality", {})
    return {
        "filename": os.path.basename(filename),
        "filepath": filename,
        "mode": mode,
        "extracted_value": extract_primary_value(result, mode),
        "confidence": result.get("confidence", ""),
        "validation": result.get("_validation", ""),
        "validation_reason": result.get("_validation_reason", ""),
        "blur_score": quality.get("blur_score", ""),
        "contrast_score": quality.get("contrast_score", ""),
        "resolution": "x".join(str(d) for d in quality.get("resolution", [])),
        "quality_warnings": "; ".join(quality.get("warnings", [])),
        "debug_message": result.get("debugMessage", ""),
        "elapsed_seconds": round(elapsed, 2),
        # Extra fields for tyre_size mode
        "pattern_name": result.get("patternName", ""),
        # Human review fields (blank, filled in during review)
        "human_verdict": "",
        "human_notes": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process tyre images for OCR extraction."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing tyre images to process.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stencil",
        choices=["stencil", "tyre_size", "tube_size", "flap_size"],
        help="Extraction mode (default: stencil).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use Apple Silicon MLX acceleration.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save CSV results (e.g. results.csv).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save full JSON results (e.g. results.json).",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Discover images
    images = discover_images(args.input_dir)
    if not images:
        print(f"No supported images found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    total = len(images)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  BATCH TYRE OCR PROCESSING", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Images found:  {total}", file=sys.stderr)
    print(f"  Mode:          {args.mode}", file=sys.stderr)
    print(f"  Model:         {args.model}", file=sys.stderr)
    print(f"  Framework:     {'MLX' if args.mlx else 'PyTorch'}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Load engine once
    print("Loading model...", file=sys.stderr)
    engine = TyreOCREngine(model_id=args.model, use_mlx=args.mlx)
    engine.load_model()
    print("Model loaded.\n", file=sys.stderr)

    # Process each image
    all_results = []
    csv_rows = []
    skipped = 0
    passed = 0
    failed = 0

    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        progress = f"[{idx}/{total}]"

        t0 = time.time()
        try:
            result = engine.extract_tyre_info(img_path, mode=args.mode)
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "_mode": args.mode,
                "_validation": "ERROR",
                "_validation_reason": str(e),
                "confidence": "low",
                "debugMessage": f"Exception: {e}",
            }

        # Track stats
        validation = result.get("_validation", "UNKNOWN")
        if validation == "SKIPPED":
            skipped += 1
            status_icon = "⏭️ "
        elif validation == "FAIL" or validation == "ERROR":
            failed += 1
            status_icon = "❌"
        else:
            passed += 1
            status_icon = "✅"

        value = extract_primary_value(result, args.mode)
        confidence = result.get("confidence", "?")
        print(
            f"  {progress} {status_icon} {fname:<35} "
            f"val={value or 'null':<20} conf={confidence:<7} "
            f"valid={validation:<8} ({elapsed:.1f}s)",
            file=sys.stderr,
        )

        # Collect results
        result["_source_file"] = img_path
        all_results.append(result)
        csv_rows.append(build_csv_row(img_path, result, args.mode, elapsed))

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  BATCH COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total:    {total}", file=sys.stderr)
    print(f"  ✅ Passed: {passed}", file=sys.stderr)
    print(f"  ❌ Failed: {failed}", file=sys.stderr)
    print(f"  ⏭️  Skipped: {skipped} (low quality images)", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Save CSV
    if args.output_csv:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to: {args.output_csv}", file=sys.stderr)

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"JSON saved to: {args.output_json}", file=sys.stderr)

    # If neither output specified, print JSON to stdout
    if not args.output_csv and not args.output_json:
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
