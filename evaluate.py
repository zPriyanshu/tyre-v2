#!/usr/bin/env python
"""
Model accuracy evaluation script for Tyre OCR.

Loads a spreadsheet (CSV or Excel) containing ground truth stencil numbers and image URLs/paths,
samples a subset of data or uses a specific row range, runs OCR inference on each image,
and computes accuracy metrics.

Usage:
    python src/evaluate.py
    python src/evaluate.py --rows 100:150 --mlx
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.request
import pandas as pd

# Adjust path to import ocr_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine


def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_stencil(s):
    """Normalize a stencil number by stripping punctuation, whitespace, and converting to uppercase."""
    if s is None or pd.isna(s):
        return ""
    # Convert to string and strip outer spaces
    s = str(s).strip().upper()
    # If the model returned literal "NULL", treat as empty
    if s == "NULL":
        return ""
    # Remove all non-alphanumeric characters
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def download_image(url, temp_dir):
    """Download an image from a URL to a temporary file."""
    # Custom headers to bypass simple user-agent blocks
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    req = urllib.request.Request(url, headers=headers)

    # Guess extension from URL
    parsed_url = url.split('?')[0]
    ext = ".jpg"  # default fallback
    for possible_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']:
        if parsed_url.lower().endswith(possible_ext):
            ext = possible_ext
            break

    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=temp_dir)
    os.close(fd)

    try:
        with urllib.request.urlopen(req, timeout=20) as response, open(temp_path, 'wb') as out_file:
            out_file.write(response.read())
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def load_spreadsheet(path):
    """Loads a CSV or Excel spreadsheet using pandas."""
    if not os.path.exists(path):
        print(f"Error: Spreadsheet file not found at '{path}'", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(path)
        except ImportError:
            print("Error: Missing dependency 'openpyxl' required to read Excel files.", file=sys.stderr)
            print("Please run: pip install openpyxl", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback to CSV
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error: Failed to read CSV file: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tyre OCR model accuracy against spreadsheet ground truth."
    )
    parser.add_argument(
        "--spreadsheet",
        type=str,
        default=None,
        help="Path to CSV or Excel spreadsheet. Automatically detected if omitted."
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default=None,
        help="Column name containing image links/URLs/paths (auto-detected if omitted)."
    )
    parser.add_argument(
        "--stencil-col",
        type=str,
        default=None,
        help="Column name containing ground truth stencil numbers (auto-detected if omitted)."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of random rows to sample for testing (default: 20 if --rows is not set)."
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Range of rows to evaluate, formatted as start:end (e.g. '100:150')."
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
        "--output-csv",
        type=str,
        default="eval_results.csv",
        help="Path to save detailed evaluation report CSV (default: eval_results.csv)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)."
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # 1. Spreadsheet Auto-Detection
    spreadsheet_path = args.spreadsheet
    if not spreadsheet_path:
        candidates = []
        data_dir = "data"
        
        # Check 'data' directory in project root
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                    candidates.append(os.path.join(data_dir, f))
        
        # Also check current directory
        for f in os.listdir('.'):
            if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                candidates.append(f)
                
        if candidates:
            # Prefer files in data/ if available, otherwise check root
            data_folder_candidates = [c for c in candidates if c.startswith('data' + os.sep)]
            if data_folder_candidates:
                spreadsheet_path = sorted(data_folder_candidates)[0]
            elif 'data.csv' in candidates:
                spreadsheet_path = 'data.csv'
            else:
                spreadsheet_path = sorted(candidates)[0]
            print(f"Auto-detected spreadsheet: '{spreadsheet_path}'", file=sys.stderr)
        else:
            print("Error: No CSV or Excel files found in the current directory or 'data' folder.", file=sys.stderr)
            print("Please specify your file path with --spreadsheet.", file=sys.stderr)
            sys.exit(1)

    print("Loading spreadsheet...", file=sys.stderr)
    df = load_spreadsheet(spreadsheet_path)

    # 2. Columns Auto-Detection
    image_col = args.image_col
    stencil_col = args.stencil_col

    if not image_col or not stencil_col:
        # Lowercase headers to make search robust
        cols = list(df.columns)
        
        # Find image column candidate
        if not image_col:
            image_keywords = ['image', 'img', 'url', 'link', 'path', 'pic', 'photo', 'href']
            for c in cols:
                if any(k in c.lower() for k in image_keywords):
                    image_col = c
                    break
            if not image_col and len(cols) > 1:
                # Default fallback to second column
                image_col = cols[1]
                
        # Find stencil column candidate
        if not stencil_col:
            stencil_keywords = ['stencil', 'text', 'val', 'gt', 'ground', 'number', 'label', 'truth', 'actual']
            for c in cols:
                if c == image_col:
                    continue
                if any(k in c.lower() for k in stencil_keywords):
                    stencil_col = c
                    break
            if not stencil_col:
                # Default fallback to first column that is not image_col
                for c in cols:
                    if c != image_col:
                        stencil_col = c
                        break

    if not image_col or image_col not in df.columns:
        print(f"Error: Image column '{image_col}' not found. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if not stencil_col or stencil_col not in df.columns:
        print(f"Error: Stencil column '{stencil_col}' not found. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using columns -> Stencil (GT): '{stencil_col}', Image Link: '{image_col}'", file=sys.stderr)

    # Filter out rows with missing values
    clean_df = df.dropna(subset=[image_col, stencil_col])
    total_available = len(clean_df)
    
    if total_available == 0:
        print("Error: No valid rows found with non-null image links and stencil numbers.", file=sys.stderr)
        sys.exit(1)

    print(f"Spreadsheet loaded. Found {len(df)} total rows ({total_available} complete).", file=sys.stderr)

    # 3. Row Selection / Slicing Logic
    if args.rows:
        try:
            parts = args.rows.split(':')
            start = int(parts[0])
            end = int(parts[1])
            if start < 0 or end <= start or start >= len(clean_df):
                raise ValueError
            sampled_df = clean_df.iloc[start:end]
            sample_size = len(sampled_df)
            print(f"Selected row range {start} to {end} (Total: {sample_size} records).", file=sys.stderr)
        except Exception:
            print(f"Error: Invalid row range '{args.rows}'. Must be formatted as 'start:end' (e.g. 100:150).", file=sys.stderr)
            sys.exit(1)
    else:
        if args.sample_size is not None:
            sample_size = min(args.sample_size, total_available)
            sampled_df = clean_df.sample(n=sample_size, random_state=args.seed)
            print(f"Randomly sampled {sample_size} rows for evaluation.", file=sys.stderr)
        else:
            if total_available <= 20:
                sample_size = total_available
                sampled_df = clean_df
                print(f"Evaluating all {sample_size} rows.", file=sys.stderr)
            else:
                sample_size = 20
                sampled_df = clean_df.sample(n=sample_size, random_state=args.seed)
                print(f"Auto-sampled 20 random rows.", file=sys.stderr)
                print("Use --rows start:end or --sample-size N to evaluate a different selection.", file=sys.stderr)

    # Load OCR engine
    print(f"Initializing model '{args.model}'...", file=sys.stderr)
    try:
        engine = TyreOCREngine(model_id=args.model, use_mlx=args.mlx)
        engine.load_model()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nStarting Evaluation Loop...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    temp_dir = tempfile.mkdtemp(prefix="tyre_eval_")
    results = []
    
    exact_matches = 0
    normalized_matches = 0
    total_valid_extractions = 0
    total_download_failures = 0
    total_inference_failures = 0
    total_distance = 0
    total_cer = 0.0

    for idx, (_, row) in enumerate(sampled_df.iterrows(), 1):
        image_link = str(row[image_col]).strip()
        gt_stencil = str(row[stencil_col]).strip()
        norm_gt = normalize_stencil(gt_stencil)

        print(f"[{idx}/{sample_size}] Processing Image: {image_link[:60]}...", file=sys.stderr)
        print(f"      Ground Truth Stencil: '{gt_stencil}' (normalized: '{norm_gt}')", file=sys.stderr)

        local_path = None
        is_temp = False
        error_msg = ""
        download_success = True
        inference_success = True
        
        t0 = time.time()
        
        # Download image if it is a remote link
        if image_link.startswith(('http://', 'https://')):
            try:
                local_path = download_image(image_link, temp_dir)
                is_temp = True
            except Exception as e:
                download_success = False
                error_msg = f"Download Failed: {e}"
                print(f"      ❌ {error_msg}", file=sys.stderr)
                total_download_failures += 1
        else:
            # Check local file
            if os.path.exists(image_link):
                local_path = image_link
            else:
                download_success = False
                error_msg = f"Local file not found: {image_link}"
                print(f"      ❌ {error_msg}", file=sys.stderr)
                total_download_failures += 1

        pred_raw = ""
        norm_pred = ""
        is_exact = False
        is_norm_match = False
        dist = None
        cer = None
        confidence = "unknown"
        val_status = "unknown"
        elapsed = 0.0

        if download_success and local_path:
            try:
                # Run OCR extraction
                ocr_result = engine.extract_tyre_info(local_path, mode="stencil")
                elapsed = time.time() - t0
                
                pred_raw = ocr_result.get("stencilNumber") or ""
                norm_pred = normalize_stencil(pred_raw)
                confidence = ocr_result.get("confidence", "unknown")
                val_status = ocr_result.get("_validation", "unknown")

                # Metrics calculations
                is_exact = (pred_raw == gt_stencil)
                is_norm_match = (norm_pred == norm_gt)
                dist = levenshtein_distance(norm_gt, norm_pred)
                
                # CER calculation
                max_len = max(len(norm_gt), 1)
                cer = dist / max_len
                
                total_valid_extractions += 1
                total_distance += dist
                total_cer += cer

                if is_exact:
                    exact_matches += 1
                    print(f"      ✅ Exact Match! Predict: '{pred_raw}' ({elapsed:.2f}s)", file=sys.stderr)
                elif is_norm_match:
                    normalized_matches += 1
                    print(f"      🟨 Normalized Match! Predict: '{pred_raw}' (norm: '{norm_pred}') ({elapsed:.2f}s)", file=sys.stderr)
                else:
                    print(f"      ❌ Mismatch! Predict: '{pred_raw}' (norm: '{norm_pred}') (Dist: {dist}, CER: {cer:.1%}) ({elapsed:.2f}s)", file=sys.stderr)

            except Exception as e:
                inference_success = False
                error_msg = f"Inference Failed: {e}"
                print(f"      ❌ {error_msg}", file=sys.stderr)
                total_inference_failures += 1
            finally:
                # Clean up downloaded file
                if is_temp and local_path and os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
        
        # Log evaluation detail
        results.append({
            "image_link": image_link,
            "ground_truth": gt_stencil,
            "normalized_ground_truth": norm_gt,
            "predicted_raw": pred_raw,
            "normalized_prediction": norm_pred,
            "is_exact_match": is_exact,
            "is_normalized_match": is_norm_match or is_exact,
            "levenshtein_distance": dist,
            "cer": cer,
            "confidence": confidence,
            "validation_status": val_status,
            "download_success": download_success,
            "inference_success": inference_success,
            "elapsed_seconds": round(elapsed, 2),
            "error_message": error_msg
        })
        print("-" * 50, file=sys.stderr)

    # Clean up temp folder
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    # Calculations
    denom = max(1, total_valid_extractions)
    avg_cer = total_cer / denom
    avg_dist = total_distance / denom
    
    em_rate = exact_matches / sample_size
    nm_rate = (exact_matches + normalized_matches) / sample_size

    # Save CSV Report
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "image_link", "ground_truth", "normalized_ground_truth", 
            "predicted_raw", "normalized_prediction", "is_exact_match", 
            "is_normalized_match", "levenshtein_distance", "cer", 
            "confidence", "validation_status", "download_success", 
            "inference_success", "elapsed_seconds", "error_message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Output Summary Table
    print("\n" + "=" * 60)
    print("                 EVALUATION SUMMARY")
    print("=" * 60)
    print(f" Model Used:                 {args.model}")
    print(f" Total Rows Slices/Sampled:  {sample_size}")
    print(f"   - Successful Runs:        {total_valid_extractions}")
    print(f"   - Download Failures:      {total_download_failures}")
    print(f"   - Model/Inference Failures: {total_inference_failures}")
    print("-" * 60)
    print(f" Exact Match Accuracy (EM):   {em_rate * 100:.1f}% ({exact_matches}/{sample_size})")
    print(f" Normalized Match Accuracy:   {nm_rate * 100:.1f}% ({exact_matches + normalized_matches}/{sample_size})")
    print(f" Avg Levenshtein Distance:    {avg_dist:.2f} chars")
    print(f" Average Character Error Rate (CER): {avg_cer * 100:.1f}%")
    print("-" * 60)
    print(f" Detailed Report Saved to:    {args.output_csv}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
