import os
# ── Aggressive tqdm monitor kill ──────────────────────────────────────
# Python 3.13: tqdm's background monitor thread causes fatal GIL crashes.
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_MONITOR_INTERVAL"] = "0"

def _kill_tqdm_monitor():
    """Stop and remove the tqdm monitor thread if it was already started."""
    try:
        import tqdm
        monitor = getattr(tqdm, "_monitor", None) or getattr(tqdm.tqdm, "monitor", None)
        if monitor is not None:
            monitor.exit()
            try:
                monitor.join(timeout=2)
            except Exception:
                pass
        tqdm.tqdm.monitor_interval = 0
        tqdm.tqdm.monitor = None
        if hasattr(tqdm, "_monitor"):
            tqdm._monitor = None
    except Exception:
        pass

import json
import re
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Try to import MLX tools for optimized Mac support if requested and available
try:
    import mlx_vlm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# ============================================================
# Specialized Extraction Prompts
# ============================================================

# Mandatory anti-hallucination preamble injected into every prompt
_ANTI_HALLUCINATION = (
    "CRITICAL RULES — READ BEFORE ANSWERING:\n"
    "- If the image is blurry, unclear, out of focus, overexposed, or you cannot\n"
    "  confidently read the text, you MUST return null for all extraction fields.\n"
    "- Do NOT guess or invent characters you cannot clearly see.\n"
    "- Only return text you are highly confident about.\n"
    "- You MUST include a \"confidence\" field in your JSON response with one of:\n"
    "  \"high\" (clearly readable), \"medium\" (partially readable), or \"low\" (barely readable / guessing).\n"
    "- If confidence is \"low\", set the extracted value to null.\n\n"
)

STENCIL_PROMPT = (
    _ANTI_HALLUCINATION +
    "You are an OCR and tyre-marking extraction specialist.\n"
    "Your task is to extract the Tyre Stencil Number from a tyre sidewall image.\n\n"
    "A valid Tyre Stencil Number follows one of these formats:\n"
    "- 5 characters: PWWYY\n"
    "- 11 characters: PXXXXXXXWWYY\n\n"
    "Where P = Plant Code (one of B, C, K, V, R, H, M, S, P), "
    "WW = Manufacturing Week (01-53), YY = Manufacturing Year.\n\n"
    "The first character in the FINAL OUTPUT must always be the Plant Code. "
    "All remaining characters must be numeric.\n\n"
    "Examples: B5224, C1221, H3088320725, H3123131225\n\n"
    "IMPORTANT EXTRACTION RULES:\n"
    "- Stencil numbers may be embossed, engraved, molded, or raised from rubber.\n"
    "- They may be partially covered by dirt/dust, viewed at an angle, or rotated.\n"
    "- They may appear as one continuous string or split into groups.\n"
    "- If the Plant Code appears at the END of a numeric sequence, move it to the FRONT.\n"
    "  Example: 5224B \u2192 B5224, 1924 H \u2192 H1924\n\n"
    "OCR NORMALIZATION:\n"
    "- O\u21920, Q\u21920, D\u21920 (in numeric context), I\u21921, l\u21921, |\u21921, Z\u21922, S\u21925 (in numeric context)\n"
    "- B\u21948: use surrounding characters to determine if Plant Code or digit.\n"
    "- Remove spaces, dashes, dots, underscores, separators.\n\n"
    "VALIDATION:\n"
    "- Short format: ^[BCKVRHMSP][0-9]{4}$\n"
    "- Long format: ^[BCKVRHMSP][0-9]{10}$\n"
    "- Prefer long format over short format.\n"
    "- Last 4 digits are WWYY where WW=01-53.\n\n"
    "Return ONLY valid JSON:\n"
    '{"stencilNumber": "<value_or_null>", "confidence": "high|medium|low", "debugMessage": "<explanation>"}\n'
    "Do not include any introductory or concluding text. Return only the JSON."
)

TYRE_SIZE_PROMPT = (
    _ANTI_HALLUCINATION +
    "You are an OCR specialist for tyres.\n"
    "Your task is to extract from the tyre image:\n"
    '1. The tyre size (e.g. "10.00R20", "295/80R22.5", "10.00-20", "205/55R16", "225/45ZR17")\n'
    '2. The pattern name (e.g. "JET XTRA", "JUH5", "JETSTEEL Y4", "Pilot Sport", "Turanza")\n'
    "3. Strictly return only the text you are confident about, not guessed characters.\n\n"
    "Return ONLY valid JSON:\n"
    '{"tyreSize": "<size_or_null>", "patternName": "<pattern_or_null>", "confidence": "high|medium|low", "debugMessage": "<explanation>"}\n'
    "Do not include any introductory or concluding text. Return only the JSON."
)

TUBE_SIZE_PROMPT = (
    _ANTI_HALLUCINATION +
    "You are an OCR specialist.\n"
    'Your task is to extract the Tube Size (e.g. "10.00-20", "7.50-16", "10.00R20") from the image.\n'
    "Strictly return only the letters/numbers you are confident about, not guessed characters.\n\n"
    "Return ONLY valid JSON:\n"
    '{"tubeSize": "<size_or_null>", "confidence": "high|medium|low", "debugMessage": "<explanation>"}\n'
    "Do not include any introductory or concluding text. Return only the JSON."
)

FLAP_SIZE_PROMPT = (
    _ANTI_HALLUCINATION +
    "You are an OCR specialist.\n"
    'Your task is to extract the Flap Size (e.g. "20", "16") from the image.\n'
    "Strictly return only the letters/numbers you are confident about, not guessed characters.\n\n"
    "Return ONLY valid JSON:\n"
    '{"flapSize": "<size_or_null>", "confidence": "high|medium|low", "debugMessage": "<explanation>"}\n'
    "Do not include any introductory or concluding text. Return only the JSON."
)

EXTRACTION_PROMPTS = {
    "stencil": STENCIL_PROMPT,
    "tyre_size": TYRE_SIZE_PROMPT,
    "tube_size": TUBE_SIZE_PROMPT,
    "flap_size": FLAP_SIZE_PROMPT,
}

# ============================================================
# Validation Patterns
# ============================================================

_TYRE_SIZE     = re.compile(r"^\d{2,3}[./]?\d{0,3}\s*[-/]?\s*[A-Z]?[RZ]?\s*\d{2}(\.\d)?$", re.IGNORECASE)
_TUBE_SIZE     = re.compile(r"^\d+\.\d+\s*[-R]\s*\d+$", re.IGNORECASE)
_FLAP_SIZE     = re.compile(r"^\d{2}$")



# Quality assessment thresholds  (relaxed for real-world tyre photos)
BLUR_THRESHOLD     = 15.0   # Laplacian variance; lower = blurrier
CONTRAST_THRESHOLD = 12.0   # Grayscale std dev; lower = flatter
MIN_RESOLUTION     = 200    # Minimum dimension in pixels

# Stencil pattern regexes for post-processing extraction
_STENCIL_SHORT = re.compile(r'[BCKVRHMSP]\d{4}')
_STENCIL_LONG  = re.compile(r'[BCKVRHMSP]\d{10}')

# OCR character normalization map (ambiguous characters on rubber)
_OCR_CHAR_MAP = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1',
    'Z': '2',
    'S': '5',
})


class TyreOCREngine:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", use_mlx=False):
        """
        Initializes the Tyre OCR Engine.
        
        Args:
            model_id (str): HuggingFace model identifier.
            use_mlx (bool): If True and on Apple Silicon, use MLX for faster inference.
        """
        self.model_id = model_id
        self.use_mlx = use_mlx and MLX_AVAILABLE
        self.model = None
        self.processor = None
        self.mlx_config = None
        self.device = None
        self.dtype = None
        self.trocr_processor = None
        self.trocr_model = None

    @property
    def is_yolo_model(self):
        """
        Checks if the selected model is a YOLOv11 / Ultralytics YOLO model.
        """
        model_lower = self.model_id.lower()
        return "yolo11" in model_lower or model_lower.endswith(".pt") or "yolov11" in model_lower

    def load_model(self):
        """
        Loads the model and processor into memory.
        """
        if self.is_yolo_model:
            print(f"Loading YOLO model: {self.model_id}...")
            from ultralytics import YOLO
            self.model = YOLO(self.model_id)
            print("YOLO Model loaded successfully.")
            
            print("Loading TrOCR model (microsoft/trocr-small-printed)...")
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
            
            # Determine best device for TrOCR
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            self.trocr_model.to(device)
            print(f"TrOCR Model loaded successfully on device: {device}.")
            _kill_tqdm_monitor()
            return

        if self.use_mlx:
            print(f"Loading {self.model_id} via MLX for optimized Apple Silicon execution...")
            # Quantized MLX models are usually hosted under mlx-community
            # We map standard model ID to mlx-community counterpart if standard is passed
            mlx_model_id = self.model_id
            if "mlx-community" not in mlx_model_id:
                # E.g. Qwen/Qwen2-VL-2B-Instruct -> mlx-community/Qwen2-VL-2B-Instruct-4bit
                name = self.model_id.split("/")[-1]
                mlx_model_id = f"mlx-community/{name}-4bit"
            
            try:
                self.model, self.processor = mlx_vlm.load(mlx_model_id)
                self.mlx_config = mlx_vlm.utils.load_config(mlx_model_id)
                print("MLX Model loaded successfully.")
                _kill_tqdm_monitor()
                return
            except Exception as e:
                print(f"Failed to load MLX model: {e}. Falling back to standard PyTorch/Transformers...")
                self.use_mlx = False

        # Determine best PyTorch device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            # Some M-series chips have better support for float16 over bfloat16
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"Loading {self.model_id} using PyTorch on device: {self.device} ({self.dtype})...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model with optimal settings
        model_kwargs = {
            "torch_dtype": self.dtype,
        }
        
        if self.device != "cpu":
            # auto maps to available GPU/MPS automatically
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None
            
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        if self.device == "cpu":
            # Explicitly move to CPU if no device map was set
            self.model = self.model.to(self.device)

        print("PyTorch Model loaded successfully.")
        _kill_tqdm_monitor()

    def preprocess_image(self, image_path):
        """
        Enhances a tyre image to make embossed/engraved text more visible.
        
        Pipeline:
            1. Convert to grayscale
            2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            3. Bilateral filter for edge-preserving denoising
            4. Unsharp mask for sharpening
            5. Convert back to 3-channel BGR
        
        Returns:
            str: Path to the preprocessed image (temp file).
        """
        import tempfile
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return image_path  # fallback to original
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. CLAHE — dramatically improves embossed text on rubber
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Bilateral filter — smooths noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Unsharp mask — sharpens text edges
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        
        # Convert back to 3-channel for VLM input
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        # Save to temp file
        _, ext = os.path.splitext(image_path)
        ext = ext if ext else '.jpg'
        fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='tyre_enhanced_')
        os.close(fd)
        cv2.imwrite(temp_path, result)
        
        return temp_path

    def assess_image_quality(self, image_path):
        """
        Analyzes image quality to detect blur, low contrast, and low resolution.
        
        Returns:
            dict with keys: is_usable, blur_score, contrast_score, resolution, warnings
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {
                "is_usable": False,
                "blur_score": 0.0,
                "contrast_score": 0.0,
                "resolution": [0, 0],
                "warnings": ["Could not read image file"]
            }
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        warnings = []
        
        # Blur detection via Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < BLUR_THRESHOLD:
            warnings.append(f"Image is blurry (score: {blur_score:.1f}, threshold: {BLUR_THRESHOLD})")
        
        # Contrast detection via standard deviation of pixel values
        contrast_score = float(np.std(gray))
        if contrast_score < CONTRAST_THRESHOLD:
            warnings.append(f"Image has low contrast (score: {contrast_score:.1f}, threshold: {CONTRAST_THRESHOLD})")
        
        # Resolution check
        min_dim = min(h, w)
        if min_dim < MIN_RESOLUTION:
            warnings.append(f"Image resolution too low ({w}x{h}, minimum dimension: {MIN_RESOLUTION}px)")
        
        is_usable = len(warnings) == 0
        
        return {
            "is_usable": is_usable,
            "blur_score": round(blur_score, 1),
            "contrast_score": round(contrast_score, 1),
            "resolution": [w, h],
            "warnings": warnings
        }

    def _validate_result(self, result, mode):
        """
        Validates extracted values against known patterns for each mode.
        Adds _validation and _validation_reason keys to the result dict.
        """
        val = None
        valid = True
        reason = ""
        
        # Stencil checking is bypassed as requested; everything else is validated
        if mode == "tyre_size":
            val = result.get("tyreSize")
            if val and val != "null":
                if not _TYRE_SIZE.match(val.strip()):
                    valid = False
                    reason = f"'{val}' doesn't match common tyre size patterns"
        elif mode == "tube_size":
            val = result.get("tubeSize")
            if val and val != "null":
                if not _TUBE_SIZE.match(val.strip()):
                    valid = False
                    reason = f"'{val}' doesn't match tube size format (e.g. 10.00-20)"
        elif mode == "flap_size":
            val = result.get("flapSize")
            if val and val != "null":
                stripped = val.strip()
                if not _FLAP_SIZE.match(stripped):
                    valid = False
                    reason = f"'{val}' doesn't match flap size format (2-digit number)"
                else:
                    num = int(stripped)
                    if num < 10 or num > 24:
                        valid = False
                        reason = f"Flap size {num} outside expected range (10-24)"
        
        # If model reported low confidence, also flag as suspicious
        confidence = result.get("confidence", "unknown")
        if confidence == "low":
            valid = False
            reason = (reason + "; " if reason else "") + "Model reported low confidence"
        
        result["_validation"] = "PASS" if valid else "FAIL"
        if reason:
            result["_validation_reason"] = reason
        
        return result

    def extract_tyre_info(self, image_path, mode="stencil"):
        """
        Extracts tyre specifications or runs object detection from an image.
        
        Args:
            image_path (str): Local path to the tyre image.
            mode (str): Extraction mode — 'stencil', 'tyre_size', 'tube_size', 'flap_size'.
            
        Returns:
            dict: Parsed tyre details or detection results.
        """
        # Ensure model is loaded
        if self.model is None or (not self.is_yolo_model and self.processor is None):
            self.load_model()

        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Verify image format using PIL
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

        # ── Image preprocessing ──────────────────────────────────────
        # Enhance the image BEFORE quality assessment so embossed text
        # on dark rubber becomes visible (CLAHE + denoise + sharpen).
        preprocessed_path = None
        inference_path = image_path
        if not self.is_yolo_model:
            preprocessed_path = self.preprocess_image(image_path)
            inference_path = preprocessed_path

        # ── Image quality gate (on preprocessed image) ─────────────
        quality = self.assess_image_quality(inference_path)
        
        if not quality["is_usable"] and not self.is_yolo_model:
            # Clean up temp file before returning
            if preprocessed_path and os.path.exists(preprocessed_path):
                try: os.remove(preprocessed_path)
                except Exception: pass
            # Return early with quality failure instead of hallucinated output
            return {
                "_mode": mode,
                "_quality": quality,
                "_validation": "SKIPPED",
                "_validation_reason": "Image quality too low for reliable extraction",
                "confidence": "low",
                "debugMessage": "Image skipped: " + "; ".join(quality["warnings"]),
            }

        if self.is_yolo_model:
            # Run YOLO inference
            results = self.model(image_path)
            result = results[0]
            
            # Save annotated image in scratch directory
            scratch_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
            os.makedirs(scratch_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            annotated_filename = f"{name}_annotated{ext}"
            annotated_path = os.path.join(scratch_dir, annotated_filename)
            
            annotated_img_bgr = result.plot()
            annotated_img_rgb = annotated_img_bgr[:, :, ::-1]
            pil_img = Image.fromarray(annotated_img_rgb)
            pil_img.save(annotated_path)
            
            # Parse detections and run TrOCR on cropped regions
            detections = []
            boxes = result.boxes
            names = result.names
            
            with Image.open(image_path) as full_img:
                width, height = full_img.size
                
                for box in boxes:
                    cls_idx = int(box.cls[0].item())
                    class_name = names[cls_idx]
                    confidence = float(box.conf[0].item())
                    xyxy = [float(val) for val in box.xyxy[0].tolist()]
                    
                    # Crop image using integer pixel coordinates
                    xmin = max(0, int(xyxy[0]))
                    ymin = max(0, int(xyxy[1]))
                    xmax = min(width, int(xyxy[2]))
                    ymax = min(height, int(xyxy[3]))
                    
                    extracted_text = ""
                    if xmax > xmin and ymax > ymin:
                        try:
                            cropped_img = full_img.crop((xmin, ymin, xmax, ymax))
                            # Run TrOCR
                            pixel_values = self.trocr_processor(images=cropped_img, return_tensors="pt").pixel_values.to(self.trocr_model.device)
                            with torch.no_grad():
                                generated_ids = self.trocr_model.generate(pixel_values)
                            extracted_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        except Exception as e:
                            print(f"TrOCR extraction failed for box {xyxy}: {e}")
                            extracted_text = "[Error]"
                    
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": xyxy,
                        "text": extracted_text
                    })
                
            return {
                "is_detection": True,
                "detections": detections,
                "annotated_image_path": annotated_path,
                "model_id": self.model_id
            }

        # Select the prompt based on the extraction mode
        prompt = EXTRACTION_PROMPTS.get(mode, STENCIL_PROMPT)

        if self.use_mlx:
            formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
                self.processor, self.mlx_config, prompt, num_images=1
            )
            raw_output = mlx_vlm.generate(
                self.model, self.processor, formatted_prompt, [inference_path], verbose=False
            )
            if hasattr(raw_output, "text"):
                raw_output = raw_output.text
        else:
            # Prepare inputs in HF chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": inference_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Preprocess
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False  # Greedy decoding for consistent structured OCR
                )
                
            generated_ids = [ids[len(inputs["input_ids"][0]):] for ids in output_ids]
            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

        result = self._clean_and_parse_json(raw_output)
        result["_mode"] = mode
        result["_quality"] = quality
        
        # ── Stencil post-processing: regex extraction fallback ──────
        if mode == "stencil":
            result = self._postprocess_stencil(result, raw_output)
        
        # ── Post-extraction validation ──────────────────────────────
        result = self._validate_result(result, mode)
        
        # ── Clean up preprocessed temp file ─────────────────────────
        if preprocessed_path and os.path.exists(preprocessed_path):
            try: os.remove(preprocessed_path)
            except Exception: pass
        
        return result

    def _clean_and_parse_json(self, raw_output):
        """
        Parses raw model output into a valid Python dictionary, cleaning up markdown code blocks if present.
        """
        cleaned = raw_output.strip()
        
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("Warning: Direct JSON parsing failed. Attempting cleanup extraction...")
            
            # Try to extract content between curly braces
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Generic fallback — return raw text in a debugMessage
            return {
                "debugMessage": f"Raw model output (JSON parse failed): {raw_output}",
                "_raw": raw_output
            }

    def _postprocess_stencil(self, result, raw_output):
        """
        Post-processing for stencil mode: if the model's stencilNumber doesn't
        match a valid stencil pattern, attempt to regex-extract one from the
        raw output text and apply OCR character normalization.
        """
        current = result.get("stencilNumber") or ""
        
        # Check if the current value already looks like a valid stencil
        if current:
            normalized = self._normalize_stencil_chars(current)
            if _STENCIL_LONG.fullmatch(normalized) or _STENCIL_SHORT.fullmatch(normalized):
                result["stencilNumber"] = normalized
                return result
        
        # Otherwise, search the raw output and debugMessage for stencil patterns
        search_text = raw_output + " " + result.get("debugMessage", "")
        search_text = search_text.upper()
        
        # Prefer long format (11 chars) over short (5 chars)
        candidates = []
        for match in _STENCIL_LONG.finditer(search_text):
            candidates.append((match.group(), 11))
        for match in _STENCIL_SHORT.finditer(search_text):
            candidates.append((match.group(), 5))
        
        if candidates:
            # Sort: prefer long format, then first occurrence
            candidates.sort(key=lambda x: -x[1])
            best = candidates[0][0]
            best = self._normalize_stencil_chars(best)
            
            if best != current:
                old_debug = result.get("debugMessage", "")
                result["stencilNumber"] = best
                result["debugMessage"] = (
                    f"Post-processing: regex-extracted '{best}' from model output. "
                    f"Original value: '{current}'. {old_debug}"
                )
        elif current:
            # Try normalizing what we have even if it didn't fully match
            normalized = self._normalize_stencil_chars(current)
            # Remove non-alphanumeric
            normalized = re.sub(r'[^A-Z0-9]', '', normalized)
            if _STENCIL_LONG.fullmatch(normalized) or _STENCIL_SHORT.fullmatch(normalized):
                result["stencilNumber"] = normalized
        
        return result
    
    def _normalize_stencil_chars(self, text):
        """
        Apply OCR normalization rules for stencil numbers:
        O→0, Q→0, D→0 (in numeric context), I→1, l→1, |→1, Z→2, S→5.
        Also strips spaces, dashes, dots.
        """
        text = text.upper().strip()
        # Remove separators
        text = re.sub(r'[\s\-\._]+', '', text)
        
        if not text:
            return text
        
        # Keep the first char as-is if it's a valid plant code
        plant_codes = set('BCKVRHMSP')
        chars = list(text)
        
        for i in range(len(chars)):
            if i == 0 and chars[i] in plant_codes:
                continue  # preserve plant code letter
            # Apply numeric context normalization for positions after plant code
            if chars[i] in _OCR_CHAR_MAP:
                chars[i] = chr(_OCR_CHAR_MAP[ord(chars[i])])
        
        return ''.join(chars)

    def _regex_extract(self, text, pattern):
        """Helper to extract values from text if JSON parsing fails."""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None