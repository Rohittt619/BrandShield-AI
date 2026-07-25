import os
import io
import cv2
import json
import numpy as np
from PIL import Image
import requests


class BrandShieldDetector:
    """
    Multi-Stage Forensic Logo Detector.
    Stage 1: OpenCV Structural Forensics (image quality metrics)
    Stage 2: Google Gemini Vision AI (brand identification + counterfeit detection)
    """

    SUPPORTED_BRANDS = [
        "Nike", "Adidas", "Apple", "Starbucks",
        "Gucci", "Louis Vuitton", "Rolex", "Puma", "Samsung", "Custom / Unspecified"
    ]

    CANDIDATE_MODELS = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    )

    def __init__(self, api_key: str = None):
        self.gemini_available = False
        self._client = None
        self._init_error = None
        self.last_vision_errors: dict = {}

        if api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=str(api_key).strip())
                self.gemini_available = True
            except Exception as e:
                self._init_error = str(e)

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")

    def test_api_key(self) -> dict:
        if not self._client:
            return {
                "ok": False,
                "summary": "No Gemini client initialized.",
                "init_error": self._init_error,
            }

        results = {}
        any_ok = False
        for model_name in self.CANDIDATE_MODELS:
            try:
                resp = self._client.models.generate_content(
                    model=model_name,
                    contents=["Reply with the single word: OK"],
                )
                text = (resp.text or "").strip()
                results[model_name] = f"OK - responded: {text!r}"
                any_ok = True
            except Exception as e:
                results[model_name] = f"FAILED - {type(e).__name__}: {e}"

        return {
            "ok": any_ok,
            "summary": "At least one model works." if any_ok else "All candidate models failed - see details below.",
            "models": results,
        }

    def _opencv_forensics(self, img_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_density = float(np.mean(edges > 0))

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, _ = orb.detectAndCompute(gray, None)
        num_keypoints = len(keypoints) if keypoints else 0

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(100.0, lap_var / 5.0)

        flipped = cv2.flip(gray, 1)
        diff = cv2.absdiff(gray, flipped)
        symmetry = 100.0 - float(np.mean(diff) / 2.55)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        kp_img  = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

        return {
            "edge_density":    round(edge_density, 4),
            "keypoints_count": num_keypoints,
            "sharpness_score": round(sharpness, 1),
            "symmetry_score":  round(symmetry, 1),
            "num_contours":    num_contours,
            "kp_image_rgb":    cv2.cvtColor(kp_img,  cv2.COLOR_BGR2RGB),
            "heatmap_rgb":     cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        }

    def _run_gemini_vision(self, image_pil: Image.Image, target_brand: str):
        if not self.gemini_available or not self._client:
            return None

        self.last_vision_errors = {}

        try:
            from google.genai import types
        except Exception as e:
            self.last_vision_errors["import"] = f"{type(e).__name__}: {e}"
            return None

        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        prompt = f"""You are a forensic brand-protection specialist.
Analyze this logo image. The user claims it belongs to: "{target_brand}".

STEP 1 - BRAND IDENTIFICATION:
What brand does this logo ACTUALLY represent? Look at the shape, text, and design.

STEP 2 - BRAND MATCH:
Does the detected brand match "{target_brand}"?
- If NO (e.g., user selected Nike but image shows Adidas) -> COUNTERFEIT, score=5, threat="CRITICAL"
- If YES -> proceed to Step 3

STEP 3 - AUTHENTICITY CHECK (only if brand matches):
Examine for counterfeit indicators:
- Typography: wrong font, spacing, weight
- Colors: off-brand colors, wrong gradients
- Geometry: asymmetric, distorted, stretched
- Quality: pixelated, blurry, artifacts
- Trademark: missing TM or R where expected

Score 80-99 for authentic logos.
Score 20-50 for suspected counterfeits.
Score 1-10 for obvious fakes or wrong brand.

Return ONLY this JSON:
{{
  "detected_brand": "the actual brand in the image",
  "matches_target": true,
  "authenticity_score": 95.0,
  "verdict": "AUTHENTIC",
  "threat_level": "LOW",
  "forensic_reasons": [
    "Brand identity: Nike swoosh confirmed",
    "Typography: consistent with official brand"
  ]
}}
"""
        for model_name in self.CANDIDATE_MODELS:
            try:
                resp = self._client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image_part],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                if resp.text and resp.text.strip():
                    return json.loads(resp.text)
                self.last_vision_errors[model_name] = "Empty response from model"
            except Exception as e:
                self.last_vision_errors[model_name] = f"{type(e).__name__}: {e}"
                continue

        return None

    def analyze_logo(self, image_pil: Image.Image, target_brand: str = "Nike") -> dict:
        img_np  = np.array(image_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv = self._opencv_forensics(img_bgr)
        ai = self._run_gemini_vision(image_pil, target_brand)

        ai_used = False
        ai_error_summary = None
        if ai and isinstance(ai, dict) and "verdict" in ai:
            ai_used        = True
            final_score    = float(ai.get("authenticity_score", 50.0))
            verdict_raw    = ai.get("verdict", "").upper()
            is_authentic   = verdict_raw == "AUTHENTIC"
            threat_level   = ai.get("threat_level", "LOW" if is_authentic else "HIGH")
            detected_brand = ai.get("detected_brand", "Unknown")
            reasons        = ai.get("forensic_reasons", []) or []
            reasons.insert(0, f"Gemini Vision detected brand: **{detected_brand}**")
        else:
            final_score  = 0.0
            is_authentic = None
            threat_level = "UNKNOWN"
            detected_brand = None
            reasons = [
                f"Sharpness: {cv['sharpness_score']} (Laplacian variance)",
                f"Symmetry: {cv['symmetry_score']}% (horizontal flip)",
                f"Edge density: {cv['edge_density']} (Canny detection)",
                f"ORB keypoints: {cv['keypoints_count']} vectors",
                f"Contours: {cv['num_contours']} shapes",
            ]
            if self.last_vision_errors:
                ai_error_summary = "; ".join(
                    f"{model}: {err}" for model, err in self.last_vision_errors.items()
                )
                reasons.append(f"Gemini Vision unavailable - {ai_error_summary}")

        return {
            "target_brand":       target_brand,
            "authenticity_score": round(final_score, 1),
            "is_authentic":       is_authentic,
            "verdict_label":      "AUTHENTIC" if is_authentic else ("COUNTERFEIT / ALTERED" if is_authentic is False else "STRUCTURAL SCAN ONLY"),
            "threat_level":       threat_level,
            "edge_density":       cv["edge_density"],
            "keypoints_count":    cv["keypoints_count"],
            "sharpness_score":    cv["sharpness_score"],
            "symmetry_score":     cv["symmetry_score"],
            "keypoint_image":     Image.fromarray(cv["kp_image_rgb"]),
            "heatmap_image":      Image.fromarray(cv["heatmap_rgb"]),
            "forensic_reasons":   reasons,
            "ai_used":            ai_used,
            "ai_error":           ai_error_summary,
        }
