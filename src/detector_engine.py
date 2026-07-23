import os
import io
import cv2
import json
import math
import numpy as np
from PIL import Image
import requests


class BrandShieldDetector:
    """
    Multi-Stage Forensic Logo Detector.
    Stage 1: OpenCV Multi-Metric Structural Forensics (offline — always runs)
    Stage 2: Google Gemini 2.0 Flash Vision AI (online — when API key is configured)
    """

    SUPPORTED_BRANDS = [
        "Nike", "Adidas", "Apple", "Starbucks",
        "Gucci", "Louis Vuitton", "Rolex", "Puma", "Samsung", "Custom / Unspecified"
    ]

    def __init__(self):
        # ── Load API key from every possible source ──────────────────────
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

        # Streamlit Cloud stores secrets in st.secrets, NOT env vars
        if not api_key:
            try:
                import streamlit as _st
                api_key = _st.secrets.get("GEMINI_API_KEY", None) or _st.secrets.get("GOOGLE_API_KEY", None)
            except Exception:
                pass

        self.gemini_available = False
        self._client = None

        if api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key)
                self.gemini_available = True
            except Exception:
                pass

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: OpenCV Multi-Metric Structural Forensics
    # ══════════════════════════════════════════════════════════════════════
    def _opencv_forensics(self, img_bgr: np.ndarray) -> dict:
        """Runs 6 independent structural forensic checks on the image."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # ── 1. Edge Density (Canny) ──────────────────────────────────────
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # ── 2. ORB Keypoints ─────────────────────────────────────────────
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        num_keypoints = len(keypoints) if keypoints else 0

        # ── 3. Blur / Sharpness (Laplacian Variance) ────────────────────
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # High = sharp/authentic, Low = blurry/compressed fake
        sharpness_score = min(100.0, laplacian_var / 5.0)

        # ── 4. Color Uniformity (Histogram Std Dev) ──────────────────────
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        # Real logos have consistent, limited color palettes
        color_spread = float(np.std(h_hist))

        # ── 5. Symmetry Score (flip and compare) ────────────────────────
        flipped = cv2.flip(gray, 1)  # horizontal flip
        diff = cv2.absdiff(gray, flipped)
        symmetry_score = 100.0 - float(np.mean(diff) / 2.55)  # 0-100

        # ── 6. Contour Regularity ───────────────────────────────────────
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        if contours:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            contour_regularity = float(np.std(areas)) if areas else 0.0
        else:
            contour_regularity = 0.0

        # ── Visualization images ────────────────────────────────────────
        kp_img = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

        # ── Composite Structural Score ──────────────────────────────────
        # Weight each metric to compute a blended structural score
        score = 50.0  # neutral baseline

        # Sharpness: sharp logos are more likely authentic
        if sharpness_score > 60:
            score += 15.0
        elif sharpness_score < 20:
            score -= 20.0

        # Keypoints: too few = simple/degraded, too many = noisy
        if 30 <= num_keypoints <= 300:
            score += 10.0
        elif num_keypoints < 10:
            score -= 15.0

        # Edge density: very high = noisy counterfeit, very low = blank
        if 0.03 <= edge_density <= 0.30:
            score += 10.0
        else:
            score -= 10.0

        # Symmetry: logos tend to be symmetric
        if symmetry_score > 70:
            score += 10.0
        elif symmetry_score < 40:
            score -= 10.0

        # Contour regularity: authentic logos have clean, consistent contours
        if num_contours > 0 and contour_regularity < 5000:
            score += 5.0
        elif contour_regularity > 15000:
            score -= 10.0

        score = max(10.0, min(99.0, score))

        return {
            "edge_density":       round(edge_density, 4),
            "keypoints_count":    num_keypoints,
            "sharpness_score":    round(sharpness_score, 1),
            "symmetry_score":     round(symmetry_score, 1),
            "color_spread":       round(color_spread, 1),
            "num_contours":       num_contours,
            "contour_regularity": round(contour_regularity, 1),
            "structural_score":   round(score, 1),
            "kp_image_rgb":       cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB),
            "heatmap_rgb":        cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Gemini 2.0 Flash Vision AI Audit
    # ══════════════════════════════════════════════════════════════════════
    def _run_gemini_vision(self, image_pil: Image.Image, target_brand: str) -> dict | None:
        if not self.gemini_available:
            return None
        try:
            from google.genai import types

            buf = io.BytesIO()
            image_pil.save(buf, format="PNG")
            image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

            prompt = f"""You are a forensic brand-protection specialist.
Analyze this logo image for brand: "{target_brand}".

CRITICAL RULES — apply in order:
1. BRAND IDENTITY: First, identify what brand this logo actually represents.
   If the logo belongs to a DIFFERENT brand than "{target_brand}" → COUNTERFEIT, score=5.
2. VISUAL AUTHENTICITY: Check for:
   - Typography errors (wrong font, spacing, kerning)
   - Color deviations from official brand guidelines
   - Geometry distortions (stretched, skewed, asymmetric)
   - Resolution artifacts or pixelation
   - Missing or altered trademark symbols (™, ®)
3. OVERALL QUALITY: Professional vector-quality vs. amateur raster copy.

Return ONLY a JSON object with these exact keys:
{{
  "detected_brand": "the brand you actually see in the image",
  "matches_target": true,
  "authenticity_score": 92.0,
  "verdict": "AUTHENTIC",
  "threat_level": "LOW",
  "forensic_reasons": [
    "Brand identity matches target: {target_brand}",
    "Typography is consistent with official guidelines"
  ]
}}

If counterfeit or fake: verdict="COUNTERFEIT", authenticity_score<40, threat_level="HIGH".
If brand mismatch: verdict="COUNTERFEIT", authenticity_score=5, threat_level="CRITICAL".
"""
            # gemini-2.0-flash is the fastest and most capable current model
            for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
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
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: Full Analysis Pipeline
    # ══════════════════════════════════════════════════════════════════════
    def analyze_logo(self, image_pil: Image.Image, target_brand: str = "Nike") -> dict:
        img_np  = np.array(image_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Stage 1: OpenCV structural forensics (always runs)
        cv_results = self._opencv_forensics(img_bgr)

        # Stage 2: Gemini Vision AI (runs if API key is configured)
        ai_verdict = self._run_gemini_vision(image_pil, target_brand)

        ai_used = False
        if ai_verdict and isinstance(ai_verdict, dict):
            ai_used          = True
            final_score      = float(ai_verdict.get("authenticity_score", cv_results["structural_score"]))
            is_authentic     = ai_verdict.get("verdict", "").upper() == "AUTHENTIC"
            threat_level     = ai_verdict.get("threat_level", "LOW" if is_authentic else "HIGH")
            detected_brand   = ai_verdict.get("detected_brand", "Unknown")
            forensic_reasons = ai_verdict.get("forensic_reasons", [])
            # Prepend AI source label
            forensic_reasons.insert(0, f"🤖 Gemini 2.0 Flash Vision AI — Detected brand: **{detected_brand}**")
        else:
            final_score    = cv_results["structural_score"]
            is_authentic   = final_score >= 70.0
            threat_level   = "LOW" if is_authentic else ("HIGH" if final_score < 45.0 else "MEDIUM")
            forensic_reasons = [
                "⚠️ Running in **OpenCV-only mode** (Gemini API key not configured)",
                f"Sharpness score: {cv_results['sharpness_score']}  (Laplacian variance)",
                f"Symmetry score: {cv_results['symmetry_score']}%",
                f"Edge density: {cv_results['edge_density']}",
                f"ORB Keypoints: {cv_results['keypoints_count']}",
                f"Contours detected: {cv_results['num_contours']}",
                f"Color spread (Hue σ): {cv_results['color_spread']}",
                "ℹ️ Add `GEMINI_API_KEY` in Streamlit Secrets for real brand-mismatch & counterfeit detection.",
            ]

        return {
            "target_brand":       target_brand,
            "authenticity_score": round(final_score, 1),
            "is_authentic":       is_authentic,
            "verdict_label":      "AUTHENTIC" if is_authentic else "COUNTERFEIT / ALTERED",
            "threat_level":       threat_level,
            "edge_density":       cv_results["edge_density"],
            "keypoints_count":    cv_results["keypoints_count"],
            "sharpness_score":    cv_results["sharpness_score"],
            "symmetry_score":     cv_results["symmetry_score"],
            "keypoint_image":     Image.fromarray(cv_results["kp_image_rgb"]),
            "heatmap_image":      Image.fromarray(cv_results["heatmap_rgb"]),
            "forensic_reasons":   forensic_reasons,
            "ai_used":            ai_used,
        }
