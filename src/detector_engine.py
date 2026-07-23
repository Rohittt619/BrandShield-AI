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
    Stage 1: OpenCV 6-Metric Structural Forensics (always runs)
    Stage 2: Google Gemini 2.0 Flash Vision AI (when API key is configured)
    """

    SUPPORTED_BRANDS = [
        "Nike", "Adidas", "Apple", "Starbucks",
        "Gucci", "Louis Vuitton", "Rolex", "Puma", "Samsung", "Custom / Unspecified"
    ]

    def __init__(self):
        api_key = None

        # Source 1: Environment variables (local dev)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

        # Source 2: Streamlit Cloud secrets (st.secrets["KEY"])
        if not api_key:
            try:
                import streamlit as _st
                try:
                    api_key = _st.secrets["GEMINI_API_KEY"]
                except Exception:
                    pass
                if not api_key:
                    try:
                        api_key = _st.secrets["GOOGLE_API_KEY"]
                    except Exception:
                        pass
            except Exception:
                pass

        self.gemini_available = False
        self._client = None
        self._init_error = None

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

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: OpenCV Multi-Metric Structural Forensics
    # ══════════════════════════════════════════════════════════════════════
    def _opencv_forensics(self, img_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 1. Edge Density
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # 2. ORB Keypoints
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, _ = orb.detectAndCompute(gray, None)
        num_keypoints = len(keypoints) if keypoints else 0

        # 3. Sharpness (Laplacian Variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(100.0, lap_var / 5.0)

        # 4. Symmetry (horizontal flip comparison)
        flipped = cv2.flip(gray, 1)
        diff = cv2.absdiff(gray, flipped)
        symmetry = 100.0 - float(np.mean(diff) / 2.55)

        # 5. Color Spread
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        color_spread = float(np.std(h_hist))

        # 6. Contour Analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
        contour_regularity = float(np.std(areas)) if len(areas) > 1 else 0.0

        # ── Continuous Composite Score (weighted blend) ──────────────────
        # Each metric contributes a continuous 0–100 sub-score
        # then they are blended with weights

        # Sharpness sub-score: 0-100 linearly
        s_sharp = min(100.0, max(0.0, sharpness))

        # Symmetry sub-score: already 0-100
        s_sym = min(100.0, max(0.0, symmetry))

        # Edge density sub-score: peak at 0.08-0.20, drops off outside
        if 0.05 <= edge_density <= 0.25:
            s_edge = 90.0
        elif edge_density < 0.05:
            s_edge = max(20.0, edge_density / 0.05 * 90.0)
        else:
            s_edge = max(20.0, 90.0 - (edge_density - 0.25) * 200.0)

        # Keypoint sub-score: peak at 50-250
        if 50 <= num_keypoints <= 250:
            s_kp = 90.0
        elif num_keypoints < 50:
            s_kp = max(30.0, num_keypoints / 50.0 * 90.0)
        else:
            s_kp = max(40.0, 90.0 - (num_keypoints - 250) * 0.2)

        # Contour regularity sub-score: lower = more regular = better
        if contour_regularity < 2000:
            s_contour = 85.0
        elif contour_regularity < 8000:
            s_contour = 70.0
        else:
            s_contour = max(30.0, 85.0 - (contour_regularity - 2000) / 500.0)

        # Color consistency sub-score: moderate spread is expected
        if 500 < color_spread < 8000:
            s_color = 85.0
        else:
            s_color = 50.0

        # Weighted blend
        composite = (
            s_sharp   * 0.25 +
            s_sym     * 0.20 +
            s_edge    * 0.20 +
            s_kp      * 0.15 +
            s_contour * 0.10 +
            s_color   * 0.10
        )
        composite = round(max(10.0, min(99.0, composite)), 1)

        # Visualization images
        kp_img  = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

        return {
            "edge_density":       round(edge_density, 4),
            "keypoints_count":    num_keypoints,
            "sharpness_score":    round(sharpness, 1),
            "symmetry_score":     round(symmetry, 1),
            "color_spread":       round(color_spread, 1),
            "num_contours":       num_contours,
            "contour_regularity": round(contour_regularity, 1),
            "structural_score":   composite,
            "kp_image_rgb":       cv2.cvtColor(kp_img,  cv2.COLOR_BGR2RGB),
            "heatmap_rgb":        cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Gemini 2.0 Flash Vision AI
    # ══════════════════════════════════════════════════════════════════════
    def _run_gemini_vision(self, image_pil: Image.Image, target_brand: str) -> dict | None:
        if not self.gemini_available or not self._client:
            return None
        try:
            from google.genai import types

            buf = io.BytesIO()
            image_pil.save(buf, format="PNG")
            image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

            prompt = f"""You are a forensic brand-protection specialist.
Analyze this logo image. Target brand: "{target_brand}".

RULES (apply in order):
1. BRAND IDENTITY: What brand does this logo actually represent?
   If it belongs to a DIFFERENT brand → verdict="COUNTERFEIT", score=5, threat="CRITICAL".
2. VISUAL AUTHENTICITY: Check typography, colors, geometry, resolution, trademark symbols.
   If the logo looks like a low-quality copy, knock-off, or distorted version → verdict="COUNTERFEIT".
3. If logo matches the target brand with professional quality → verdict="AUTHENTIC".

Return ONLY valid JSON:
{{
  "detected_brand": "Nike",
  "matches_target": true,
  "authenticity_score": 95.0,
  "verdict": "AUTHENTIC",
  "threat_level": "LOW",
  "forensic_reasons": [
    "Brand identity verified as Nike",
    "Typography consistent with official guidelines"
  ]
}}
"""
            for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
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

        # Stage 1: OpenCV (always runs)
        cv = self._opencv_forensics(img_bgr)

        # Stage 2: Gemini Vision (if available)
        ai = self._run_gemini_vision(image_pil, target_brand)

        ai_used = False
        if ai and isinstance(ai, dict):
            ai_used          = True
            final_score      = float(ai.get("authenticity_score", cv["structural_score"]))
            is_authentic     = ai.get("verdict", "").upper() == "AUTHENTIC"
            threat_level     = ai.get("threat_level", "LOW" if is_authentic else "HIGH")
            detected_brand   = ai.get("detected_brand", "Unknown")
            forensic_reasons = ai.get("forensic_reasons", [])
            forensic_reasons.insert(0, f"Gemini Vision identified brand: **{detected_brand}**")
        else:
            final_score  = cv["structural_score"]
            is_authentic = final_score >= 70.0
            threat_level = "LOW" if is_authentic else ("HIGH" if final_score < 40.0 else "MEDIUM")
            forensic_reasons = [
                f"Sharpness: {cv['sharpness_score']} (Laplacian variance analysis)",
                f"Symmetry: {cv['symmetry_score']}% (horizontal flip comparison)",
                f"Edge density: {cv['edge_density']} (Canny edge detection)",
                f"ORB keypoints: {cv['keypoints_count']} feature vectors extracted",
                f"Contours: {cv['num_contours']} shapes detected",
                f"Contour regularity σ: {cv['contour_regularity']}",
            ]

        return {
            "target_brand":       target_brand,
            "authenticity_score": round(final_score, 1),
            "is_authentic":       is_authentic,
            "verdict_label":      "AUTHENTIC" if is_authentic else "COUNTERFEIT / ALTERED",
            "threat_level":       threat_level,
            "edge_density":       cv["edge_density"],
            "keypoints_count":    cv["keypoints_count"],
            "sharpness_score":    cv["sharpness_score"],
            "symmetry_score":     cv["symmetry_score"],
            "keypoint_image":     Image.fromarray(cv["kp_image_rgb"]),
            "heatmap_image":      Image.fromarray(cv["heatmap_rgb"]),
            "forensic_reasons":   forensic_reasons,
            "ai_used":            ai_used,
        }
