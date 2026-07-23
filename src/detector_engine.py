import os
import io
import cv2
import json
import numpy as np
from PIL import Image
import requests

class BrandShieldDetector:
    """
    Hybrid Multimodal Engine for Fake Logo Detection & Brand Protection.
    Combines OpenCV Structural Analysis (Canny + ORB) and Google Gemini Vision.
    """

    SUPPORTED_BRANDS = [
        "Nike", "Adidas", "Apple", "Starbucks",
        "Gucci", "Louis Vuitton", "Rolex", "Puma", "Samsung", "Custom / Unspecified"
    ]

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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

    def analyze_logo(self, image_pil: Image.Image, target_brand: str = "Nike") -> dict:
        img_np  = np.array(image_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, _ = orb.detectAndCompute(gray, None)

        kp_img  = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

        kp_img_rgb  = cv2.cvtColor(kp_img,  cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        edge_density  = float(np.mean(edges > 0))
        num_keypoints = len(keypoints) if keypoints else 0

        base_score = 85.0
        if edge_density > 0.35 or edge_density < 0.02:
            base_score -= 25.0
        if num_keypoints < 10:
            base_score -= 30.0
        elif num_keypoints > 350:
            base_score -= 15.0
        base_score = max(15.0, min(99.0, base_score))

        ai_verdict = None
        if self.gemini_available:
            ai_verdict = self._run_gemini_vision_audit(image_pil, target_brand)

        if ai_verdict and isinstance(ai_verdict, dict):
            final_score      = float(ai_verdict.get("authenticity_score", base_score))
            is_authentic     = ai_verdict.get("verdict") == "AUTHENTIC"
            threat_level     = ai_verdict.get("threat_level", "LOW" if is_authentic else "HIGH")
            forensic_summary = ai_verdict.get("forensic_reasons", ["Passed structural inspection."])
        else:
            final_score  = base_score
            is_authentic = final_score >= 70.0
            threat_level = "LOW" if is_authentic else ("HIGH" if final_score < 45.0 else "MEDIUM")
            forensic_summary = [
                f"Edge density ratio: {edge_density:.4f}",
                f"ORB Keypoints extracted: {num_keypoints}",
                "Add GEMINI_API_KEY in Streamlit Secrets for brand-mismatch AI detection.",
            ]

        return {
            "target_brand":       target_brand,
            "authenticity_score": round(final_score, 1),
            "is_authentic":       is_authentic,
            "verdict_label":      "AUTHENTIC" if is_authentic else "COUNTERFEIT / ALTERED",
            "threat_level":       threat_level,
            "edge_density":       round(edge_density, 4),
            "keypoints_count":    num_keypoints,
            "keypoint_image":     Image.fromarray(kp_img_rgb),
            "heatmap_image":      Image.fromarray(heatmap_rgb),
            "forensic_reasons":   forensic_summary,
        }

    def _run_gemini_vision_audit(self, image_pil: Image.Image, target_brand: str):
        try:
            from google.genai import types

            buf = io.BytesIO()
            image_pil.save(buf, format="PNG")
            image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

            prompt = f"""
You are a Brand Protection Forensic Specialist.
Target brand: "{target_brand}".

RULES:
1. BRAND MISMATCH: If the logo belongs to a DIFFERENT brand than "{target_brand}", set verdict="COUNTERFEIT", authenticity_score=5.0, threat_level="HIGH".
2. Check typography, symmetry, color gradients.

Return ONLY valid JSON:
{{
    "authenticity_score": 95.0,
    "verdict": "AUTHENTIC",
    "threat_level": "LOW",
    "forensic_reasons": ["Brand match verified", "Typography correct"]
}}
"""
            for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                try:
                    resp = self._client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image_part],
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    if resp.text and resp.text.strip():
                        return json.loads(resp.text)
                except Exception:
                    continue
        except Exception:
            pass
        return None
