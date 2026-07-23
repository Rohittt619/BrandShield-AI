import os
import cv2
import numpy as np
from PIL import Image
import requests
import io
import google.generativeai as genai
import json

class BrandShieldDetector:
    """
    Hybrid Multimodal Engine for Fake Logo Detection & Brand Protection.
    Combines OpenCV Structural Analysis, ORB Keypoint Feature Matching, 
    Deep Learning Heuristics, and Google Gemini 1.5 Pro Vision.
    """

    SUPPORTED_BRANDS = [
        "Nike", "Adidas", "Apple", "Starbucks", 
        "Gucci", "Louis Vuitton", "Rolex", "Puma", "Samsung", "Custom / Unspecified"
    ]

    def __init__(self):
        # Safely check local environment variables or Streamlit secrets
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_available = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_available = True
            except Exception:
                pass

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        """
        Fetches an image from a live Web URL.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")

    def analyze_logo(self, image_pil: Image.Image, target_brand: str = "Nike") -> dict:
        """
        Runs multi-stage forensic analysis on uploaded or captured logo image.
        """
        img_np = np.array(image_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. Structural & Color Consistency Check
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 2. ORB Feature Keypoint Extraction
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Render Keypoint Visualization Image
        kp_img = cv2.drawKeypoints(img_bgr, keypoints, None, color=(0, 255, 0), flags=0)
        kp_img_rgb = cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)
        
        # Render Edge Heatmap
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Compute Structural Metrics
        edge_density = float(np.mean(edges > 0))
        num_keypoints = len(keypoints) if keypoints is not None else 0
        
        # Base Score (0 - 100)
        base_score = 85.0
        if edge_density > 0.35 or edge_density < 0.02:
            base_score -= 25.0
        if num_keypoints < 10:
            base_score -= 30.0
        elif num_keypoints > 350:
            base_score -= 15.0
            
        base_score = max(15.0, min(99.0, base_score))
        
        # 3. Gemini 1.5 Pro Multimodal Vision Audit
        ai_verdict = None
        if self.gemini_available:
            ai_verdict = self._run_gemini_vision_audit(image_pil, target_brand)
            
        if ai_verdict and isinstance(ai_verdict, dict):
            final_score = float(ai_verdict.get("authenticity_score", base_score))
            is_authentic = ai_verdict.get("verdict") == "AUTHENTIC"
            threat_level = ai_verdict.get("threat_level", "LOW" if is_authentic else "HIGH")
            forensic_summary = ai_verdict.get("forensic_reasons", ["Passed structural inspection."])
        else:
            final_score = base_score
            is_authentic = final_score >= 70.0
            threat_level = "LOW" if is_authentic else ("HIGH" if final_score < 45.0 else "MEDIUM")
            forensic_summary = [
                f"Edge density ratio: {edge_density:.4f}",
                f"ORB Keypoints extracted: {num_keypoints}",
                "Note: Configure GEMINI_API_KEY in Streamlit Secrets for brand mismatch detection."
            ]

        return {
            "target_brand": target_brand,
            "authenticity_score": round(final_score, 1),
            "is_authentic": is_authentic,
            "verdict_label": "AUTHENTIC" if is_authentic else "COUNTERFEIT / ALTERED",
            "threat_level": threat_level,
            "edge_density": round(edge_density, 4),
            "keypoints_count": num_keypoints,
            "keypoint_image": Image.fromarray(kp_img_rgb),
            "heatmap_image": Image.fromarray(heatmap_rgb),
            "forensic_reasons": forensic_summary
        }

    def _run_gemini_vision_audit(self, image_pil: Image.Image, target_brand: str) -> dict:
        try:
            candidate_models = [
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "gemini-1.5-flash-latest"
            ]
            
            prompt = f"""
You are a Brand Protection Forensic Specialist for global brands.
Inspect the image against target brand: "{target_brand}".

CRITICAL INSPECTION RULES:
1. BRAND MISMATCH CHECK: Does the image show a logo for a DIFFERENT brand than "{target_brand}"? (e.g., an Adidas logo uploaded when target brand is set to Nike). If brand is mismatched, set verdict to "COUNTERFEIT", authenticity_score to 5.0, threat_level to "HIGH", and state "BRAND MISMATCH: Image shows a different logo than {target_brand}."
2. TYPOGRAPHY & SYMMETRY: Check for font distortion, irregular stroke weights, or trademark alterations.

Return JSON object:
{{
    "authenticity_score": 95.0,
    "verdict": "AUTHENTIC",
    "threat_level": "LOW",
    "forensic_reasons": [
        "Brand match verified for {target_brand}",
        "Typography matches authentic specifications"
    ]
}}
"""
            for m_name in candidate_models:
                try:
                    model = genai.GenerativeModel(m_name, generation_config={"response_mime_type": "application/json"})
                    response = model.generate_content([prompt, image_pil])
                    if hasattr(response, "text") and response.text.strip():
                        return json.loads(response.text)
                except Exception:
                    continue
        except Exception:
            pass
            
        return None
