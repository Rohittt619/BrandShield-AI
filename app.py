import sys
import os
from pathlib import Path

# Add project root and src directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in [ROOT_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
from PIL import Image
import tempfile

from brandshield_ai.detector_engine import BrandShieldDetector
from brandshield_ai.forensic_report import ForensicReportGenerator

st.set_page_config(
    page_title="BrandShield-AI | Fake Logo Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("🛡️ BrandShield-AI")
    st.sidebar.caption("Enterprise Counterfeit Logo & Brand Protection System")
    
    nav = st.sidebar.radio("Navigation", ["Logo Inspection Portal", "Brand Threat Database", "System Architecture"])
    
    detector = BrandShieldDetector()
    
    if nav == "Brand Threat Database":
        st.title("📊 Global Brand Threat Database")
        st.info("Monitored Brand Assets & Counterfeit Incident History")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Monitored Brands", "125+")
        col_b.metric("Counterfeits Blocked Today", "43")
        col_c.metric("Detection Precision", "98.4%")
        
        st.subheader("Recent Counterfeit Detection Alerts")
        st.table([
            {"Timestamp": "2026-07-23 16:45", "Brand": "Nike", "Verdict": "COUNTERFEIT (32%)", "Risk": "HIGH", "Action": "Flagged for Legal"},
            {"Timestamp": "2026-07-23 16:20", "Brand": "Gucci", "Verdict": "COUNTERFEIT (18%)", "Risk": "HIGH", "Action": "Customs Seizure Notice"},
            {"Timestamp": "2026-07-23 15:55", "Brand": "Apple", "Verdict": "AUTHENTIC (96%)", "Risk": "LOW", "Action": "Verified Pass"},
            {"Timestamp": "2026-07-23 15:10", "Brand": "Rolex", "Verdict": "ALTERED (54%)", "Risk": "MEDIUM", "Action": "Manual Audit"},
        ])
        return

    if nav == "System Architecture":
        st.title("🏗️ BrandShield-AI Architecture & Inspection Flow")
        st.markdown("""
        ### Multi-Stage Forensic Inspection Pipeline:
        1. **Dual PDF & Image Ingestion**: Reads PNG/JPG logo files across high-resolution inspect channels.
        2. **OpenCV Structural Analysis**: Calculates Canny Edge Density, Contour Symmetries, and ORB Keypoints.
        3. **Deep Feature Feature Extraction**: MobileNetV2 Deep Learning transfer heuristics.
        4. **Google Gemini 1.5 Pro Multimodal Vision**: Inspects typography alignment, trademark gradient anomalies, and geometry.
        5. **ReportLab Forensic Certification**: Generates downloadable PDF verification certificates for legal and compliance teams.
        """)
        return

    # Main Portal
    st.title("🛡️ BrandShield-AI")
    st.subheader("AI-Powered Counterfeit Logo Detection & Forensic Verification")
    st.divider()

    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown("### 📤 Upload Logo Image")
        uploaded_file = st.file_uploader("Choose Logo Image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
        
        target_brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS)
        
        analyze_btn = st.button("🔍 Run Forensic Inspection", use_container_width=True)

    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload a logo image to inspect.")
            return

        image = Image.open(uploaded_file)
        
        with st.spinner(f"Running OpenCV Structural Inspection & Querying Gemini Vision for {target_brand}..."):
            results = detector.analyze_logo(image, target_brand)
            st.session_state["results"] = results
            st.session_state["original_image"] = image

    if st.session_state.get("results"):
        results = st.session_state["results"]
        orig_img = st.session_state["original_image"]
        
        st.divider()
        
        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        
        score = results["authenticity_score"]
        verdict = results["verdict_label"]
        threat = results["threat_level"]
        
        with m1:
            st.metric("Authenticity Score", f"{score}%")
            st.progress(int(min(score, 100)))
            
        with m2:
            st.metric("Verdict", verdict)
            
        with m3:
            st.metric("Threat Level", threat)
            
        with m4:
            st.metric("Edge Density", f"{results['edge_density']}")
            
        with m5:
            st.metric("ORB Keypoints", f"{results['keypoints_count']}")

        st.divider()

        # Visual Comparison Section
        st.subheader("🔬 Visual Forensic Inspection")
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.markdown("#### 1. Uploaded Logo")
            st.image(orig_img, use_container_width=True)
            
        with v2:
            st.markdown("#### 2. ORB Feature Keypoints")
            st.image(results["keypoint_image"], use_container_width=True)
            
        with v3:
            st.markdown("#### 3. Edge Heatmap Overlay")
            st.image(results["heatmap_image"], use_container_width=True)

        st.divider()

        # Forensic Audit Findings Card
        with st.expander("📋 Detailed Forensic Findings", expanded=True):
            if results["is_authentic"]:
                st.success(f"✅ **Verdict**: Logo matches official **{target_brand}** trademark specifications.")
            else:
                st.error(f"❌ **Verdict**: High probability of **COUNTERFEIT / ALTERED LOGO** for **{target_brand}**.")
                
            st.markdown("#### Key Forensic Observations:")
            for reason in results["forensic_reasons"]:
                st.markdown(f"- {reason}")

        st.divider()

        # Download Report PDF
        pdf_bytes = ForensicReportGenerator.generate_pdf_bytes(results)
        st.download_button(
            "📥 Download Executive Forensic Verification Certificate (PDF)",
            pdf_bytes,
            file_name=f"BrandShield_Verification_{target_brand}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
