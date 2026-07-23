import sys
import os
from pathlib import Path

# Add project root and src directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DB_DIR = ROOT_DIR / "database"

for p in [ROOT_DIR, SRC_DIR, DB_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
from PIL import Image
import tempfile

from brandshield_ai.detector_engine import BrandShieldDetector
from brandshield_ai.forensic_report import ForensicReportGenerator
from database.db import BrandShieldDB

st.set_page_config(
    page_title="BrandShield-AI | Counterfeit Detection & Brand Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("🛡️ BrandShield-AI")
    st.sidebar.caption("Enterprise Counterfeit Logo & Brand Protection System")
    
    db = BrandShieldDB()
    detector = BrandShieldDetector()
    
    nav = st.sidebar.radio("Navigation", [
        "🛡️ Live Inspection Portal", 
        "🌐 Web URL Scanner",
        "📊 SQLite Audit Logs & Analytics", 
        "🏗️ System Architecture"
    ])

    # 1. SQLite Audit Logs & Analytics
    if nav == "📊 SQLite Audit Logs & Analytics":
        st.title("📊 SQLite Audit Logs & Brand Analytics")
        st.divider()

        stats = db.fetch_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspections Logged", stats["total_inspections"])
        c2.metric("Counterfeits Blocked", stats["counterfeits_blocked"])
        c3.metric("Authentics Verified", stats["authentics_verified"])

        st.divider()
        st.subheader("📜 Recent Inspection History (SQLite Database)")
        history = db.fetch_inspections(limit=50)

        if history:
            formatted_data = []
            for row in history:
                formatted_data.append({
                    "ID": row[0],
                    "Brand": row[1],
                    "Verdict": row[2],
                    "Score (%)": f"{row[3]}%",
                    "Threat Level": row[4],
                    "Edge Density": row[5],
                    "Keypoints": row[6],
                    "Source": row[7],
                    "Timestamp": row[8]
                })
            st.dataframe(formatted_data, use_container_width=True)
        else:
            st.info("No inspection logs recorded in SQLite database yet.")
        return

    # 2. System Architecture
    if nav == "🏗️ System Architecture":
        st.title("🏗️ BrandShield-AI Architecture & Multi-Stage Pipeline")
        st.markdown("""
        ### Multi-Stage Forensic Inspection Engine:
        1. **Dual Ingestion Engine**: Accepts high-res local image uploads (PNG/JPG) OR live web product image URLs.
        2. **OpenCV Structural Analysis**: Calculates Canny Edge Density, Contour Symmetries, and 500 ORB feature keypoint vectors.
        3. **Deep Feature Extraction**: MobileNetV2 Deep Learning transfer heuristics.
        4. **Google Gemini 1.5 Pro Multimodal Vision**: Inspects typography alignment, trademark gradient anomalies, and geometry.
        5. **SQLite Persistence**: Stores all inspection alerts and brand threat trends in `database/brandshield.db`.
        6. **ReportLab Forensic Certification**: Generates downloadable PDF verification certificates for legal compliance teams.
        """)
        return

    # 3. Main Live Inspection Portal (File Upload or Web URL)
    st.title("🛡️ BrandShield-AI")
    st.subheader("AI-Powered Counterfeit Logo Detection & Forensic Verification")
    st.divider()

    image_to_analyze = None
    source_type = "File Upload"

    if nav == "🌐 Web URL Scanner":
        st.markdown("### 🌐 Inspect Image from Web URL")
        url_input = st.text_input("Paste Live Image URL (e.g. e-commerce product image)", placeholder="https://example.com/logo.jpg")
        target_brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="url_brand")
        analyze_btn = st.button("🌐 Scan Image URL", use_container_width=True)
        
        if analyze_btn:
            if not url_input.strip():
                st.error("Please enter a valid Image URL.")
                return
            try:
                with st.spinner("Fetching image from Web URL..."):
                    image_to_analyze = detector.load_image_from_url(url_input)
                    source_type = "Web URL Scan"
            except Exception as e:
                st.error(f"Failed to fetch image from URL: {e}")
                return
    else:
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.markdown("### 📤 Upload Logo File")
            uploaded_file = st.file_uploader("Choose Logo Image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
            target_brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="file_brand")
            analyze_btn = st.button("🔍 Run Forensic Inspection", use_container_width=True)

            if analyze_btn:
                if uploaded_file is None:
                    st.error("Please upload a logo image to inspect.")
                    return
                image_to_analyze = Image.open(uploaded_file)
                source_type = "File Upload"

    if image_to_analyze:
        with st.spinner(f"Running OpenCV Structural Inspection & Querying Gemini Vision for {target_brand}..."):
            results = detector.analyze_logo(image_to_analyze, target_brand)
            
            # Save to SQLite Database
            db.save_inspection(
                brand=target_brand,
                verdict=results["verdict_label"],
                score=results["authenticity_score"],
                threat_level=results["threat_level"],
                edge_density=results["edge_density"],
                keypoints_count=results["keypoints_count"],
                source_type=source_type
            )
            
            st.session_state["results"] = results
            st.session_state["original_image"] = image_to_analyze

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
        st.subheader("🔬 Visual Forensic Inspection Studio")
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.markdown("#### 1. Input Logo")
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
