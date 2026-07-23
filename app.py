import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
DB_DIR   = ROOT_DIR / "database"

for p in [ROOT_DIR, SRC_DIR, DB_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
from PIL import Image

from brandshield_ai.detector_engine import BrandShieldDetector
from brandshield_ai.forensic_report import ForensicReportGenerator
from database.db import BrandShieldDB

st.set_page_config(
    page_title="BrandShield-AI | Counterfeit Detection & Brand Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# Helper — run analysis, save to DB, persist in session_state
# ──────────────────────────────────────────────────────────────
def run_analysis(image: Image.Image, target_brand: str, source_type: str,
                 detector: BrandShieldDetector, db: BrandShieldDB):
    results = detector.analyze_logo(image, target_brand)
    db.save_inspection(
        brand=target_brand,
        verdict=results["verdict_label"],
        score=results["authenticity_score"],
        threat_level=results["threat_level"],
        edge_density=results["edge_density"],
        keypoints_count=results["keypoints_count"],
        source_type=source_type,
    )
    st.session_state["results"]        = results
    st.session_state["original_image"] = image


# ──────────────────────────────────────────────────────────────
# Helper — render persisted results
# ──────────────────────────────────────────────────────────────
def render_results():
    results  = st.session_state.get("results")
    orig_img = st.session_state.get("original_image")
    if not results or orig_img is None:
        return

    target_brand = results["target_brand"]
    score        = results["authenticity_score"]
    verdict      = results["verdict_label"]
    threat       = results["threat_level"]

    st.divider()

    # Banner
    if results["is_authentic"]:
        st.success(
            f"✅  **AUTHENTIC** — Logo matches **{target_brand}** "
            f"| Score: **{score}%** | Threat: **{threat}**"
        )
    else:
        st.error(
            f"❌  **COUNTERFEIT / ALTERED** — High-risk logo for **{target_brand}** "
            f"| Score: **{score}%** | Threat: **{threat}**"
        )

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Authenticity Score", f"{score}%")
        st.progress(int(min(score, 100)))
    m2.metric("Verdict",       verdict)
    m3.metric("Threat Level",  threat)
    m4.metric("Edge Density",  str(results["edge_density"]))
    m5.metric("ORB Keypoints", str(results["keypoints_count"]))

    st.divider()

    # Visual studio
    st.subheader("🔬 Visual Forensic Inspection Studio")
    v1, v2, v3 = st.columns(3)
    v1.markdown("**1. Input Logo**")
    v1.image(orig_img, width="stretch")
    v2.markdown("**2. ORB Feature Keypoints**")
    v2.image(results["keypoint_image"], width="stretch")
    v3.markdown("**3. Edge Heatmap Overlay**")
    v3.image(results["heatmap_image"], width="stretch")

    st.divider()

    # Forensic findings
    with st.expander("📋 Detailed Forensic Findings", expanded=True):
        st.markdown("#### Key Forensic Observations:")
        for reason in results["forensic_reasons"]:
            st.markdown(f"- {reason}")

    st.divider()

    # PDF download
    pdf_bytes = ForensicReportGenerator.generate_pdf_bytes(results)
    st.download_button(
        label="📥 Download Forensic Verification Certificate (PDF)",
        data=pdf_bytes,
        file_name=f"BrandShield_Verification_{target_brand}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # Initialise session state
    for key in ("results", "original_image", "active_tab"):
        if key not in st.session_state:
            st.session_state[key] = None

    db       = BrandShieldDB()
    detector = BrandShieldDetector()

    # Sidebar
    st.sidebar.title("🛡️ BrandShield-AI")
    st.sidebar.caption("Enterprise Counterfeit Logo & Brand Protection System")

    if detector.gemini_available:
        st.sidebar.success("🟢 Gemini Vision API Active")
    else:
        st.sidebar.warning("🟡 Gemini API Key Missing — OpenCV Structural Mode")

    nav = st.sidebar.radio("Navigation", [
        "🛡️ File Upload Inspection",
        "📷 Live Webcam Scanner",
        "🌐 Web URL Scanner",
        "📊 SQLite Audit Logs",
        "🏗️ System Architecture",
    ])

    # Clear results when user switches tabs
    if st.session_state["active_tab"] != nav:
        st.session_state["results"]        = None
        st.session_state["original_image"] = None
        st.session_state["active_tab"]     = nav

    # ═══════════════════════════════════════
    # TAB: Audit Logs
    # ═══════════════════════════════════════
    if nav == "📊 SQLite Audit Logs":
        st.title("📊 SQLite Audit Logs & Brand Analytics")
        st.divider()

        stats = db.fetch_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspections",    stats["total_inspections"])
        c2.metric("Counterfeits Blocked", stats["counterfeits_blocked"])
        c3.metric("Authentics Verified",  stats["authentics_verified"])

        st.divider()
        st.subheader("📜 Recent Inspection History")
        history = db.fetch_inspections(limit=50)
        if history:
            rows = [
                {
                    "ID": r[0], "Brand": r[1], "Verdict": r[2],
                    "Score": f"{r[3]}%", "Threat": r[4],
                    "Edge Density": r[5], "Keypoints": r[6],
                    "Source": r[7], "Timestamp": r[8],
                }
                for r in history
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No inspection logs yet — run a scan first.")
        return

    # ═══════════════════════════════════════
    # TAB: Architecture
    # ═══════════════════════════════════════
    if nav == "🏗️ System Architecture":
        st.title("🏗️ BrandShield-AI — System Architecture")
        st.markdown("""
        ### Multi-Stage Forensic Inspection Pipeline
        1. **Multi-Input Ingestion** — File Upload / Live Webcam / Web URL
        2. **OpenCV Structural Analysis** — Canny Edge Density + 500 ORB Keypoints
        3. **Google Gemini Vision** — Brand Mismatch detection, Typography & Symmetry audit
        4. **SQLite Persistence** — Every scan saved to `database/brandshield.db`
        5. **ReportLab PDF Exporter** — Downloadable Forensic Verification Certificate
        """)
        return

    # ═══════════════════════════════════════
    # TAB: Webcam Scanner
    # ═══════════════════════════════════════
    if nav == "📷 Live Webcam Scanner":
        st.title("📷 Live Webcam Logo Scanner")
        st.divider()
        target_brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="cam_brand")
        camera_photo = st.camera_input("📸 Take a photo of the product logo")

        if camera_photo:
            image = Image.open(camera_photo)
            with st.spinner(f"Running forensic inspection for **{target_brand}**…"):
                run_analysis(image, target_brand, "Webcam Scan", detector, db)

        render_results()
        return

    # ═══════════════════════════════════════
    # TAB: Web URL Scanner
    # ═══════════════════════════════════════
    if nav == "🌐 Web URL Scanner":
        st.title("🌐 Web URL Logo Scanner")
        st.divider()
        target_brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="url_brand")
        url_input    = st.text_input(
            "Paste a direct image URL (PNG / JPG)",
            placeholder="https://example.com/nike-logo.png",
        )
        scan_btn = st.button("🌐 Scan Image from URL", use_container_width=True)

        if scan_btn:
            if not url_input.strip():
                st.error("Please paste a valid image URL first.")
            else:
                with st.spinner("Fetching image from URL…"):
                    try:
                        image = detector.load_image_from_url(url_input)
                    except Exception as e:
                        st.error(f"Could not fetch image: {e}")
                        st.stop()
                with st.spinner(f"Running forensic inspection for **{target_brand}**…"):
                    run_analysis(image, target_brand, "Web URL Scan", detector, db)

        render_results()
        return

    # ═══════════════════════════════════════
    # TAB: File Upload (DEFAULT)
    # ═══════════════════════════════════════
    st.title("🛡️ BrandShield-AI")
    st.subheader("AI-Powered Counterfeit Logo Detection & Forensic Verification")
    st.divider()

    target_brand  = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="file_brand")
    uploaded_file = st.file_uploader(
        "Choose Logo Image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
    )
    analyze_btn = st.button("🔍 Run Forensic Inspection", use_container_width=True)

    if analyze_btn:
        if uploaded_file is None:
            st.error("⚠️ Please upload a logo image first, then click the button.")
        else:
            # Immediate feedback so user knows click was registered
            status = st.info("⏳ Starting forensic inspection — please wait…")
            with st.spinner(f"Running OpenCV + Gemini Vision inspection for **{target_brand}**…"):
                image = Image.open(uploaded_file)
                run_analysis(image, target_brand, "File Upload", detector, db)
            status.empty()   # remove the info banner once done

    # Always render from session_state — survives all reruns
    render_results()


if __name__ == "__main__":
    main()
