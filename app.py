import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
DB_DIR   = ROOT_DIR / "database"

for p in [str(ROOT_DIR), str(SRC_DIR), str(DB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from PIL import Image

from detector_engine import BrandShieldDetector
from forensic_report import ForensicReportGenerator
from db import BrandShieldDB

st.set_page_config(
    page_title="BrandShield-AI | Counterfeit Detection & Brand Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Read Gemini API key from ALL possible sources
# ──────────────────────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """Try every possible source for the Gemini API key."""
    key = None

    # 1. Environment variable (local dev / .env)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    # 2. Streamlit secrets (Streamlit Cloud)
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass

    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Render results
# ──────────────────────────────────────────────────────────────────────────────
def render_results():
    results  = st.session_state.get("bs_results")
    orig_img = st.session_state.get("bs_image")
    if not results or orig_img is None:
        return

    brand      = results["target_brand"]
    score      = results["authenticity_score"]
    verdict    = results["verdict_label"]
    threat     = results["threat_level"]
    ai_used    = results.get("ai_used", False)
    is_auth    = results["is_authentic"]

    st.divider()

    # Source indicator
    if ai_used:
        st.info("🤖 **Analysis powered by Gemini 2.0 Flash Vision AI**")
    else:
        st.caption("🔬 Structural scan only — Gemini Vision AI not available")

    # Verdict banner
    if is_auth is True:
        st.success(f"✅  **AUTHENTIC** — matches **{brand}** | Score: **{score}%** | Threat: **{threat}**")
    elif is_auth is False:
        st.error(f"❌  **COUNTERFEIT / ALTERED** — flagged for **{brand}** | Score: **{score}%** | Threat: **{threat}**")
    else:
        st.warning(f"🔬  **STRUCTURAL SCAN** — Image quality metrics only. Brand verification requires Gemini Vision AI.")

    # Metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        if ai_used:
            st.metric("Authenticity", f"{score}%")
            st.progress(int(min(max(score, 0), 100)))
        else:
            st.metric("Authenticity", "N/A")
    c2.metric("Verdict",       verdict)
    c3.metric("Threat Level",  threat)
    c4.metric("Sharpness",     f"{results.get('sharpness_score', 'N/A')}")
    c5.metric("Symmetry",      f"{results.get('symmetry_score', 'N/A')}%")
    c6.metric("ORB Keypoints", str(results["keypoints_count"]))

    st.divider()

    # Visual forensic studio
    st.subheader("🔬 Visual Forensic Inspection Studio")
    v1, v2, v3 = st.columns(3)
    v1.markdown("**1. Input Logo**");            v1.image(orig_img,                  use_container_width=True)
    v2.markdown("**2. ORB Feature Keypoints**"); v2.image(results["keypoint_image"], use_container_width=True)
    v3.markdown("**3. Edge Heatmap Overlay**");  v3.image(results["heatmap_image"],  use_container_width=True)

    st.divider()

    # Forensic findings
    with st.expander("📋 Detailed Forensic Findings", expanded=True):
        st.markdown("#### Key Forensic Observations:")
        for r in results["forensic_reasons"]:
            st.markdown(f"- {r}")

    st.divider()

    # PDF download
    pdf_bytes = ForensicReportGenerator.generate_pdf_bytes(results)
    st.download_button(
        label     = "📥 Download Forensic Verification Certificate (PDF)",
        data      = pdf_bytes,
        file_name = f"BrandShield_{brand}.pdf",
        mime      = "application/pdf",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────
def do_analysis(image: Image.Image, brand: str, source: str,
                detector: BrandShieldDetector, db: BrandShieldDB):
    results = detector.analyze_logo(image, brand)
    db.save_inspection(
        brand           = brand,
        verdict         = results["verdict_label"],
        score           = results["authenticity_score"],
        threat_level    = results["threat_level"],
        edge_density    = results["edge_density"],
        keypoints_count = results["keypoints_count"],
        source_type     = source,
    )
    st.session_state["bs_results"] = results
    st.session_state["bs_image"]   = image


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.session_state.setdefault("bs_results",    None)
    st.session_state.setdefault("bs_image",      None)
    st.session_state.setdefault("bs_active_tab", None)

    db = BrandShieldDB()

    # ── Read API key in app.py (where st.secrets is guaranteed to work) ───
    api_key = get_api_key()
    detector = BrandShieldDetector(api_key=api_key)

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.title("🛡️ BrandShield-AI")
    st.sidebar.caption("Enterprise Counterfeit Logo & Brand Protection System")

    if detector.gemini_available:
        st.sidebar.success("🟢 Gemini 2.0 Flash Vision AI Active")
    else:
        st.sidebar.warning("🟡 OpenCV Structural Analysis Mode")
        # Debug: show what's happening so we can diagnose
        if api_key:
            st.sidebar.caption(f"API key found ({len(api_key)} chars) but Gemini init failed")
            if detector._init_error:
                st.sidebar.caption(f"Error: {detector._init_error}")
        else:
            st.sidebar.caption("No API key found in env vars or st.secrets")
            try:
                avail_keys = list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else []
                if avail_keys:
                    st.sidebar.caption(f"Secret keys present: {avail_keys}")
                else:
                    st.sidebar.caption("st.secrets is empty")
            except Exception as e:
                st.sidebar.caption(f"st.secrets error: {e}")

    nav = st.sidebar.radio("Navigation", [
        "🛡️ File Upload Inspection",
        "📷 Live Webcam Scanner",
        "🌐 Web URL Scanner",
        "📊 SQLite Audit Logs",
        "🏗️ System Architecture",
    ])

    # Clear results on tab switch
    if st.session_state["bs_active_tab"] != nav:
        st.session_state["bs_results"]    = None
        st.session_state["bs_image"]      = None
        st.session_state["bs_active_tab"] = nav

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Audit Logs
    # ═══════════════════════════════════════════════════════════════════════
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
        rows = db.fetch_inspections(limit=50)
        if rows:
            st.dataframe(
                [{"ID": r[0], "Brand": r[1], "Verdict": r[2], "Score": f"{r[3]}%",
                  "Threat": r[4], "EdgeDensity": r[5], "Keypoints": r[6],
                  "Source": r[7], "Timestamp": r[8]} for r in rows],
                use_container_width=True,
            )
        else:
            st.info("No logs yet — run a scan first.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Architecture
    # ═══════════════════════════════════════════════════════════════════════
    if nav == "🏗️ System Architecture":
        st.title("🏗️ BrandShield-AI — System Architecture")
        st.markdown("""
        ### Multi-Stage Forensic Inspection Pipeline
        1. **Multi-Input Ingestion** — File Upload / Live Webcam / Web URL
        2. **OpenCV Structural Analysis** — Canny Edge Density + 500 ORB Keypoints + Sharpness + Symmetry
        3. **Google Gemini 2.0 Flash Vision AI** — Brand identification, mismatch detection, typography & geometry audit
        4. **SQLite Persistence** — Every scan saved to `database/brandshield.db`
        5. **ReportLab PDF Exporter** — Downloadable Forensic Verification Certificate
        """)
        return

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Webcam Scanner
    # ═══════════════════════════════════════════════════════════════════════
    if nav == "📷 Live Webcam Scanner":
        st.title("📷 Live Webcam Logo Scanner")
        st.divider()
        brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="cam_brand")
        photo = st.camera_input("📸 Take a photo of the product logo")
        if photo:
            image = Image.open(photo)
            with st.spinner(f"Inspecting for **{brand}**…"):
                do_analysis(image, brand, "Webcam Scan", detector, db)
            st.rerun()
        render_results()
        return

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Web URL Scanner
    # ═══════════════════════════════════════════════════════════════════════
    if nav == "🌐 Web URL Scanner":
        st.title("🌐 Web URL Logo Scanner")
        st.divider()
        brand     = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="url_brand")
        url_input = st.text_input("Paste image URL (PNG / JPG)",
                                  placeholder="https://example.com/logo.png")
        if st.button("🌐 Scan Image from URL", key="url_btn"):
            if not url_input.strip():
                st.error("Please paste a valid image URL first.")
            else:
                with st.spinner("Fetching image…"):
                    try:
                        image = detector.load_image_from_url(url_input)
                    except Exception as e:
                        st.error(f"Could not fetch image: {e}")
                        st.stop()
                with st.spinner(f"Inspecting for **{brand}**…"):
                    do_analysis(image, brand, "Web URL Scan", detector, db)
                st.rerun()
        render_results()
        return

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: File Upload (DEFAULT)
    # ═══════════════════════════════════════════════════════════════════════
    st.title("🛡️ BrandShield-AI")
    st.subheader("AI-Powered Counterfeit Logo Detection & Forensic Verification")
    st.divider()

    brand         = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="file_brand")
    uploaded_file = st.file_uploader(
        "Choose Logo Image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        key="file_uploader",
    )

    if st.button("🔍 Run Forensic Inspection", key="inspect_btn"):
        if uploaded_file is None:
            st.error("⚠️ Upload a logo image first, then click the button.")
        else:
            with st.spinner(f"Inspecting logo for **{brand}**…"):
                image = Image.open(uploaded_file)
                do_analysis(image, brand, "File Upload", detector, db)
            st.rerun()

    render_results()


if __name__ == "__main__":
    main()
