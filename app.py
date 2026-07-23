import os
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


def get_api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
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


def render_results():
    results  = st.session_state.get("bs_results")
    orig_img = st.session_state.get("bs_image")
    if not results or orig_img is None:
        return

    brand   = results["target_brand"]
    score   = results["authenticity_score"]
    verdict = results["verdict_label"]
    threat  = results["threat_level"]
    ai_used = results.get("ai_used", False)
    is_auth = results["is_authentic"]

    st.divider()

    if ai_used:
        st.info("🤖 **Analysis powered by Gemini 2.0 Flash Vision AI**")
    else:
        st.caption("🔬 Structural scan only — Gemini Vision AI not available")

    if is_auth is True:
        st.success(f"✅  **AUTHENTIC** — matches **{brand}** | Score: **{score}%** | Threat: **{threat}**")
    elif is_auth is False:
        st.error(f"❌  **COUNTERFEIT / ALTERED** — flagged for **{brand}** | Score: **{score}%** | Threat: **{threat}**")
    else:
        st.warning("🔬  **STRUCTURAL SCAN** — Image quality metrics only. Brand verification requires Gemini Vision AI.")

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
    st.subheader("🔬 Visual Forensic Inspection Studio")
    v1, v2, v3 = st.columns(3)
    v1.markdown("**1. Input Logo**");            v1.image(orig_img,                  width="stretch")
    v2.markdown("**2. ORB Feature Keypoints**"); v2.image(results["keypoint_image"], width="stretch")
    v3.markdown("**3. Edge Heatmap Overlay**");  v3.image(results["heatmap_image"],  width="stretch")

    st.divider()
    with st.expander("📋 Detailed Forensic Findings", expanded=True):
        st.markdown("#### Key Forensic Observations:")
        for r in results["forensic_reasons"]:
            st.markdown(f"- {r}")

    st.divider()
    pdf_bytes = ForensicReportGenerator.generate_pdf_bytes(results)
    st.download_button(
        label     = "📥 Download Forensic Verification Certificate (PDF)",
        data      = pdf_bytes,
        file_name = f"BrandShield_{brand}.pdf",
        mime      = "application/pdf",
    )


def do_analysis(image, brand, source, detector, db):
    results = detector.analyze_logo(image, brand)
    db.save_inspection(
        brand=brand, verdict=results["verdict_label"],
        score=results["authenticity_score"], threat_level=results["threat_level"],
        edge_density=results["edge_density"], keypoints_count=results["keypoints_count"],
        source_type=source,
    )
    st.session_state["bs_results"] = results
    st.session_state["bs_image"]   = image


def main():
    st.session_state.setdefault("bs_results",    None)
    st.session_state.setdefault("bs_image",      None)
    st.session_state.setdefault("bs_active_tab", None)

    db       = BrandShieldDB()
    api_key  = get_api_key()
    detector = BrandShieldDetector(api_key=api_key)

    st.sidebar.title("🛡️ BrandShield-AI")
    st.sidebar.caption("Enterprise Counterfeit Logo & Brand Protection System")
    if detector.gemini_available:
        st.sidebar.success("🟢 Gemini 2.0 Flash Vision AI Active")
    else:
        st.sidebar.warning("🟡 OpenCV Structural Analysis Mode")
        if api_key:
            st.sidebar.caption(f"Key found ({len(api_key)} chars, starts with '{api_key[:4]}...')")
            if detector._init_error:
                st.sidebar.error(f"Init error: {detector._init_error}")
        else:
            st.sidebar.caption("No API key found")

    # Test API Key button — shows exact success/failure
    with st.sidebar.expander("🔧 API Key Diagnostics"):
        if api_key:
            st.code(f"Key: {api_key[:6]}...{api_key[-4:]}\nLength: {len(api_key)} chars\nClient OK: {detector.gemini_available}")
        else:
            st.write("No key detected in `st.secrets` or env vars")
        if st.button("🧪 Test API Key Now", key="test_api"):
            with st.spinner("Testing connection to Gemini..."):
                result = detector.test_api_key()
            st.write(result)

    nav = st.sidebar.radio("Navigation", [
        "🛡️ File Upload Inspection",
        "📷 Live Webcam Scanner",
        "🌐 Web URL Scanner",
        "📊 SQLite Audit Logs",
        "🏗️ System Architecture",
    ])

    if st.session_state["bs_active_tab"] != nav:
        st.session_state["bs_results"]    = None
        st.session_state["bs_image"]      = None
        st.session_state["bs_active_tab"] = nav

    if nav == "📊 SQLite Audit Logs":
        st.title("📊 SQLite Audit Logs & Brand Analytics")
        st.divider()
        stats = db.fetch_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspections",    stats["total_inspections"])
        c2.metric("Counterfeits Blocked", stats["counterfeits_blocked"])
        c3.metric("Authentics Verified",  stats["authentics_verified"])
        st.divider()
        rows = db.fetch_inspections(limit=50)
        if rows:
            st.dataframe(
                [{"ID": r[0], "Brand": r[1], "Verdict": r[2], "Score": f"{r[3]}%",
                  "Threat": r[4], "EdgeDensity": r[5], "Keypoints": r[6],
                  "Source": r[7], "Timestamp": r[8]} for r in rows],
                use_container_width=True)  # dataframe supports use_container_width
        else:
            st.info("No logs yet.")
        return

    if nav == "🏗️ System Architecture":
        st.title("🏗️ System Architecture")
        st.markdown("""
        ### Multi-Stage Pipeline
        1. **Multi-Input** — File Upload / Webcam / Web URL
        2. **OpenCV** — Canny Edges + ORB Keypoints + Sharpness + Symmetry
        3. **Gemini 2.0 Flash** — Brand ID, mismatch detection, typography audit
        4. **SQLite** — Persistent audit trail
        5. **ReportLab** — PDF verification certificates
        """)
        return

    if nav == "📷 Live Webcam Scanner":
        st.title("📷 Live Webcam Logo Scanner")
        st.divider()
        brand = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="cam_brand")
        photo = st.camera_input("📸 Take a photo")
        if photo:
            image = Image.open(photo)
            with st.spinner(f"Inspecting for **{brand}**…"):
                do_analysis(image, brand, "Webcam Scan", detector, db)
            st.rerun()
        render_results()
        return

    if nav == "🌐 Web URL Scanner":
        st.title("🌐 Web URL Logo Scanner")
        st.divider()
        brand     = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="url_brand")
        url_input = st.text_input("Paste image URL", placeholder="https://example.com/logo.png")
        if st.button("🌐 Scan Image from URL", key="url_btn"):
            if not url_input.strip():
                st.error("Paste a valid URL first.")
            else:
                with st.spinner("Fetching image…"):
                    try:
                        image = detector.load_image_from_url(url_input)
                    except Exception as e:
                        st.error(f"Could not fetch: {e}")
                        st.stop()
                with st.spinner(f"Inspecting for **{brand}**…"):
                    do_analysis(image, brand, "Web URL Scan", detector, db)
                st.rerun()
        render_results()
        return

    st.title("🛡️ BrandShield-AI")
    st.subheader("AI-Powered Counterfeit Logo Detection & Forensic Verification")
    st.divider()
    brand         = st.selectbox("Select Target Brand", detector.SUPPORTED_BRANDS, key="file_brand")
    uploaded_file = st.file_uploader("Choose Logo Image (PNG / JPG / JPEG)",
                                     type=["png", "jpg", "jpeg"], key="file_uploader")
    if st.button("🔍 Run Forensic Inspection", key="inspect_btn"):
        if uploaded_file is None:
            st.error("Upload a logo image first.")
        else:
            with st.spinner(f"Inspecting for **{brand}**…"):
                image = Image.open(uploaded_file)
                do_analysis(image, brand, "File Upload", detector, db)
            st.rerun()
    render_results()


if __name__ == "__main__":
    main()
