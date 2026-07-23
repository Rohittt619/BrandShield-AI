import os
import io
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

class ForensicReportGenerator:
    """
    Generates downloadable PDF Brand Protection Verification Certificates.
    """

    @staticmethod
    def generate_pdf_bytes(audit_results: dict) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Header Title
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0F172A'),
            alignment=1
        )
        
        story.append(Paragraph("🛡️ BrandShield AI — Forensic Verification Certificate", title_style))
        story.append(Spacer(1, 15))
        
        # Status Banner
        verdict = audit_results.get("verdict_label", "AUTHENTIC")
        score = audit_results.get("authenticity_score", 90.0)
        brand = audit_results.get("target_brand", "Brand")
        threat = audit_results.get("threat_level", "LOW")
        
        banner_color = colors.HexColor('#16A34A') if verdict == "AUTHENTIC" else colors.HexColor('#DC2626')
        
        banner_style = ParagraphStyle(
            'BannerStyle',
            parent=styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.white,
            alignment=1
        )
        
        banner_table = Table([[Paragraph(f"VERDICT: {verdict} ({score}% Authenticity)", banner_style)]], colWidths=[520])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), banner_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
        ]))
        
        story.append(banner_table)
        story.append(Spacer(1, 15))
        
        # Details Table
        data = [
            [Paragraph("<b>Target Brand</b>", styles['BodyText']), Paragraph(str(brand), styles['BodyText'])],
            [Paragraph("<b>Authenticity Score</b>", styles['BodyText']), Paragraph(f"{score}%", styles['BodyText'])],
            [Paragraph("<b>Threat Level</b>", styles['BodyText']), Paragraph(str(threat), styles['BodyText'])],
            [Paragraph("<b>Edge Density Ratio</b>", styles['BodyText']), Paragraph(str(audit_results.get('edge_density', 'N/A')), styles['BodyText'])],
            [Paragraph("<b>Keypoint Count</b>", styles['BodyText']), Paragraph(str(audit_results.get('keypoints_count', 'N/A')), styles['BodyText'])],
        ]
        
        t = Table(data, colWidths=[200, 320])
        t.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))
        
        # Forensic Reasons Section
        story.append(Paragraph("<b>Forensic Analysis Findings</b>", styles['Heading3']))
        story.append(Spacer(1, 5))
        
        for reason in audit_results.get("forensic_reasons", []):
            story.append(Paragraph(f"• {reason}", styles['BodyText']))
            story.append(Spacer(1, 3))
            
        story.append(Spacer(1, 20))
        story.append(Paragraph("<i>Report generated automatically by BrandShield-AI Fraud Inspection System.</i>", styles['Italic']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
