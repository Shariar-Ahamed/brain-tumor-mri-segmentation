import os
import io
import numpy as np
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

def generate_pdf_report(patient_id, modality, slice_idx, model_name, dice_score, iou_score, tumor_vol, img_buf=None):
    """
    Generates a professional medical diagnostic PDF report for Brain Tumor MRI Analysis.
    """
    if not HAS_REPORTLAB:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = []

    # Title Banner
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#00f2fe'),
        spaceAfter=12
    )
    story.append(Paragraph("NeuroSeg AI — Brain Tumor Diagnostic Report", title_style))
    story.append(Paragraph("<b>Daffodil International University — Department of CSE</b>", styles['Normal']))
    story.append(Spacer(1, 15))

    # Patient Metadata Table
    meta_data = [
        ["Patient Case ID:", patient_id, "MRI Sequence:", modality],
        ["Axial Slice Index:", f"Slice #{slice_idx} / 155", "Model Pipeline:", model_name],
        ["Predicted Dice:", f"{dice_score:.4f}", "Mean IoU:", f"{iou_score:.4f}"],
        ["Estimated Tumor Volume:", f"{tumor_vol} cm³", "Status:", "Analysis Complete ✅"]
    ]

    t = Table(meta_data, colWidths=[130, 140, 120, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0e131f')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2a364f')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Add Image if provided
    if img_buf:
        try:
            rl_img = RLImage(img_buf, width=280, height=280)
            story.append(rl_img)
            story.append(Spacer(1, 15))
        except Exception:
            pass

    # Findings Summary
    story.append(Paragraph("<b>Radiology Diagnostic Summary:</b>", styles['Heading2']))
    story.append(Paragraph(f"Multi-modal 2D axial segmentation slice #{slice_idx} evaluated using <b>{model_name}</b>. Quantitative assessment demonstrates high structural overlap (Dice: {dice_score:.4f}, IoU: {iou_score:.4f}) across Enhancing Tumor (ET), Tumor Core (TC), and Peritumoral Edema (ED) regions.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer
