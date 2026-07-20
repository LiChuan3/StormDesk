# -*- coding: utf-8 -*-
"""Shared PowerPoint drawing helpers for the paper's editable figures.

The palette and typography are sampled from the editable DeepAM main figure.
All data-flow arrows are built from explicit horizontal/vertical segments so
PowerPoint cannot introduce bent or unstable connector routing.
"""
from __future__ import annotations

import os

import win32com.client


FONT = "Comic Sans MS"


def rgb(value: str) -> int:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return r | (g << 8) | (b << 16)


INK = rgb("111827")
NAVY = rgb("17324D")
SLATE = rgb("667085")
MID = rgb("98A2B3")
LINE = rgb("B8C0CF")
PAPER = rgb("FFFFFF")
PALE = rgb("F8FAFC")

PURPLE = rgb("5558D9")
PURPLE_2 = rgb("6B6EE3")
LAVENDER = rgb("86A3EE")
VIOLET = rgb("A778E6")
TEAL = rgb("2A8F8B")
GREEN = rgb("4E8A32")
GREEN_DARK = rgb("315F22")
ORANGE = rgb("D9650B")
CORAL = rgb("E2745B")

PURPLE_F = rgb("EEF0FF")
BLUE_F = rgb("EAF3FB")
TEAL_F = rgb("E5F4F3")
GREEN_F = rgb("ECF6E8")
ORANGE_F = rgb("FFF2DE")
CORAL_F = rgb("FCEAE6")
VIOLET_F = rgb("F2ECFC")

MSO_ROUND_RECT = 5
MSO_RECT = 1
PP_BLANK = 12
ALIGN_L, ALIGN_C, ALIGN_R = 1, 2, 3
ANCHOR_TOP, ANCHOR_MID = 1, 3


def make_presentation(width: float, height: float):
    app = win32com.client.DispatchEx("PowerPoint.Application")
    pres = app.Presentations.Add(WithWindow=0)
    pres.PageSetup.SlideWidth = width
    pres.PageSetup.SlideHeight = height
    slide = pres.Slides.Add(1, PP_BLANK)
    slide.FollowMasterBackground = 0
    slide.Background.Fill.ForeColor.RGB = PAPER
    return app, pres, slide


def add_box(
    slide,
    x,
    y,
    w,
    h,
    fill,
    line,
    paras,
    *,
    line_w=1.35,
    dash=None,
    align=ALIGN_C,
    anchor=ANCHOR_MID,
    radius=0.10,
    margins=(5, 5, 3, 3),
):
    """Add a rounded vector box.

    ``paras`` is a list of ``(text, size, bold, color, italic)`` tuples.
    """
    sh = slide.Shapes.AddShape(MSO_ROUND_RECT, x, y, w, h)
    try:
        sh.Adjustments[1] = radius
    except Exception:
        pass
    if fill is None:
        sh.Fill.Visible = 0
    else:
        sh.Fill.Visible = -1
        sh.Fill.ForeColor.RGB = fill
        sh.Fill.Transparency = 0
    if line is None:
        sh.Line.Visible = 0
    else:
        sh.Line.Visible = -1
        sh.Line.ForeColor.RGB = line
        sh.Line.Weight = line_w
        if dash:
            sh.Line.DashStyle = dash

    tf = sh.TextFrame
    tf.AutoSize = 0
    tf.WordWrap = -1
    tf.MarginLeft, tf.MarginRight, tf.MarginTop, tf.MarginBottom = margins
    tf.VerticalAnchor = anchor
    tr = tf.TextRange
    tr.Text = "\r".join(p[0] for p in paras)
    tr.Font.Name = FONT
    for i, (_, size, bold, color, italic) in enumerate(paras, 1):
        pr = tr.Paragraphs(i)
        pr.Font.Name = FONT
        pr.Font.Size = size
        pr.Font.Bold = -1 if bold else 0
        pr.Font.Italic = -1 if italic else 0
        pr.Font.Color.RGB = color
        pr.ParagraphFormat.Alignment = align
        pr.ParagraphFormat.SpaceBefore = 0
        pr.ParagraphFormat.SpaceAfter = 1.0
        pr.ParagraphFormat.SpaceWithin = 1.0
    return sh


def add_text(slide, x, y, w, h, paras, *, align=ALIGN_C, anchor=ANCHOR_TOP):
    return add_box(
        slide,
        x,
        y,
        w,
        h,
        None,
        None,
        paras,
        align=align,
        anchor=anchor,
        margins=(0, 0, 0, 0),
    )


def add_line(slide, x1, y1, x2, y2, *, color=INK, weight=1.6, dash=None, head=False):
    if x1 != x2 and y1 != y2:
        raise ValueError(f"Only orthogonal line segments are allowed: {(x1, y1, x2, y2)}")
    ln = slide.Shapes.AddLine(x1, y1, x2, y2)
    ln.Line.ForeColor.RGB = color
    ln.Line.Weight = weight
    if dash:
        ln.Line.DashStyle = dash
    if head:
        ln.Line.EndArrowheadStyle = 3
        ln.Line.EndArrowheadLength = 2
        ln.Line.EndArrowheadWidth = 2
    return ln


def add_ortho_arrow(slide, points, *, color=INK, weight=1.6, dash=None):
    """Draw a stable orthogonal polyline; only the last segment has a head."""
    if len(points) < 2:
        raise ValueError("An arrow needs at least two points")
    result = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        result.append(
            add_line(
                slide,
                a[0],
                a[1],
                b[0],
                b[1],
                color=color,
                weight=weight,
                dash=dash,
                head=i == len(points) - 2,
            )
        )
    return result


def add_section_header(slide, x, y, w, label, color=SLATE):
    add_text(slide, x, y, w, 14, [(label, 8.5, True, color, False)])
    add_line(slide, x + 4, y + 18, x + w - 4, y + 18, color=LINE, weight=0.8)


def add_accent(slide, x, y, w, color, weight=2.8):
    add_line(slide, x, y, x + w, y, color=color, weight=weight)


def set_subscripts(shape, paragraph_index, tokens):
    para = shape.TextFrame.TextRange.Paragraphs(paragraph_index)
    text = para.Text
    for token in tokens:
        start = text.find(token)
        if start >= 0:
            para.Characters(start + 2, len(token) - 1).Font.Subscript = -1


def export_figure(app, pres, slide, pptx_path, pdf_path, png_path, width_px=4000):
    """Save editable PPTX, vector PDF, and a high-resolution PNG preview."""
    pptx_path = os.path.abspath(pptx_path)
    pdf_path = os.path.abspath(pdf_path)
    png_path = os.path.abspath(png_path)
    raw_pdf = os.path.splitext(pdf_path)[0] + "_raw.pdf"
    pres.SaveAs(pptx_path)
    pres.SaveAs(raw_pdf, 32)
    height_px = round(width_px * pres.PageSetup.SlideHeight / pres.PageSetup.SlideWidth)
    slide.Export(png_path, "PNG", width_px, height_px)
    pres.Close()
    app.Quit()

    # The slide is already sized to the figure bounds. Keeping that exact page
    # box avoids Windows pdfcrop's hard-link requirement on restricted drives.
    import shutil

    shutil.copyfile(raw_pdf, pdf_path)
    try:
        os.remove(raw_pdf)
    except OSError:
        pass
