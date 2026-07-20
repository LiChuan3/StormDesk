# -*- coding: utf-8 -*-
"""Draw the controlled-question introduction figure as editable PowerPoint."""
import os

from paper_figure_style import (
    ALIGN_C,
    ALIGN_L,
    ANCHOR_MID,
    BLUE_F,
    CORAL,
    CORAL_F,
    GREEN,
    GREEN_DARK,
    GREEN_F,
    INK,
    LAVENDER,
    LINE,
    NAVY,
    ORANGE,
    ORANGE_F,
    PALE,
    PAPER,
    PURPLE,
    PURPLE_F,
    SLATE,
    TEAL,
    TEAL_F,
    VIOLET,
    add_accent,
    add_box,
    add_line,
    add_ortho_arrow,
    add_section_header,
    add_text,
    export_figure,
    make_presentation,
    set_subscripts,
)


HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "fig1_overview.pptx")
PDF = os.path.join(HERE, "fig1_overview.pdf")
PNG = os.path.join(HERE, "fig1_overview.png")


def main():
    app, pres, slide = make_presentation(960, 360)

    # Title and section grammar mirror the DeepAM editable main figure.
    add_text(
        slide,
        42,
        7,
        876,
        26,
        [("Can an LLM add case-specific skill beyond calibrated use of its own tools?", 17.5, True, INK, False)],
    )
    add_text(
        slide,
        155,
        34,
        650,
        14,
        [("Controlled evaluation with matched evidence and a matched bounded action space", 8.5, False, SLATE, False)],
    )
    add_section_header(slide, 20, 56, 218, "MATCHED EVIDENCE", NAVY)
    add_section_header(slide, 380, 56, 354, "MATCHED ACTION SPACE", PURPLE)
    add_section_header(slide, 766, 56, 176, "MEASURED CONTRIBUTION", TEAL)

    # Background containers first.
    add_box(slide, 20, 88, 218, 176, PALE, NAVY, [("", 7, False, INK, False)], line_w=1.25, dash=4, radius=0.04)
    add_box(slide, 382, 88, 352, 198, PALE, LINE, [("", 7, False, INK, False)], line_w=1.05, dash=4, radius=0.04)

    # Stable data-flow corridors, drawn before the cards.
    add_ortho_arrow(slide, [(238, 176), (262, 176)], color=NAVY, weight=1.8)
    add_ortho_arrow(slide, [(356, 176), (400, 176), (400, 143), (430, 143)], color=PURPLE, weight=1.8)
    add_ortho_arrow(slide, [(400, 176), (400, 240), (430, 240)], color=GREEN, weight=1.8)
    add_line(slide, 705, 143, 746, 143, color=PURPLE, weight=1.6)
    add_line(slide, 705, 240, 746, 240, color=GREEN, weight=1.6)
    add_line(slide, 746, 143, 746, 240, color=TEAL, weight=1.6)
    add_ortho_arrow(slide, [(746, 191), (766, 191)], color=TEAL, weight=1.9)

    # Evidence cards.
    add_box(slide, 34, 105, 190, 38, BLUE_F, NAVY,
            [("AIWP models + tracker", 10.0, True, INK, False),
             ("Pangu-Weather · FengWu", 8.0, False, SLATE, False)], line_w=1.0)
    add_accent(slide, 48, 111, 31, LAVENDER, 2.3)
    add_box(slide, 34, 151, 190, 38, TEAL_F, TEAL,
            [("SHIPS-class + satellite IR", 9.7, True, INK, False),
             ("storm-centered diagnostics", 8.0, False, SLATE, False)], line_w=1.0)
    add_accent(slide, 48, 157, 31, TEAL, 2.3)
    add_box(slide, 34, 197, 190, 38, PURPLE_F, VIOLET,
            [("Analogs + member priors", 9.7, True, INK, False),
             ("skill and bias profiles · 2018–2020", 7.8, False, SLATE, False)], line_w=1.0)
    add_accent(slide, 48, 203, 31, VIOLET, 2.3)
    add_text(slide, 50, 241, 158, 14,
             [("deterministic tools; one frozen briefing", 7.1, False, SLATE, True)])

    add_box(slide, 266, 132, 88, 88, PAPER, NAVY,
            [("briefing", 11.0, True, INK, False),
             ("identical for", 9.0, False, SLATE, False),
             ("every policy", 9.0, False, SLATE, False)], line_w=1.45)
    add_accent(slide, 280, 143, 28, NAVY, 2.5)

    # The matched contract and its two policies.
    add_text(slide, 399, 96, 318, 18,
             [("the contract  ·  bounded action space", 10.5, True, INK, False)])
    add_text(slide, 402, 115, 312, 14,
             [("same trust/delta bounds · same calibration · code assembles every number", 6.9, False, SLATE, False)])

    add_box(slide, 430, 133, 275, 62, PURPLE_F, PURPLE,
            [("zero-shot LLM office", 10.8, True, INK, False),
             ("Chief · Track · Intensity · Physics Auditor", 8.5, False, INK, False),
             ("iterates on deterministic physics checks", 8.0, False, SLATE, True)], line_w=1.5)
    add_accent(slide, 447, 141, 34, PURPLE, 2.8)
    add_text(slide, 528, 197, 80, 13, [("versus", 7.4, False, SLATE, True)])
    add_box(slide, 430, 211, 275, 62, GREEN_F, GREEN,
            [("supervised per-case gate", 10.8, True, INK, False),
             ("gradient-boosted · same features · same bounds", 8.5, False, INK, False),
             ("proves attainable case-adaptive headroom", 8.0, False, SLATE, True)], line_w=1.5)
    add_accent(slide, 447, 219, 34, GREEN, 2.8)

    # Contribution measure.
    measure = add_box(slide, 770, 107, 168, 158, PAPER, TEAL,
                      [("headroom utilization", 11.0, True, INK, False),
                       ("U = (L_static − L_LLM) /", 9.0, False, INK, False),
                       ("(L_static − L_learned)", 9.0, False, INK, False),
                       ("share of the supervised gain", 8.0, False, SLATE, True),
                       ("realized by the LLM", 8.0, False, SLATE, True),
                       ("conservative  U ≈ 0", 8.2, True, ORANGE, False),
                       ("decisive  U < 0", 8.2, True, CORAL, False),
                       ("supervised  U = 1", 8.2, True, GREEN_DARK, False)], line_w=1.8)
    add_accent(slide, 788, 117, 34, TEAL, 2.8)
    set_subscripts(measure, 2, ["L_static", "L_LLM"])
    set_subscripts(measure, 3, ["L_static", "L_learned"])

    # Safety is explicitly separated from skill credit.
    add_box(slide, 74, 302, 300, 42, CORAL_F, CORAL,
            [("without the contract", 8.3, True, CORAL, False),
             ("43–45% of free-generated positions land >2,000 km off", 8.4, False, INK, False)], line_w=1.25)
    add_ortho_arrow(slide, [(374, 323), (416, 323)], color=CORAL, weight=1.55)
    add_box(slide, 420, 302, 300, 42, GREEN_F, GREEN,
            [("inside the contract", 8.3, True, GREEN_DARK, False),
             ("format and physics safety is enforced by code—not credited to the LLM", 8.0, False, INK, False)], line_w=1.25)
    add_text(slide, 742, 306, 196, 32,
             [("Safety is a property of the interface;\nskill is what remains to be measured.", 7.8, True, SLATE, True)], align=ALIGN_L, anchor=ANCHOR_MID)

    export_figure(app, pres, slide, PPTX, PDF, PNG)
    print("wrote", PPTX)
    print("wrote", PDF)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
