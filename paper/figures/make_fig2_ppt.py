# -*- coding: utf-8 -*-
"""Draw the StormDesk method figure as editable PowerPoint."""
import os

from paper_figure_style import (
    ALIGN_L,
    ANCHOR_MID,
    BLUE_F,
    CORAL,
    GREEN,
    GREEN_DARK,
    GREEN_F,
    INK,
    LAVENDER,
    LINE,
    NAVY,
    PALE,
    PAPER,
    PURPLE,
    PURPLE_2,
    PURPLE_F,
    SLATE,
    TEAL,
    TEAL_F,
    VIOLET,
    VIOLET_F,
    add_accent,
    add_box,
    add_line,
    add_ortho_arrow,
    add_section_header,
    add_text,
    export_figure,
    make_presentation,
)


HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "fig2_office.pptx")
PDF = os.path.join(HERE, "fig2_office.pdf")
PNG = os.path.join(HERE, "fig2_office.png")


def main():
    app, pres, slide = make_presentation(960, 390)

    add_text(slide, 110, 7, 740, 26,
             [("StormDesk: bounded adjudication over deterministic tropical-cyclone tools", 17.0, True, INK, False)])
    add_text(slide, 190, 34, 580, 14,
             [("Five LLM calls instrument trust, corrections, auditing, and synthesis inside one code-enforced contract", 8.2, False, SLATE, False)])
    add_section_header(slide, 16, 56, 274, "(a) DETERMINISTIC EVIDENCE", NAVY)
    add_section_header(slide, 326, 56, 618, "(b) BOUNDED ITERATIVE ADJUDICATION", PURPLE)
    add_line(slide, 307, 54, 307, 356, color=LINE, weight=0.8)

    # Panel containers.
    add_box(slide, 326, 82, 618, 270, PALE, LINE, [("", 7, False, INK, False)], line_w=1.0, dash=4, radius=0.04)

    # Evidence aggregation spine and the complete office routing are placed
    # before the cards so connection endpoints stay visually clean.
    add_line(slide, 230, 112, 230, 290, color=NAVY, weight=1.25)
    for cy in (112, 170, 228, 286):
        add_line(slide, 220, cy, 230, cy, color=NAVY, weight=1.25)
    add_ortho_arrow(slide, [(230, 199), (244, 199)], color=NAVY, weight=1.65)

    # Briefing -> Chief, using the narrow corridor between panels.
    add_ortho_arrow(slide, [(300, 199), (316, 199), (316, 125), (344, 125)], color=NAVY, weight=1.75)

    # Chief -> two specialists: one vertical fork, two horizontal branches.
    add_line(slide, 504, 125, 520, 125, color=PURPLE, weight=1.65)
    add_line(slide, 520, 112, 520, 199, color=PURPLE, weight=1.65)
    add_ortho_arrow(slide, [(520, 112), (536, 112)], color=PURPLE, weight=1.65)
    add_ortho_arrow(slide, [(520, 199), (536, 199)], color=VIOLET, weight=1.65)

    # Specialists -> code-assembled draft: merge into a dedicated corridor.
    add_line(slide, 690, 112, 708, 112, color=PURPLE, weight=1.55)
    add_line(slide, 690, 199, 708, 199, color=VIOLET, weight=1.55)
    add_line(slide, 708, 112, 708, 199, color=TEAL, weight=1.55)
    add_ortho_arrow(slide, [(708, 156), (722, 156)], color=TEAL, weight=1.75)

    # Draft <-> Auditor on two separated horizontal rails.
    add_ortho_arrow(slide, [(812, 143), (832, 143)], color=INK, weight=1.65)
    add_ortho_arrow(slide, [(832, 190), (812, 190)], color=CORAL, weight=1.65)

    # Draft and auditor enter Chief synthesis through independent vertical rails.
    add_ortho_arrow(slide, [(767, 210), (767, 244)], color=TEAL, weight=1.65)
    add_ortho_arrow(slide, [(877, 210), (877, 244)], color=CORAL, weight=1.65)

    # Evidence cards.
    cards = [
        (86, BLUE_F, NAVY, "storm state + satellite", "best-track history · motion · IR structure"),
        (144, TEAL_F, TEAL, "environmental diagnostics", "shear · steering · SST · RH · MPI/POT"),
        (202, PURPLE_F, PURPLE, "guidance + skill priors", "AIWP · GRU · Transformer · CLIPER5-class"),
        (260, VIOLET_F, VIOLET, "historical analogs", "top-k state/environment match · observed ΔV/RI"),
    ]
    for y, fill, color, title, body in cards:
        add_box(slide, 16, y, 204, 52, fill, color,
                [(title, 9.8, True, INK, False),
                 (body, 7.8, False, SLATE, False)], line_w=1.05)
        add_accent(slide, 30, y + 8, 28, color, 2.3)

    add_box(slide, 244, 158, 56, 82, PAPER, NAVY,
            [("shared", 8.3, True, INK, False),
             ("briefing", 8.3, True, INK, False),
             ("matched", 6.8, False, SLATE, True),
             ("information", 6.8, False, SLATE, True)], line_w=1.35)
    add_text(slide, 30, 326, 250, 18,
             [("All evidence is deterministic; the LLM never generates a forecast number directly.", 6.8, False, SLATE, True)])

    # Office nodes: DeepAM's saturated blocks + white-on-color labels.
    add_box(slide, 344, 96, 160, 58, PURPLE, PURPLE,
            [("1 · Chief Forecaster", 10.0, True, PAPER, False),
             ("agenda + synoptic assessment", 8.0, False, PAPER, False),
             ("steering? RI? land?", 7.6, False, PAPER, True)], line_w=1.2)
    add_accent(slide, 359, 104, 30, LAVENDER, 2.6)

    add_box(slide, 536, 88, 154, 60, LAVENDER, LAVENDER,
            [("2 · Track Specialist", 9.8, True, PAPER, False),
             ("trust factors on skill priors", 7.8, False, PAPER, False),
             ("optional nudge ≤ 60 km", 7.5, False, PAPER, True)], line_w=1.2)
    add_accent(slide, 551, 96, 30, PAPER, 2.3)

    add_box(slide, 536, 169, 154, 60, VIOLET, VIOLET,
            [("3 · Intensity Specialist", 9.5, True, PAPER, False),
             ("bounded correction ±25 kt", 7.8, False, PAPER, False),
             ("explicit RI probability", 7.5, False, PAPER, True)], line_w=1.2)
    add_accent(slide, 551, 177, 30, PAPER, 2.3)

    add_box(slide, 722, 116, 90, 94, TEAL, TEAL,
            [("code-assembled", 8.8, True, PAPER, False),
             ("draft forecast", 8.8, True, PAPER, False),
             ("weighted track", 6.8, False, PAPER, False),
             ("prior + shrinkage", 6.8, False, PAPER, False),
             ("physics caps", 6.8, False, PAPER, False)], line_w=1.2)
    add_accent(slide, 736, 125, 28, PAPER, 2.3)

    add_box(slide, 832, 116, 90, 94, CORAL, CORAL,
            [("4 · Physics", 8.8, True, PAPER, False),
             ("Auditor", 8.8, True, PAPER, False),
             ("automated checks", 6.8, False, PAPER, False),
             ("bounded fixes", 6.8, False, PAPER, False),
             ("pass / revise", 6.8, False, PAPER, False)], line_w=1.2)
    add_accent(slide, 846, 125, 28, PAPER, 2.3)
    add_text(slide, 812, 126, 20, 14, [("draft", 5.8, False, SLATE, False)])
    add_text(slide, 812, 193, 20, 14, [("fix", 5.8, False, CORAL, False)])

    add_box(slide, 722, 244, 200, 58, GREEN, GREEN,
            [("5 · Chief synthesis → issued forecast", 9.4, True, PAPER, False),
             ("accept/reject fixes · RI probability", 7.5, False, PAPER, False),
             ("written discussion (artifact)", 7.2, False, PAPER, True)], line_w=1.2)
    add_accent(slide, 738, 252, 32, PAPER, 2.4)

    # Contract and comparison notes.
    add_box(slide, 346, 314, 574, 24, GREEN_F, GREEN,
            [("contract:  τ∈[0.25,4] · track nudge≤60 km · Δv clipped±25 kt · audit±20 kt · code assembles all numbers", 6.7, True, GREEN_DARK, False)],
            line_w=0.9, radius=0.18)
    add_text(slide, 332, 358, 606, 25,
             [("Equal-budget single agent: the same five calls (answer + four self-refinement rounds over the same checks) · evaluated as a co-equal policy", 7.0, False, SLATE, True)],
             align=ALIGN_L, anchor=ANCHOR_MID)

    export_figure(app, pres, slide, PPTX, PDF, PNG)
    print("wrote", PPTX)
    print("wrote", PDF)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
