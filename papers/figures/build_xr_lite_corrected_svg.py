from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("XR-Lite-corrected.svg")

WIDTH = 1800
HEIGHT = 1220

INK = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"
WHITE = "#ffffff"
BG = "#fbfbfd"

TEAL = "#0f766e"
TEAL_SOFT = "#ecfeff"
BLUE = "#2563eb"
BLUE_SOFT = "#eff6ff"
ORANGE = "#ea580c"
ORANGE_SOFT = "#fff7ed"
RED = "#dc2626"
RED_SOFT = "#fef2f2"
PURPLE = "#7c3aed"
PURPLE_SOFT = "#f5f3ff"


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str,
    rx: float = 18,
    stroke_width: float = 2.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}/>'
    )


def line(
    points: list[tuple[float, float]],
    stroke: str = INK,
    width: float = 3.2,
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    coords = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    end = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash}{end}/>'
    )


def text(
    x: float,
    y: float,
    value: str,
    size: float = 18,
    fill: str = INK,
    weight: int = 500,
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif">{escape(value)}</text>'
    )


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    size: float = 18,
    fill: str = INK,
    weight: int = 500,
    anchor: str = "middle",
    leading: float = 22,
) -> list[str]:
    rendered = [
        (
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif">'
        )
    ]
    for idx, item in enumerate(lines):
        dy = "0" if idx == 0 else str(leading)
        rendered.append(f'<tspan x="{x}" dy="{dy}">{escape(item)}</tspan>')
    rendered.append("</text>")
    return rendered


def badge(x: float, y: float, w: float, label: str, fill: str, stroke: str) -> list[str]:
    return [
        rect(x, y, w, 38, fill, stroke, rx=19, stroke_width=1.8),
        text(x + w / 2, y + 25, label, 15, stroke, 700),
    ]


def section(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    index: str,
    stroke: str,
    fill: str,
) -> list[str]:
    return [
        rect(x, y, w, h, fill, stroke, rx=24, stroke_width=2.2, dash="10 8"),
        rect(x + 18, y - 22, 42, 42, WHITE, stroke, rx=21, stroke_width=2.2),
        text(x + 39, y + 7, index, 20, stroke, 700),
        text(x + 82, y + 8, title, 22, stroke, 700, anchor="start"),
    ]


def module(
    x: float,
    y: float,
    w: float,
    h: float,
    title_lines: list[str],
    subtitle_lines: list[str],
    fill: str,
    stroke: str,
    title_size: float = 18,
    subtitle_size: float = 13,
) -> list[str]:
    pieces = [rect(x, y, w, h, fill, stroke, rx=16, stroke_width=2.0)]
    pieces.extend(multiline_text(x + w / 2, y + 28, title_lines, title_size, INK, 700, leading=21))
    if subtitle_lines:
        pieces.extend(
            multiline_text(
                x + w / 2,
                y + h - (18 + (len(subtitle_lines) - 1) * 18),
                subtitle_lines,
                subtitle_size,
                MUTED,
                500,
                leading=18,
            )
        )
    return pieces


def footer_box(x: float, y: float, w: float, h: float, stroke: str, title: str, body: list[str]) -> list[str]:
    pieces = [rect(x, y, w, h, WHITE, stroke, rx=14, stroke_width=2.0)]
    pieces.append(text(x + 18, y + 28, title, 17, stroke, 700, anchor="start"))
    for idx, item in enumerate(body):
        pieces.append(text(x + 18, y + 58 + idx * 22, item, 14, INK, 500, anchor="start"))
    return pieces


def render() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        f'<path d="M0,0 L12,6 L0,12 z" fill="{INK}"/>',
        "</marker>",
        "</defs>",
        rect(0, 0, WIDTH, HEIGHT, BG, BG, rx=0, stroke_width=0),
        text(WIDTH / 2, 58, "XR-Lite Architecture", 36, INK, 700),
    ]

    parts.extend(section(28, 104, 346, 820, "Input Preprocessing", "1", TEAL, TEAL_SOFT))
    parts.extend(section(400, 104, 366, 820, "YOLOv8n Backbone", "2", BLUE, BLUE_SOFT))
    parts.extend(section(792, 104, 560, 820, "PAN-FPN Neck", "3", ORANGE, ORANGE_SOFT))
    parts.extend(section(1378, 104, 394, 820, "Detection Head", "4", RED, RED_SOFT))

    # Input preprocessing
    parts.extend(module(62, 146, 278, 118, ["Pseudo-color X-ray", "Image (RGB)"], ["Domain image prior"], WHITE, TEAL, 18))
    parts.extend(module(62, 304, 278, 78, ["PCN"], ["Pseudo-color normalization"], WHITE, TEAL, 20))
    parts.extend(module(62, 420, 278, 104, ["EGI"], ["Edge-guided input", "(Scharr edge map)"], WHITE, TEAL, 20))
    parts.extend(module(62, 568, 278, 110, ["4-channel XR Input"], ["R, G, B, Edge"], TEAL_SOFT, TEAL, 18))
    parts.append(line([(201, 264), (201, 304)]))
    parts.append(line([(201, 382), (201, 420)]))
    parts.append(line([(201, 524), (201, 568)]))
    parts.extend(footer_box(44, 952, 314, 118, TEAL, "Input summary", ["Channels = 4", "Preprocessing = PCN + EGI"]))

    # Backbone
    parts.extend(module(432, 144, 302, 90, ["P1 / 2"], ["Conv 3x3, s=2", "320 x 320"], WHITE, BLUE))
    parts.extend(module(432, 266, 302, 108, ["P2 / 4"], ["Conv 3x3, s=2", "C2f x3 | 160 x 160"], WHITE, BLUE))
    parts.extend(module(432, 412, 302, 108, ["P3 / 8"], ["Conv 3x3, s=2", "C2f x6 | 80 x 80"], WHITE, BLUE))
    parts.extend(module(432, 558, 302, 118, ["P4 / 16"], ["Conv 3x3, s=2", "C2fEMALite x6 | 40 x 40"], PURPLE_SOFT, PURPLE))
    parts.extend(module(432, 720, 302, 126, ["P5 / 32"], ["Conv 3x3, s=2", "C2f x3 + SPPFAIFILite", "20 x 20"], PURPLE_SOFT, PURPLE))
    parts.append(line([(583, 234), (583, 266)]))
    parts.append(line([(583, 374), (583, 412)]))
    parts.append(line([(583, 520), (583, 558)]))
    parts.append(line([(583, 676), (583, 720)]))
    parts.extend(badge(528, 604, 136, "EMA-Lite", PURPLE_SOFT, PURPLE))
    parts.extend(badge(506, 798, 180, "SPPF + AIFI-Lite", PURPLE_SOFT, PURPLE))
    parts.extend(footer_box(416, 952, 334, 118, BLUE, "Backbone summary", ["YOLOv8n-XR-Lite", "P3 / P4 / P5 features"]))

    # Bridge input to backbone
    parts.append(line([(340, 623), (390, 623), (390, 189), (432, 189)]))

    # Neck modules
    parts.extend(module(838, 150, 194, 74, ["Upsample x2"], ["P5 -> P4"], WHITE, ORANGE))
    parts.extend(module(1050, 150, 194, 74, ["Concat"], ["Upsampled P5 + P4"], WHITE, ORANGE))
    parts.extend(module(838, 258, 406, 78, ["C2f x3"], ["P4 top-down intermediate"], WHITE, ORANGE))
    parts.extend(module(838, 376, 194, 74, ["Upsample x2"], ["P4 -> P3"], WHITE, ORANGE))
    parts.extend(module(1050, 376, 194, 74, ["Concat"], ["Upsampled P4 + P3"], WHITE, ORANGE))
    parts.extend(module(838, 484, 406, 78, ["C2f x3"], ["Neck P3 / 8 (top-down output)"], WHITE, ORANGE))
    parts.extend(module(838, 606, 194, 74, ["Conv 3x3, s=2"], ["P3 -> P4"], WHITE, ORANGE))
    parts.extend(module(1050, 606, 194, 74, ["Concat"], ["P4 return fusion"], WHITE, ORANGE))
    parts.extend(module(838, 714, 406, 82, ["C2fEMALite x3"], ["Neck P4 / 16 (refined output)"], PURPLE_SOFT, PURPLE))
    parts.extend(module(838, 830, 194, 74, ["Conv 3x3, s=2"], ["P4 -> P5"], WHITE, ORANGE))
    parts.extend(module(1050, 830, 194, 74, ["Concat"], ["P5 return fusion"], WHITE, ORANGE))
    parts.extend(module(838, 936, 406, 82, ["C2fEMALite x3"], ["Neck P5 / 32 (refined output)"], PURPLE_SOFT, PURPLE))

    # Neck topology
    parts.append(line([(734, 783), (790, 783), (790, 187), (838, 187)], stroke=BLUE))
    parts.append(line([(734, 617), (1012, 617), (1012, 187), (1050, 187)], stroke=BLUE))
    parts.append(line([(1032, 187), (1050, 187)]))
    parts.append(line([(1147, 224), (1147, 258)]))
    parts.append(line([(1041, 336), (1041, 376)]))
    parts.append(line([(734, 466), (1008, 466), (1008, 413), (1050, 413)], stroke=BLUE))
    parts.append(line([(1032, 413), (1050, 413)]))
    parts.append(line([(1147, 450), (1147, 484)]))
    parts.append(line([(1041, 562), (1041, 606)]))
    parts.append(line([(1147, 680), (1147, 696), (1041, 696), (1041, 714)]))
    parts.append(line([(1244, 297), (1308, 297), (1308, 643), (1244, 643)], stroke=ORANGE))
    parts.append(line([(1041, 796), (1041, 830)]))
    parts.append(line([(734, 783), (812, 783), (812, 867), (1050, 867)], stroke=BLUE))
    parts.append(line([(1032, 867), (1050, 867)]))
    parts.append(line([(1147, 904), (1147, 920), (1041, 920), (1041, 936)]))

    # Detection head
    parts.extend(module(1422, 264, 308, 126, ["Detect: P3 / 8"], ["Small-object head"], WHITE, RED, 20))
    parts.extend(module(1422, 516, 308, 126, ["Detect: P4 / 16"], ["Medium-object head"], WHITE, RED, 20))
    parts.extend(module(1422, 768, 308, 126, ["Detect: P5 / 32"], ["Large-object head"], WHITE, RED, 20))
    parts.append(line([(1244, 523), (1362, 523), (1362, 327), (1422, 327)], stroke=RED))
    parts.append(line([(1244, 755), (1362, 755), (1362, 579), (1422, 579)], stroke=RED))
    parts.append(line([(1244, 977), (1362, 977), (1362, 831), (1422, 831)], stroke=RED))
    parts.extend(footer_box(1402, 952, 344, 118, RED, "Detection summary", ["Three-scale predictions", "P3 / P4 / P5"]))

    # Small annotations clarifying corrected flows
    parts.extend(badge(956, 342, 174, "Corrected top-down", ORANGE_SOFT, ORANGE))
    parts.extend(badge(950, 878, 186, "Corrected bottom-up", ORANGE_SOFT, ORANGE))
    parts.extend(multiline_text(1086, 1090, ["XR-Lite = PCN + EGI + EMA-Lite + AIFI-Lite + YOLOv8n"], 24, INK, 700))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
