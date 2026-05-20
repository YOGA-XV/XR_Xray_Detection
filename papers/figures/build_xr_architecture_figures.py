from __future__ import annotations

from pathlib import Path
from typing import Iterable


OUT_DIR = Path(__file__).resolve().parent

INK = "#182235"
MUTED = "#5A6678"
GRID = "#D9E1EC"
PANEL = "#F7F9FC"
INPUT = "#1E9C8C"
BACKBONE = "#2F5DCC"
NECK = "#E8872E"
HEAD = "#D94B4B"
ACCENT = "#8A4FFF"
ACCENT_SOFT = "#F0EAFF"
TEAL_SOFT = "#E7F7F4"
BLUE_SOFT = "#EDF3FF"
ORANGE_SOFT = "#FFF1E4"
RED_SOFT = "#FDEBEC"
WHITE = "#FFFFFF"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        f'  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>',
        f'  <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ACCENT}"/></marker>',
        f'  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#C8D3E3" flood-opacity="0.22"/></filter>',
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{WHITE}"/>',
        text(width / 2, 48, title, 28, 700, INK, anchor="middle"),
    ]


def svg_end() -> str:
    return "</svg>\n"


def text(
    x: float,
    y: float,
    value: str,
    size: int = 16,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    family: str = "Arial, Helvetica, sans-serif",
    italic: bool = False,
) -> str:
    style = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{style}>{esc(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str,
    radius: int = 16,
    stroke_width: float = 1.6,
    dashed: bool = False,
    shadow: bool = False,
) -> str:
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    filt = ' filter="url(#softShadow)"' if shadow else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}{filt}/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = INK,
    width: float = 2.2,
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow and stroke != ACCENT else ""
    if arrow and stroke == ACCENT:
        marker = ' marker-end="url(#arrow-accent)"'
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round"{dash}{marker}/>'
    )


def polyline(
    points: Iterable[tuple[float, float]],
    stroke: str = INK,
    width: float = 2.2,
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    point_str = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow and stroke != ACCENT else ""
    if arrow and stroke == ACCENT:
        marker = ' marker-end="url(#arrow-accent)"'
    return (
        f'<polyline points="{point_str}" fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}{marker}/>'
    )


def pill(x: float, y: float, w: float, label: str, fill: str, stroke: str, txt: str, weight: int = 700) -> list[str]:
    return [
        rect(x, y, w, 30, fill, stroke, radius=15, stroke_width=1.3),
        text(x + w / 2, y + 20, label, 13, weight, txt, anchor="middle"),
    ]


def section(x: float, y: float, w: float, h: float, label: str, stroke: str, fill: str, index: int) -> list[str]:
    parts = [
        rect(x, y, w, h, fill, stroke, radius=22, stroke_width=1.5, dashed=True),
        rect(x + 20, y - 18, 34, 34, WHITE, stroke, radius=17, stroke_width=1.8),
        text(x + 37, y + 5, str(index), 16, 700, stroke, anchor="middle"),
        text(x + 68, y + 8, label, 18, 700, stroke),
    ]
    return parts


def feature_box(x: float, y: float, w: float, h: float, title: str, subtitle: str, fill: str, stroke: str, accent: bool = False) -> list[str]:
    parts = [
        rect(x, y, w, h, fill, stroke, radius=16, stroke_width=1.8, shadow=accent),
        text(x + 18, y + 26, title, 16, 700, stroke),
        text(x + 18, y + 48, subtitle, 13, 400, MUTED),
    ]
    return parts


def module_box(x: float, y: float, w: float, label: str, color: str, note: str | None = None) -> list[str]:
    parts = [
        rect(x, y, w, 54, WHITE, color, radius=14, stroke_width=1.8),
        text(x + w / 2, y + 22, label, 15, 700, color, anchor="middle"),
    ]
    if note:
        parts.append(text(x + w / 2, y + 40, note, 12, 400, MUTED, anchor="middle"))
    return parts


def title_strip(parts: list[str], x: float, y: float, label: str, fill: str, stroke: str) -> None:
    parts.extend(pill(x, y, 134, label, fill, stroke, stroke))


def _flatten(items: Iterable[str | list[str]]) -> list[str]:
    flat: list[str] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


def save(name: str, lines: list[str | list[str]]) -> None:
    path = OUT_DIR / name
    path.write_text("\n".join(_flatten(lines) + [svg_end()]), encoding="utf-8")


def overall_backbone_neck(parts: list[str], lite: bool, plus: bool) -> None:
    parts.extend(section(430, 118, 390, 628, "Backbone", BACKBONE, BLUE_SOFT, 2))
    parts.extend(section(850, 118, 470, 628, "Feature Fusion", NECK, ORANGE_SOFT, 3))
    parts.extend(section(1350, 118, 300, 628, "Detection", HEAD, RED_SOFT, 4))

    stages = [
        (472, 186, "P2 / 4", "High-resolution stage"),
        (472, 302, "P3 / 8", "Small-object semantic stage"),
        (472, 418, "P4 / 16", "Mid-level context stage"),
        (472, 534, "P5 / 32", "Global context stage"),
    ]
    for x, y, title_, subtitle in stages:
        accent = lite and title_ in {"P4 / 16", "P5 / 32"}
        stroke = ACCENT if accent else BACKBONE
        fill = ACCENT_SOFT if accent else WHITE
        parts.extend(feature_box(x, y, 292, 82, title_, subtitle, fill, stroke, accent=accent))

    parts.extend(
        [
            line(618, 268, 618, 292),
            line(618, 384, 618, 408),
            line(618, 500, 618, 524),
        ]
    )

    if lite:
        title_strip(parts, 674, 431, "EMA-Lite", ACCENT_SOFT, ACCENT)
        title_strip(parts, 674, 547, "AIFI-Lite", ACCENT_SOFT, ACCENT)

    fusion = [
        (912, 214, "P5 to P4", "Upsample + fuse"),
        (912, 330, "P4 to P3", "Upsample + fuse"),
        (1100, 446, "P3 to P4", "Downsample + fuse"),
        (1100, 562, "P4 to P5", "Downsample + fuse"),
    ]
    for x, y, title_, subtitle in fusion:
        accent = lite and ("P3 to P4" in title_ or "P4 to P5" in title_)
        stroke = ACCENT if accent else NECK
        fill = ACCENT_SOFT if accent else WHITE
        parts.extend(feature_box(x, y, 176, 74, title_, subtitle, fill, stroke, accent=accent))

    parts.extend(
        [
            polyline([(764, 575), (824, 575), (824, 251), (912, 251)]),
            polyline([(764, 459), (850, 459), (850, 367), (912, 367)]),
            polyline([(1088, 251), (1120, 251), (1120, 367), (1088, 367)]),
            polyline([(1088, 367), (1156, 367), (1156, 483), (1100, 483)]),
            polyline([(1276, 483), (1308, 483), (1308, 599), (1100, 599)]),
        ]
    )

    detections = [
        (1390, 250, "Detect P3", "stride 8"),
        (1390, 388, "Detect P4", "stride 16"),
        (1390, 526, "Detect P5", "stride 32"),
    ]
    if plus:
        detections = [
            (1390, 168, "Detect P2", "stride 4"),
            (1390, 300, "Detect P3", "stride 8"),
            (1390, 432, "Detect P4", "stride 16"),
            (1390, 564, "Detect P5", "stride 32"),
        ]
    for x, y, title_, subtitle in detections:
        parts.extend(feature_box(x, y, 220, 76, title_, subtitle, WHITE, HEAD))

    if plus:
        parts.extend(
            [
                polyline([(764, 227), (846, 227), (846, 206), (1390, 206)], stroke=ACCENT),
                pill(1008, 154, 184, "P2-Lite branch", ACCENT_SOFT, ACCENT, ACCENT),
            ]
        )
    if plus:
        parts.extend(
            [
                line(1276, 367, 1390, 338),
                line(1276, 483, 1390, 470),
                line(1276, 599, 1390, 602),
            ]
        )
    else:
        parts.extend(
            [
                line(1276, 367, 1390, 288),
                line(1276, 483, 1390, 426),
                line(1276, 599, 1390, 564),
            ]
        )


def xr_nano_overall() -> None:
    parts = svg_start(1700, 820, "XR-Nano Overall Architecture")
    parts.extend(section(42, 118, 346, 628, "Input Enhancement", INPUT, TEAL_SOFT, 1))
    parts.extend(
        [
            feature_box(78, 188, 274, 98, "Pseudo-color X-ray", "Domain image prior", WHITE, INPUT),
            feature_box(78, 330, 126, 82, "PCN", "Contrast normalization", WHITE, INPUT),
            feature_box(226, 330, 126, 82, "EGI", "Edge guidance", WHITE, INPUT),
            feature_box(78, 470, 274, 94, "4-channel input", "Image + enhanced guidance", TEAL_SOFT, INPUT, accent=True),
            polyline([(215, 286), (215, 318)]),
            polyline([(141, 412), (141, 442), (215, 442), (215, 470)]),
            polyline([(289, 412), (289, 442), (215, 442)], arrow=False),
        ]
    )
    overall_backbone_neck(parts, lite=False, plus=False)
    parts.extend(
        [
            polyline([(352, 517), (396, 517), (396, 227), (472, 227)]),
            text(850, 780, "XR-Nano = YOLOv8n + PCN + EGI", 18, 700, INPUT, anchor="middle"),
        ]
    )
    save("xr_nano_overall.svg", parts)


def xr_lite_overall() -> None:
    parts = svg_start(1700, 820, "XR-Lite Overall Architecture")
    parts.extend(section(42, 118, 346, 628, "Input Enhancement", INPUT, TEAL_SOFT, 1))
    parts.extend(
        [
            feature_box(78, 188, 274, 98, "Pseudo-color X-ray", "Domain image prior", WHITE, INPUT),
            feature_box(78, 330, 126, 82, "PCN", "Contrast normalization", WHITE, INPUT),
            feature_box(226, 330, 126, 82, "EGI", "Edge guidance", WHITE, INPUT),
            feature_box(78, 470, 274, 94, "4-channel input", "Image + enhanced guidance", TEAL_SOFT, INPUT, accent=True),
            polyline([(215, 286), (215, 318)]),
            polyline([(141, 412), (141, 442), (215, 442), (215, 470)]),
            polyline([(289, 412), (289, 442), (215, 442)], arrow=False),
            polyline([(352, 517), (396, 517), (396, 227), (472, 227)]),
        ]
    )
    overall_backbone_neck(parts, lite=True, plus=False)
    parts.extend(
        [
            pill(898, 688, 206, "Increment over XR-Nano", ACCENT_SOFT, ACCENT, ACCENT),
            text(1001, 726, "EMA-Lite + AIFI-Lite", 15, 700, ACCENT, anchor="middle"),
            text(850, 780, "XR-Lite = XR-Nano + EMA-Lite + AIFI-Lite", 18, 700, ACCENT, anchor="middle"),
        ]
    )
    save("xr_lite_overall.svg", parts)


def xr_plus_overall() -> None:
    parts = svg_start(1700, 820, "XR-Plus Overall Architecture")
    parts.extend(section(42, 118, 346, 628, "Input Enhancement", INPUT, TEAL_SOFT, 1))
    parts.extend(
        [
            feature_box(78, 188, 274, 98, "Pseudo-color X-ray", "Domain image prior", WHITE, INPUT),
            feature_box(78, 330, 126, 82, "PCN", "Contrast normalization", WHITE, INPUT),
            feature_box(226, 330, 126, 82, "EGI", "Edge guidance", WHITE, INPUT),
            feature_box(78, 470, 274, 94, "4-channel input", "Image + enhanced guidance", TEAL_SOFT, INPUT, accent=True),
            polyline([(215, 286), (215, 318)]),
            polyline([(141, 412), (141, 442), (215, 442), (215, 470)]),
            polyline([(289, 412), (289, 442), (215, 442)], arrow=False),
            polyline([(352, 517), (396, 517), (396, 227), (472, 227)]),
        ]
    )
    overall_backbone_neck(parts, lite=True, plus=True)
    parts.extend(
        [
            pill(884, 688, 238, "Increment over XR-Lite", ACCENT_SOFT, ACCENT, ACCENT),
            text(1003, 726, "P2-Lite high-resolution branch", 15, 700, ACCENT, anchor="middle"),
            text(850, 780, "XR-Plus = XR-Lite + P2-Lite", 18, 700, ACCENT, anchor="middle"),
        ]
    )
    save("xr_plus_overall.svg", parts)


def xr_lite_modules() -> None:
    parts = svg_start(1800, 880, "XR-Lite Enhanced Modules")
    parts.extend(
        [
            rect(40, 116, 820, 710, PANEL, GRID, radius=26, stroke_width=1.5),
            rect(940, 116, 820, 710, PANEL, GRID, radius=26, stroke_width=1.5),
            pill(78, 146, 174, "(a) EMA-Lite", ACCENT_SOFT, ACCENT, ACCENT),
            pill(978, 146, 188, "(b) AIFI-Lite", ACCENT_SOFT, ACCENT, ACCENT),
            text(90, 210, "Residual spatial-channel attention", 18, 700, INK),
            text(990, 210, "Reduced global interaction at P5", 18, 700, INK),
        ]
    )

    # EMA-Lite
    parts.extend(
        [
            feature_box(92, 302, 220, 88, "Input feature F", "C x H x W", WHITE, BACKBONE),
            feature_box(370, 244, 200, 74, "DWConv 3x3", "Local texture response", WHITE, ACCENT),
            feature_box(370, 348, 200, 74, "H / W pooling", "Directional context", WHITE, ACCENT),
            feature_box(370, 452, 200, 74, "Global pooling", "Channel summary", WHITE, ACCENT),
            feature_box(632, 348, 170, 92, "Attention map A", "Spatial x channel", ACCENT_SOFT, ACCENT, accent=True),
            feature_box(632, 530, 170, 88, "Output F'", "F + F x A", WHITE, ACCENT),
            polyline([(312, 346), (342, 346), (342, 281), (370, 281)]),
            polyline([(312, 346), (342, 346), (342, 385), (370, 385)]),
            polyline([(312, 346), (342, 346), (342, 489), (370, 489)]),
            polyline([(570, 281), (604, 281), (604, 394), (632, 394)]),
            polyline([(570, 385), (604, 385), (604, 394)], arrow=False),
            polyline([(570, 489), (604, 489), (604, 394)], arrow=False),
            polyline([(717, 440), (717, 530)]),
            polyline([(312, 346), (344, 346), (344, 574), (632, 574)], dashed=True),
            text(438, 658, "Edge-aware local refinement + residual recalibration", 16, 700, ACCENT, anchor="middle"),
        ]
    )

    # AIFI-Lite
    parts.extend(
        [
            feature_box(992, 302, 190, 88, "Input feature", "P5 context tensor", WHITE, BACKBONE),
            feature_box(1230, 302, 176, 88, "1x1 reduction", "Channel compression", WHITE, ACCENT),
            feature_box(1454, 302, 208, 88, "DWConv encoding", "Local positional cue", WHITE, ACCENT),
            feature_box(1112, 466, 214, 88, "Lightweight MHSA", "Global token interaction", ACCENT_SOFT, ACCENT, accent=True),
            feature_box(1372, 466, 214, 88, "FFN-Lite", "Local feed-forward branch", WHITE, ACCENT),
            feature_box(1240, 646, 220, 88, "Residual output", "Context-enhanced feature", WHITE, ACCENT),
            polyline([(1182, 346), (1230, 346)]),
            polyline([(1406, 346), (1454, 346)]),
            polyline([(1558, 390), (1558, 430), (1219, 430), (1219, 466)]),
            polyline([(1326, 510), (1372, 510)]),
            polyline([(1479, 554), (1479, 602), (1350, 602), (1350, 646)]),
            polyline([(1182, 346), (1182, 602), (1240, 690)], dashed=True),
            text(1350, 784, "Compact global modeling with preserved residual flow", 16, 700, ACCENT, anchor="middle"),
        ]
    )
    save("xr_lite_modules.svg", parts)


def p2_lite_module() -> None:
    parts = svg_start(1320, 760, "P2-Lite High-Resolution Detection Branch")
    parts.extend(
        [
            rect(64, 122, 1192, 560, PANEL, GRID, radius=28, stroke_width=1.5),
            pill(98, 156, 228, "XR-Plus new branch", ACCENT_SOFT, ACCENT, ACCENT),
            feature_box(118, 298, 256, 92, "P3 / 8 semantic feature", "Higher-level detector feature", WHITE, BACKBONE),
            feature_box(118, 454, 256, 92, "P2 / 4 backbone feature", "High-resolution detail", WHITE, BACKBONE),
            feature_box(462, 298, 194, 92, "Upsample x2", "Align to P2 / 4", WHITE, ACCENT),
            feature_box(744, 372, 170, 92, "Concat", "Fuse semantics + detail", ACCENT_SOFT, ACCENT, accent=True),
            feature_box(982, 296, 186, 92, "DWConv 3x3", "Lightweight local fusion", WHITE, ACCENT),
            feature_box(982, 456, 186, 92, "1x1 Conv", "Channel projection", WHITE, ACCENT),
            feature_box(1030, 612, 190, 82, "P2-Lite feature", "stride 4 output", WHITE, HEAD),
            polyline([(374, 344), (462, 344)]),
            polyline([(656, 344), (702, 344), (702, 418), (744, 418)]),
            polyline([(374, 500), (702, 500), (702, 418), (744, 418)], stroke=ACCENT),
            polyline([(914, 418), (944, 418), (944, 342), (982, 342)]),
            polyline([(1075, 388), (1075, 456)]),
            polyline([(1075, 548), (1075, 612)]),
            text(660, 718, "Semantic guidance at P3 enriches high-resolution P2 detection for small objects.", 18, 700, ACCENT, anchor="middle"),
        ]
    )
    save("p2_lite_module.svg", parts)


def main() -> None:
    xr_nano_overall()
    xr_lite_overall()
    xr_lite_modules()
    xr_plus_overall()
    p2_lite_module()


if __name__ == "__main__":
    main()
