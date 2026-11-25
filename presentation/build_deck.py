# -*- coding: utf-8 -*-
"""Build the MM-SDG presentation (local .pptx).

Parts: 1 conditional generation (GANs -> diffusion, conditioning), 2 RNA-CDM + bulk vs spatial,
3 small-model ablation, 4 main paper-scale experiment (hpc_experiment_multicancer).

All numbers come from the experiment READMEs / DECISIONS.md / loss JSONs.
Run:  python build_deck.py   (needs python-pptx, lxml, Pillow)
"""
import json
import os
import tempfile

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MC = os.path.join(REPO, "hpc_experiment_multicancer")
OUT = os.path.join(HERE, "MM-SDG_ablation_and_SDG.pptx")

# ---------------------------------------------------------------- palette
# H&E-inspired: hematoxylin purple dominates, eosin pink is the accent.
INK = "2B1B3D"       # dark slides / titles
HEMA = "4A2C6B"      # primary
EOSIN = "D9507A"     # accent (RNA conditioning, highlights)
CLASS = "9A8CB5"     # class conditioning (muted purple)
TINT = "F3EEF7"      # card tint
TINT2 = "FBEAF0"     # eosin card tint
MUTED = "6B6478"     # captions
TEAL = "2F8F89"      # "done / OK" status
AMBER = "C98A1B"     # "in progress"
WHITE = "FFFFFF"
GRID = "E4DEEA"

HEAD = "Cambria"
BODY = "Calibri"


def upper_gr(s):
    """Greek uppercase drops the tonos."""
    import unicodedata
    out = unicodedata.normalize("NFD", s.upper())
    out = "".join(c for c in out if c not in ("́", "̈"))
    return unicodedata.normalize("NFC", out)


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- helpers
def no_shadow(shape):
    try:
        shape.shadow.inherit = False
    except Exception:
        pass


def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, margin=0,
         space_after=0, line_spacing=None, name=None):
    """runs: str | list of paragraphs; each paragraph is str or list of (text, opts)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, opts in segs:
            r = p.add_run()
            r.text = seg
            f = r.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", italic)
            f.color.rgb = rgb(opts.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, space_after=8, bullet_color=EOSIN):
    """items: list of str or list of (text, opts) runs. Real paragraph bullets."""
    tb = text(slide, x, y, w, h, items, size=size, color=color, space_after=space_after)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25))))
        pPr.set("indent", str(-Emu(Inches(0.25))))
        buClr = etree.SubElement(pPr, qn("a:buClr"))
        clr = etree.SubElement(buClr, qn("a:srgbClr"))
        clr.set("val", bullet_color)
        buFont = etree.SubElement(pPr, qn("a:buFont"))
        buFont.set("typeface", "Arial")
        bu = etree.SubElement(pPr, qn("a:buChar"))
        bu.set("char", "•")
    return tb


def box(slide, x, y, w, h, fill=TINT, radius=0.08, line=None, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1.25)
    else:
        shp.line.fill.background()
    no_shadow(shp)
    if name:
        shp.name = name
    shp.text_frame.text = ""
    return shp


def tile(slide, x, y, s, label, fill=HEMA, fg=WHITE, size=20, name=None):
    """The deck's motif: a rounded square 'tissue tile' holding a number/letter."""
    shp = box(slide, x, y, s, s, fill=fill, radius=0.22, name=name)
    tf = shp.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name = HEAD
    r.font.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = rgb(fg)
    return shp


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def kicker(slide, label, dark=False):
    text(slide, 0.6, 0.32, 9, 0.3, upper_gr(label), size=11, bold=True,
         color=(EOSIN if not dark else "F2A7BF"), name="Kicker")


def title(slide, t, dark=False):
    text(slide, 0.6, 0.6, 12.1, 0.9, t, size=30, font=HEAD, bold=True,
         color=(WHITE if dark else INK), anchor=MSO_ANCHOR.TOP, name="Title")


def footer(slide, n, dark=False):
    text(slide, 11.9, 7.0, 0.85, 0.3, str(n), size=10, color=(MUTED if not dark else "B9AFC8"),
         align=PP_ALIGN.RIGHT, name="Slide number")
    text(slide, 0.6, 7.0, 6, 0.3, "MM-SDG · Synthetic Data Generation", size=10,
         color=(MUTED if not dark else "B9AFC8"), name="Footer")


def style_chart(chart, legend=True, size=12):
    chart.has_title = False
    chart.font.name = BODY
    chart.font.size = Pt(size)
    chart.font.color.rgb = rgb(MUTED)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
        chart.legend.font.color.rgb = rgb(INK)


def quiet_axes(chart, val_max=None, val_min=None, num_fmt=None, gridlines=True):
    va = chart.value_axis
    va.has_major_gridlines = gridlines
    if gridlines:
        va.major_gridlines.format.line.color.rgb = rgb(GRID)
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    if val_max is not None:
        va.maximum_scale = val_max
    if val_min is not None:
        va.minimum_scale = val_min
    if num_fmt:
        va.tick_labels.number_format = num_fmt
        va.tick_labels.number_format_is_linked = False
    ca = chart.category_axis
    ca.format.line.color.rgb = rgb(GRID)
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = rgb(INK)
    ca.has_major_gridlines = False


def color_series(chart, colors, labels=True, num_fmt="0.0", label_size=12, pos=XL_LABEL_POSITION.OUTSIDE_END):
    for s, c in zip(chart.plots[0].series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.format.line.fill.background()
        s.invert_if_negative = False
    if labels:
        pl = chart.plots[0]
        pl.has_data_labels = True
        dl = pl.data_labels
        dl.number_format = num_fmt
        dl.number_format_is_linked = False
        dl.position = pos
        dl.font.size = Pt(label_size)
        dl.font.bold = True
        dl.font.color.rgb = rgb(INK)


def stat(slide, x, y, w, value, label, color=HEMA, vsize=40, lsize=13, dark=False):
    text(slide, x, y, w, 0.75, value, size=vsize, font=HEAD, bold=True, color=color)
    text(slide, x, y + 0.78, w, 0.6, label, size=lsize, color=(MUTED if not dark else "D8D0E3"))


def patch_png(src, dst, size=320):
    im = Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)
    im.save(dst)
    return dst


# ---------------------------------------------------------------- theme
def set_theme(prs):
    """Write the deck's fonts + colours into the theme part (so PowerPoint's
    theme pickers and 'new text box' defaults follow the deck)."""
    master = prs.slide_master
    theme_part = None
    for rel in master.part.rels.values():
        if rel.reltype.endswith("/theme"):
            theme_part = rel.target_part
    root = etree.fromstring(theme_part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    root.set("name", "MM-SDG H&E")
    cs = root.find(".//a:clrScheme", ns)
    cs.set("name", "MM-SDG H&E")
    mapping = {"dk1": INK, "lt1": WHITE, "dk2": HEMA, "lt2": TINT, "accent1": HEMA, "accent2": EOSIN,
               "accent3": CLASS, "accent4": TEAL, "accent5": AMBER, "accent6": MUTED,
               "hlink": EOSIN, "folHlink": CLASS}
    for k, v in mapping.items():
        el = cs.find(f"a:{k}", ns)
        for ch in list(el):
            el.remove(ch)
        s = etree.SubElement(el, qn("a:srgbClr"))
        s.set("val", v)
    fs = root.find(".//a:fontScheme", ns)
    fs.set("name", "MM-SDG")
    fs.find("a:majorFont/a:latin", ns).set("typeface", HEAD)
    fs.find("a:minorFont/a:latin", ns).set("typeface", BODY)
    for tag, face in (("a:majorFont", HEAD), ("a:minorFont", BODY)):
        for f in fs.find(tag, ns).findall("a:font", ns):
            if f.get("script") == "Grek":
                f.set("typeface", face)
    theme_part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


# ---------------------------------------------------------------- build
def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    set_theme(prs)
    blank = prs.slide_layouts[6]
    tmp = tempfile.mkdtemp(prefix="mmsdg_deck_")
    n = 0

    def new(dark=False):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        bg(s, INK if dark else WHITE)
        return s

    # ------------------------------------------------------------ local helpers
    def node(slide, x, y, w, h, label, fill=TINT, fg=INK, size=14, bold=True, radius=0.15, name=None):
        shp = box(slide, x, y, w, h, fill=fill, radius=radius, name=name)
        tf = shp.text_frame
        tf.word_wrap = True
        for side in ("left", "right"):
            setattr(tf, f"margin_{side}", Inches(0.08))
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = label.split("\n")
        for k, ln in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = ln
            r.font.name = BODY
            r.font.size = Pt(size if k == 0 else size - 2)
            r.font.bold = bold if k == 0 else False
            r.font.color.rgb = rgb(fg)
        return shp

    def divider(num, head, sub, fill):
        s = new(dark=True)
        tile(s, 0.6, 2.4, 1.3, num, fill=fill, size=44)
        text(s, 2.25, 2.35, 10.5, 0.9, head, size=40, font=HEAD, bold=True, color=WHITE)
        text(s, 2.25, 3.3, 10.3, 0.9, sub, size=18, color="D8D0E3")
        footer(s, n, dark=True)
        return s

    def noise_strip(src, steps, size=200):
        """Forward diffusion q(x_t | x_0) with a linear beta schedule, on a real patch."""
        import math
        import random
        random.seed(0)
        base = Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)
        raw = base.tobytes()
        px = [v / 127.5 - 1.0 for v in raw]
        betas = [1e-4 + (0.02 - 1e-4) * i / 999 for i in range(1000)]
        abar, prod = [], 1.0
        for b in betas:
            prod *= (1 - b)
            abar.append(prod)
        eps = [random.gauss(0, 1) for _ in px]
        out = []
        for t in steps:
            a = abar[min(t, 999)] if t > 0 else 1.0
            sa, sn = math.sqrt(a), math.sqrt(1 - a)
            vals = [max(0, min(255, int(round((sa * x + sn * e + 1) * 127.5)))) for x, e in zip(px, eps)]
            im = Image.new("RGB", (size, size))
            im.putdata([tuple(vals[i:i + 3]) for i in range(0, len(vals), 3)])
            dst = os.path.join(tmp, f"noise_t{t}.png")
            im.save(dst)
            out.append(dst)
        return out

    reals = sorted(f for f in os.listdir(os.path.join(MC, "preview_step999")) if f.startswith("real_"))

    # ============================================================ 1. Title
    s = new(dark=True)
    text(s, 0.6, 0.9, 10, 0.4, "MM-SDG · SYNTHETIC DATA GENERATION", size=13, bold=True, color="F2A7BF")
    text(s, 0.6, 1.4, 11.5, 1.8, "Synthesizing histopathology images from gene expression",
         size=40, font=HEAD, bold=True, color=WHITE)
    text(s, 0.6, 3.25, 11.5, 1.0,
         "From GANs to conditional diffusion, RNA-CDM, and our spatial-transcriptomics reproduction",
         size=18, color="E6DFEE")
    sz, gap = 1.55, 0.25
    for i, f in enumerate(reals[:6]):
        p = patch_png(os.path.join(MC, "preview_step999", f), os.path.join(tmp, f))
        pic = s.shapes.add_picture(p, Inches(0.6 + i * (sz + gap)), Inches(4.75), Inches(sz), Inches(sz))
        pic.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE
        pic.name = f"Real H&E patch {i + 1}"
    text(s, 0.6, 6.42, 11, 0.3, "Real H&E patches from the multi-cancer dataset (HEST-1k, Visium)",
         size=11, italic=True, color="B9AFC8")
    s.notes_slide.notes_text_frame.text = (
        "We start from the general problem of conditional image generation, move from GANs to diffusion models and "
        "how they are conditioned, then present RNA-CDM, how our spatial setting differs from its bulk RNA-seq setup, "
        "and finally our two experiments: the small ablation and the main paper-scale reproduction.")

    # ============================================================ 2. Roadmap
    s = new()
    kicker(s, "Roadmap")
    title(s, "From conditional generation to our experiments")
    road = [
        ("1", "Conditional generation", "GANs → diffusion models, and how a condition enters a diffusion model", HEMA, TINT),
        ("2", "RNA-CDM", "The paper we build on, and how our spatial data differs from its bulk RNA-seq", HEMA, TINT),
        ("3", "Small ablation", "64×64 models on HEST-bench: conditioning, resolution, capacity", EOSIN, TINT2),
        ("4", "Main experiment", "Paper-scale multi-cancer reproduction: β-VAE → DDPM 64 → SR 256", EOSIN, TINT2),
    ]
    cw = 2.85
    for i, (num, head, body, fill, tint) in enumerate(road):
        x = 0.6 + i * (cw + 0.233)
        box(s, x, 1.95, cw, 4.4, fill=tint, name=f"Roadmap part {num}")
        tile(s, x + 0.3, 2.25, 0.85, num, fill=fill, size=28)
        text(s, x + 0.3, 3.35, cw - 0.55, 0.9, head, size=20, font=HEAD, bold=True)
        text(s, x + 0.3, 4.3, cw - 0.55, 1.9, body, size=15, color=INK, line_spacing=1.05)
    text(s, 0.6, 6.5, 12.1, 0.4, "Parts 1–2: background  ·  Parts 3–4: our work", size=13, italic=True, color=MUTED)
    footer(s, n)

    # ============================================================ PART 1
    divider("1", "Conditional image generation", "From GANs to diffusion models, and how the condition gets in", HEMA)

    # ---------------- conditional generation
    s = new()
    kicker(s, "1 · Problem")
    title(s, "Conditional generation: sample an image x given a condition c")
    text(s, 0.6, 1.8, 6.2, 0.9, [[("Learn  ", {}), ("p(x | c)", {"bold": True, "color": EOSIN, "font": HEAD, "size": 30}),
                                   ("  instead of  p(x)", {})]], size=20, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 2.75, 6.2, 0.8, "The condition steers what is generated; the model fills in plausible detail.",
         size=16, color=MUTED)
    conds = [("Label", "cancer type"), ("Text", "a prompt"), ("Image", "a low-res version"), ("Molecular", "gene expression")]
    for i, (h, b) in enumerate(conds):
        x = 0.6 + i * 1.6
        tile(s, x, 3.75, 1.35, h, fill=EOSIN if i == 3 else HEMA, size=14)
        text(s, x, 5.2, 1.4, 0.6, b, size=13, color=MUTED, align=PP_ALIGN.CENTER)
    box(s, 7.3, 1.85, 5.4, 4.6, fill=TINT)
    text(s, 7.6, 2.05, 4.9, 0.5, "Why condition on RNA in pathology?", size=18, font=HEAD, bold=True)
    bullets(s, 7.6, 2.75, 4.9, 3.6, [
        [("Synthetic data: ", {"bold": True}), ("augment rare cancer types and small cohorts", {})],
        [("Privacy: ", {"bold": True}), ("share realistic images without real patients' slides", {})],
        [("Biology: ", {"bold": True}), ("probe how a molecular profile translates into tissue morphology", {})],
        [("Multimodal link: ", {"bold": True}), ("connect transcriptomics and histology in one model", {})],
    ], size=15, space_after=10)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Unconditional models just produce plausible images. Conditional models let us ask for a specific kind of image. "
        "In our case the condition is a gene-expression profile, which makes the model a bridge between molecular data "
        "and morphology.")

    # ---------------- GANs
    s = new()
    kicker(s, "1 · GANs")
    title(s, "GANs: a generator and a discriminator in a min-max game")
    node(s, 0.6, 2.0, 1.6, 0.8, "z ~ N(0, I)\nnoise", fill=TINT)
    node(s, 0.6, 3.2, 1.6, 0.8, "c\ncondition", fill=TINT2)
    node(s, 2.8, 2.55, 1.8, 1.0, "Generator G", fill=HEMA, fg=WHITE, size=16)
    arrow(s, 2.25, 2.4, 2.75, 2.85)
    arrow(s, 2.25, 3.6, 2.75, 3.25)
    node(s, 5.2, 2.6, 1.6, 0.9, "x̃ = G(z, c)\nfake image", fill=TINT)
    arrow(s, 4.65, 3.05, 5.15, 3.05)
    node(s, 5.2, 4.2, 1.6, 0.9, "x\nreal image", fill=TINT)
    node(s, 5.2, 5.5, 1.6, 0.7, "c", fill=TINT2)
    node(s, 7.4, 4.0, 1.9, 1.0, "Discriminator D", fill=EOSIN, fg=WHITE, size=16)
    arrow(s, 6.85, 3.15, 7.45, 4.05)
    arrow(s, 6.85, 4.65, 7.35, 4.5)
    arrow(s, 6.85, 5.8, 7.45, 4.95)
    node(s, 9.8, 4.1, 1.3, 0.8, "real / fake?", fill=TINT)
    arrow(s, 9.35, 4.5, 9.75, 4.5)
    text(s, 0.6, 6.3, 8.9, 0.5, "Conditional GAN (Mirza & Osindero, 2014): c is fed to both G and D, so D also judges "
         "whether the image matches its condition.", size=13, italic=True, color=MUTED)
    text(s, 9.8, 1.85, 2.9, 0.4, "Strengths", size=15, bold=True, color=TEAL)
    text(s, 9.8, 2.25, 2.9, 1.2, "One forward pass to sample; sharp images (e.g. StyleGAN)", size=13)
    text(s, 9.8, 5.15, 2.9, 0.4, "Weaknesses", size=15, bold=True, color=EOSIN)
    text(s, 9.8, 5.55, 2.9, 0.8, "Unstable training, mode collapse, no likelihood", size=13)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "GANs (Goodfellow et al., 2014) train a generator to fool a discriminator. They sample fast and can be very "
        "sharp, but the adversarial game is unstable and tends to drop modes of the data distribution, which is a "
        "problem when we want to cover the full diversity of tissue.")

    # ---------------- Diffusion
    s = new()
    kicker(s, "1 · Diffusion models")
    title(s, "Diffusion models: learn to undo noise, one step at a time")
    steps_t = [0, 100, 250, 500, 750, 999]
    strip = noise_strip(os.path.join(MC, "preview_step999", reals[1]), steps_t)
    sz, gap = 1.75, 0.32
    for i, (pth, t) in enumerate(zip(strip, steps_t)):
        x = 0.6 + i * (sz + gap)
        pic = s.shapes.add_picture(pth, Inches(x), Inches(1.85), Inches(sz), Inches(sz))
        pic.name = f"Noised patch t={t}"
        text(s, x, 3.65, sz, 0.3, f"t = {t if t < 999 else 'T'}", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    arrow(s, 0.8, 4.1, 12.3, 4.1, color=MUTED)
    text(s, 0.6, 4.15, 6, 0.35, "forward q: add Gaussian noise (fixed)", size=12, italic=True, color=MUTED)
    arrow(s, 12.3, 4.55, 0.8, 4.55, color=EOSIN, width=2.5)
    text(s, 6.7, 4.6, 6.0, 0.35, "reverse p_θ: denoise (learned network)", size=12, italic=True, color=EOSIN,
         align=PP_ALIGN.RIGHT)
    box(s, 0.6, 5.1, 5.95, 1.3, fill=TINT)
    text(s, 0.85, 5.18, 5.5, 1.2, [
        [("Training", {"bold": True, "color": HEMA})],
        [("x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε", {"font": HEAD})],
        [("minimize  ‖ ε − ε_θ(x_t, t, c) ‖²  (a plain regression)", {})],
    ], size=14, space_after=3)
    box(s, 6.75, 5.1, 5.95, 1.3, fill=TINT2)
    text(s, 7.0, 5.18, 5.5, 1.2, [
        [("Sampling", {"bold": True, "color": EOSIN})],
        [("Start from pure noise and apply ε_θ repeatedly: 1,000 steps (DDPM) or ~30 with fast samplers "
          "(DDIM, Karras/EDM).", {})],
    ], size=14, space_after=3)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "The strip is a real H&E patch pushed through the forward process with the linear schedule we use (1,000 steps). "
        "DDPM (Ho et al., 2020) trains a network to predict the added noise; generation runs the process backwards. "
        "The condition c is simply an extra input to the denoiser.")

    # ---------------- GAN vs diffusion
    s = new()
    kicker(s, "1 · GANs vs diffusion")
    title(s, "Why the field moved from GANs to diffusion")
    rows = [("", "GAN", "Diffusion model"),
            ("Training objective", "Adversarial min-max", "Noise-prediction regression (MSE)"),
            ("Stability", "Fragile, needs tricks", "Stable, scales smoothly"),
            ("Mode coverage", "Prone to mode collapse", "Covers the data distribution well"),
            ("Sampling cost", "1 forward pass", "30–1,000 network evaluations"),
            ("Conditioning", "Inputs to G (and D)", "Inputs to ε_θ + optional guidance"),
            ("Image quality", "State of the art until ~2021", "Beats GANs (Dhariwal & Nichol, 2021)")]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.6), Inches(1.85), Inches(12.1), Inches(4.3))
    tbl.name = "GAN vs diffusion table"
    t = tbl.table
    t.columns[0].width = Inches(3.0)
    t.columns[1].width = Inches(4.3)
    t.columns[2].width = Inches(4.8)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = val
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = rgb(CLASS if c == 1 else (EOSIN if c == 2 else HEMA))
            else:
                cell.fill.fore_color.rgb = rgb(TINT if r % 2 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(15)
                    run.font.name = BODY
                    run.font.bold = (r == 0 or c == 0)
                    run.font.color.rgb = rgb(WHITE if r == 0 else (HEMA if c == 0 else INK))
    text(s, 0.6, 6.35, 12.1, 0.45, "For histology, coverage matters: rare morphologies must not be dropped. "
         "The price is slower sampling.", size=14, italic=True, color=MUTED)
    footer(s, n)

    # ---------------- Conditioning mechanisms
    s = new()
    kicker(s, "1 · Conditioning")
    title(s, "Four ways a condition enters a diffusion model")
    mech = [
        ("1", "Timestep embedding", "Project c and add it to the timestep embedding, which modulates every block (FiLM / AdaGN).",
         "Global vectors, class labels · ADM", "Main exp.: latent → time embedding"),
        ("2", "Channel concatenation", "Stack c as extra input channels: a low-res image, or a vector broadcast over pixels.",
         "Spatial conditions · SR3, cascades", "Ablation: label / RNA broadcast; all SR stages"),
        ("3", "Cross-attention", "Image features attend to c as a sequence of tokens inside the UNet.",
         "Text, latents · Latent Diffusion, Imagen", "Main exp.: RNA latent as one token"),
        ("4", "Guidance", "At sampling time, push towards c: classifier guidance or classifier-free guidance.",
         "Fidelity vs diversity knob", "Not used yet"),
    ]
    cw, chh = 5.9, 2.25
    for i, (num, head, body, where, ours) in enumerate(mech):
        x = 0.6 + (i % 2) * (cw + 0.3)
        y = 1.8 + (i // 2) * (chh + 0.25)
        box(s, x, y, cw, chh, fill=TINT)
        tile(s, x + 0.25, y + 0.25, 0.6, num, fill=HEMA if i < 3 else CLASS, size=18)
        text(s, x + 1.05, y + 0.25, 3.0, 0.6, head, size=18, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 4.0, y + 0.25, cw - 4.2, 0.6, where, size=11, italic=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.25, y + 0.98, cw - 0.5, 0.75, body, size=14)
        text(s, x + 0.25, y + 1.75, cw - 0.5, 0.4, [[("Ours: ", {"bold": True, "color": EOSIN}), (ours, {})]], size=13)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "The denoiser ε_θ(x_t, t, c) can receive c in several ways, often combined. Imagen-style models use both a "
        "pooled embedding in the timestep path and cross-attention, which is what RNA-CDM and our main experiment do. "
        "Our small ablation used the simplest option, broadcast-and-concatenate. Guidance is a sampling-time trick "
        "we have not used yet.")

    # ---------------- Cascades
    s = new()
    kicker(s, "1 · Cascaded diffusion")
    title(s, "Cascaded diffusion: generate small, then super-resolve")
    node(s, 0.6, 2.6, 1.6, 1.0, "Noise", fill=TINT)
    arrow(s, 2.25, 3.1, 2.75, 3.1)
    node(s, 2.8, 2.45, 2.4, 1.3, "Stage 1\nbase model, 64×64", fill=HEMA, fg=WHITE, size=16)
    arrow(s, 5.25, 3.1, 5.75, 3.1)
    node(s, 5.8, 2.6, 1.7, 1.0, "Upsample\n+ degrade", fill=TINT)
    arrow(s, 7.55, 3.1, 8.05, 3.1)
    node(s, 8.1, 2.45, 2.4, 1.3, "Stage 2\nsuper-resolution", fill=EOSIN, fg=WHITE, size=16)
    arrow(s, 10.55, 3.1, 11.05, 3.1)
    node(s, 11.1, 2.6, 1.6, 1.0, "256×256\nimage", fill=TINT)
    node(s, 5.2, 4.3, 3.0, 0.75, "condition c (to both stages)", fill=TINT2, size=14)
    arrow(s, 5.6, 4.3, 4.4, 3.8, color=EOSIN)
    arrow(s, 7.8, 4.3, 8.9, 3.8, color=EOSIN)
    box(s, 0.6, 5.35, 12.1, 1.1, fill=TINT)
    text(s, 0.85, 5.42, 11.6, 1.0, [
        [("Why: ", {"bold": True, "color": HEMA}),
         ("each stage solves an easier problem; stage 1 sets layout and content, stage 2 adds texture (Ho et al., 2022).", {})],
        [("Catch: ", {"bold": True, "color": EOSIN}),
         ("stage 2 trains on degraded real images but samples from stage-1 outputs. Noise-conditioning augmentation "
          "or blur on the low-res input narrows that gap.", {})],
    ], size=14, space_after=6)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Both RNA-CDM and our experiments use a two-stage cascade. Keep the train/sample mismatch of stage 2 in mind: "
        "it is exactly what hurt our small cascade in the ablation.")

    # ============================================================ PART 2
    divider("2", "RNA-CDM", "Cascaded diffusion from RNA-seq, and how our spatial setting differs", HEMA)

    # ---------------- RNA-CDM overview
    s = new()
    kicker(s, "2 · RNA-CDM")
    title(s, "RNA-CDM: synthetic tumour tiles from RNA-seq")
    text(s, 0.6, 1.65, 7, 0.4, "Carrillo-Pérez et al., Nature Biomedical Engineering", size=13, italic=True, color=MUTED)
    bullets(s, 0.6, 2.2, 6.7, 4.4, [
        [("Goal: ", {"bold": True}), ("generate realistic H&E tiles from a tumour's transcriptome", {})],
        [("One model, 5 cancer types: ", {"bold": True}), ("LUAD, KIRP, CESC, COAD, GBM (TCGA)", {})],
        [("Data: ", {"bold": True}), ("bulk RNA-seq paired with each patient's diagnostic WSI", {})],
        [("Genes: ", {"bold": True}), ("17,655 shared across the 5 types after filtering", {})],
        [("Tiles: ", {"bold": True}), ("256 px at 0.5 µm/px, at most 4,000 random tiles per slide", {})],
        [("Split: ", {"bold": True}), ("60 / 20 / 20 train / val / test", {})],
    ], size=16, space_after=10)
    for i, (v, l, c) in enumerate([("1.8B", "parameters in total", HEMA), ("23.36", "FID50k", EOSIN),
                                   ("79k", "training steps", HEMA), ("~8 days", "of training, 4 GPUs", EOSIN)]):
        cx, cy = 7.7 + (i % 2) * 2.6, 1.95 + (i // 2) * 2.3
        box(s, cx, cy, 2.35, 2.0, fill=TINT if c == HEMA else TINT2)
        stat(s, cx + 0.25, cy + 0.3, 2.0, v, l, color=c, vsize=32)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "RNA-CDM is the reference we reproduce. Its key idea: compress bulk RNA-seq into a latent with a β-VAE and use "
        "that latent to condition a cascaded diffusion model, one model across five cancer types.")

    # ---------------- RNA-CDM architecture
    s = new()
    kicker(s, "2 · RNA-CDM architecture")
    title(s, "β-VAE latent → 64×64 base model → 256×256 super-resolution")
    arch = [
        ("β-VAE", "17,655 → 6000 → 4000\n→ latent 200 (~282M)"),
        ("Stage 1 DDPM", "64×64 · cross-attention\non the latent (dim 256)"),
        ("Gaussian blur", "applied to the\nstage-1 output"),
        ("Stage 2 SR", "256×256 · cross-attention\n(dim 128)"),
    ]
    w, gap, y = 2.65, 0.5, 2.0
    for i, (head, body) in enumerate(arch):
        x = 0.6 + i * (w + gap)
        fill = HEMA if i in (0, 1, 3) else TINT
        node(s, x, y, w, 1.6, f"{head}\n{body}", fill=fill, fg=WHITE if fill == HEMA else INK, size=17)
        if i < 3:
            arrow(s, x + w + 0.07, y + 0.8, x + w + gap - 0.07, y + 0.8)
    chips = ["Linear schedule\n1,000 timesteps", "Batch 256\n64 per GPU × 4 GPUs", "EMA weights\nused for sampling",
             "Karras sampler\n32 steps", "VAE: BatchNorm +\nLeakyReLU, β = 0.005"]
    chw = (12.1 - 4 * 0.2) / 5
    for k, c in enumerate(chips):
        node(s, 0.6 + k * (chw + 0.2), 4.0, chw, 0.85, c, fill=TINT2, size=14, radius=0.3)
    box(s, 0.6, 5.3, 12.1, 1.15, fill=TINT)
    text(s, 0.85, 5.38, 11.6, 1.0, [
        [("Reported quality: ", {"bold": True, "color": HEMA}),
         ("FID50k = 23.36 on 10k tiles per type; β-VAE reconstruction RMSE 0.2475, MAE 0.1471.", {})],
        [("Conditioning style: ", {"bold": True, "color": HEMA}),
         ("Imagen-like: cross-attention plus the pooled latent in the timestep path (mechanisms 1 + 3).", {})],
    ], size=14, space_after=6)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "These are the settings we match in our main experiment. The β-VAE is trained first and frozen; both diffusion "
        "stages are conditioned on its latent.")

    # ---------------- Bulk vs spatial (visual)
    s = new()
    kicker(s, "2 · Our setting")
    title(s, "Bulk RNA-seq vs spatial transcriptomics")
    import random as _rnd
    _rnd.seed(3)
    spot_cols = [HEMA, EOSIN, CLASS, TEAL, AMBER, "7B5EA7", "E58FA9", "5E4B7A"]
    for side in range(2):
        x0 = 0.6 + side * 6.25
        box(s, x0, 1.8, 5.85, 4.55, fill=TINT if side == 0 else TINT2)
        text(s, x0 + 0.3, 1.95, 5.3, 0.5, ["RNA-CDM: bulk RNA-seq (TCGA)", "Ours: Visium spots (HEST-1k)"][side],
             size=18, font=HEAD, bold=True)
        g0x, g0y, cell, gp = x0 + 0.35, 2.75, 0.5, 0.08
        for r in range(5):
            for c in range(5):
                col = CLASS if side == 0 else _rnd.choice(spot_cols)
                sq = box(s, g0x + c * (cell + gp), g0y + r * (cell + gp), cell, cell, fill=col, radius=0.15)
                sq.name = f"{'bulk' if side == 0 else 'spot'} tile {r}-{c}"
        if side == 0:
            node(s, x0 + 3.65, 3.55, 1.9, 1.15, "1 profile\nper patient", fill=CLASS, fg=WHITE, size=15)
            arrow(s, x0 + 3.6, 4.12, x0 + 3.3, 4.12, color=INK)
            text(s, x0 + 0.35, 5.7, 5.2, 0.6, "Every tile of a slide shares the same condition vector.",
                 size=13, color=INK)
        else:
            node(s, x0 + 3.65, 3.55, 1.9, 1.15, "1 profile\nper 55 µm spot", fill=EOSIN, fg=WHITE, size=15)
            arrow(s, x0 + 3.6, 4.12, x0 + 3.3, 4.12, color=INK)
            text(s, x0 + 0.35, 5.7, 5.2, 0.6, "Each 128 µm tile is centred on a spot and gets its own vector.",
                 size=13, color=INK)
    text(s, 0.6, 6.5, 12.1, 0.4, "Same model family, a different question: local expression → local morphology.",
         size=14, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "This is the core difference. In RNA-CDM the condition is a patient-level average: thousands of tiles share one "
        "vector, so the model learns a slide-level style. With Visium each tile has its own spot's expression, so the "
        "model must tie local gene expression to local tissue. Harder, but closer to the biology.")

    # ---------------- Bulk vs spatial (table)
    s = new()
    kicker(s, "2 · Our setting")
    title(s, "What changes when the condition is a spot, not a patient")
    rows = [("", "RNA-CDM (bulk)", "Ours (spatial)"),
            ("Expression source", "TCGA bulk RNA-seq", "HEST-1k, 10x Visium"),
            ("Unit of expression", "One profile per patient / slide", "One profile per 55 µm spot"),
            ("Tile ↔ condition", "Many tiles : one vector", "One tile : one vector"),
            ("Signal", "Smooth, averaged over the tumour", "Sparse, zero-inflated (max |z| = 106.9)"),
            ("Genes", "17,655 shared", "14,368 shared + QC"),
            ("Cancer types", "LUAD, KIRP, CESC, COAD, GBM", "COAD, SCCRCC, EPM, PRAD, IDC"),
            ("Split", "Patient-level", "Patient-level where IDs exist, else WSI-level"),
            ("β-VAE RMSE", "0.25", "0.94")]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.6), Inches(1.75), Inches(12.1), Inches(4.6))
    tbl.name = "Bulk vs spatial table"
    t = tbl.table
    t.columns[0].width = Inches(2.9)
    t.columns[1].width = Inches(4.3)
    t.columns[2].width = Inches(4.9)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = val
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = rgb(CLASS if c == 1 else (EOSIN if c == 2 else HEMA))
            else:
                cell.fill.fore_color.rgb = rgb(TINT if r % 2 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(14)
                    run.font.name = BODY
                    run.font.bold = (r == 0 or c == 0)
                    run.font.color.rgb = rgb(WHITE if r == 0 else (HEMA if c == 0 else INK))
    text(s, 0.6, 6.45, 12.1, 0.4, "The paper's cancer types barely exist in HEST's Visium data (KIRP 0, CESC 1, GBM 3–4 samples).",
         size=13, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Consequences: our numbers are not directly comparable to the paper's. The VAE reconstruction error is higher "
        "because spot-level data is much noisier than bulk, and the cancer types had to change because the paper's "
        "types are not available as human Visium samples in HEST.")

    # ============================================================ PART 3
    divider("3", "Small ablation", "64×64 models on HEST-bench: conditioning, resolution, capacity",
            EOSIN)

    # ---------------- Baseline
    s = new()
    kicker(s, "3 · Baseline")
    title(s, "Baseline: class-conditioned DDPM, 64×64, 10 cancer types")
    bullets(s, 0.6, 1.8, 6.3, 4.8, [
        [("Data: ", {"bold": True}), ("HEST-bench, pooled patches from 10 types (CCRCC, COAD, HCC, IDC, LUNG, …)", {})],
        [("Model: ", {"bold": True}), ("UNet with class embedding (broadcast + concat), 64×64, lr 1e-4", {})],
        [("Evaluation: ", {"bold": True}), ("FID on 1,000 synthetic + 1,000 real images per type", {})],
        [("Infrastructure: ", {"bold": True}), ("HUA HPC, 5h job limit → auto-resume for training and sampling", {})],
        [("Note: ", {"bold": True}), ("lr 1e-3 collapsed to predicting zero noise (MSE ≈ 1.0); lr 1e-4 used", {})],
    ], size=16, space_after=12)
    for i, (v, l, c) in enumerate([("236,495", "patches (~35 GB)", HEMA), ("113.7M", "UNet parameters", HEMA),
                                   ("137.3", "FID pooled (10 types)", EOSIN), ("184.4", "FID CCRCC", EOSIN)]):
        cx, cy = 7.5 + (i % 2) * 2.75, 1.85 + (i // 2) * 2.3
        box(s, cx, cy, 2.5, 2.0, fill=TINT if c == HEMA else TINT2)
        stat(s, cx + 0.25, cy + 0.3, 2.1, v, l, color=c, vsize=32)
    text(s, 7.5, 6.3, 5.25, 0.45, "Paper (RNA-CDM): FID50k = 23.36 with a 1.8B-parameter cascaded model",
         size=12, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Baseline: 114M parameters at 64×64, class-conditioned on 10 cancer types. FID 137.3 pooled, 184.4 on CCRCC.")

    # ---------------- Three axes
    s = new()
    kicker(s, "3 · Design")
    title(s, "Three ablation axes, one factor at a time")
    cards = [
        ("1", "Conditioning", "Cancer-type label  vs  RNA expression",
         ["β-VAE: 50 genes → 16-dim latent", "CCRCC, official fold 0 (20 train / 4 held-out WSIs)",
          "Paired sampling: one image per held-out spot"]),
        ("2", "Resolution", "64×64  vs  224×224 cascade",
         ["Stage 2 SR: low-res + condition → 224×224", "SR UNet 28.5M, 64 base channels (budget)",
          "FID resize 1.33× instead of 4.67× (64→299)"]),
        ("3", "Capacity", "UNet 114M  vs  454M",
         ["Only block_out_channels ×2", "Same data and hyperparameters",
          "Sampling 200/type due to time budget"]),
    ]
    for i, (num, head, sub, items) in enumerate(cards):
        x = 0.6 + i * 4.1
        box(s, x, 1.85, 3.8, 4.25, fill=TINT)
        tile(s, x + 0.3, 2.15, 0.7, num, size=22)
        text(s, x + 1.2, 2.15, 2.5, 0.7, head, size=20, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, 3.05, 3.3, 0.5, sub, size=15, bold=True, color=EOSIN)
        bullets(s, x + 0.3, 3.65, 3.3, 2.4, items, size=14, space_after=8)
    box(s, 0.6, 6.3, 12.1, 0.55, fill=TINT2, radius=0.3)
    text(s, 0.85, 6.3, 11.8, 0.55, [[("Evaluated on every axis:  ", {"bold": True, "color": EOSIN}),
                                     ("FID (realism)  ·  classifier alignment (image → type)  ·  regressor alignment (image → expression)", {})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Each experiment changes one factor relative to the baseline. Conditioning here is broadcast-and-concat "
        "(mechanism 2). In the RNA arm each patch is conditioned on its own spot's expression.")

    # ---------------- FID conditioning x resolution
    s = new()
    kicker(s, "3 · FID (CCRCC)")
    title(s, "FID on CCRCC: class vs RNA conditioning, 64×64 and 224×224")
    cd = CategoryChartData()
    cd.categories = ["64×64", "224×224 cascade"]
    cd.add_series("Class label", (184.4, 282.86))
    cd.add_series("RNA (β-VAE latent)", (167.5, 267.25))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.75), Inches(12.1), Inches(4.8), cd)
    gf.name = "FID chart conditioning x resolution"
    ch = gf.chart
    style_chart(ch)
    quiet_axes(ch, val_max=320, val_min=0, num_fmt="0")
    color_series(ch, [CLASS, EOSIN], num_fmt="0.0")
    ch.plots[0].gap_width = 70
    ch.plots[0].overlap = -10
    text(s, 0.6, 6.6, 12.1, 0.4, "FID, CCRCC only (lower = better). Class: all WSIs. RNA: held-out WSIs (fold 0). "
         "Pooled class-conditioned FID: 137.3 (64×64), 155.05 (224×224).", size=12, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "CCRCC only, because the RNA-conditioned models are CCRCC-only. The two protocols differ: the RNA models are "
        "evaluated on held-out WSIs.")

    # ---------------- Conditioning fidelity
    s = new()
    kicker(s, "3 · Conditioning fidelity")
    title(s, "Conditioning fidelity: classifier and regressor results")
    cd = CategoryChartData()
    cd.categories = ["64×64", "224×224 cascade"]
    cd.add_series("Class label", (0.284, 0.250))
    cd.add_series("RNA", (0.311, 0.2585))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.25), Inches(5.9), Inches(4.0), cd)
    gf.name = "Classifier accuracy chart"
    ch = gf.chart
    style_chart(ch)
    quiet_axes(ch, val_max=0.4, val_min=0, num_fmt="0.0")
    ch.value_axis.major_unit = 0.1
    color_series(ch, [CLASS, EOSIN], num_fmt="0.00")
    ch.plots[0].gap_width = 70
    text(s, 0.6, 1.75, 5.9, 0.45, "Classifier: image → cancer type (CCRCC accuracy)",
         size=15, bold=True, color=INK)
    text(s, 0.6, 6.25, 5.9, 0.5, "Accuracy on real images: 0.92 (64×64) / 0.93 (224×224).",
         size=11, italic=True, color=MUTED)
    cd = CategoryChartData()
    cd.categories = ["Real images", "RNA 64×64", "RNA cascade 224"]
    cd.add_series("MAE", (0.853, 1.093, 1.274))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.9), Inches(2.25), Inches(5.8), Inches(4.0), cd)
    gf.name = "Regressor MAE chart"
    ch = gf.chart
    style_chart(ch, legend=False)
    quiet_axes(ch, val_max=1.5, val_min=0, num_fmt="0.0")
    color_series(ch, [EOSIN], num_fmt="0.00")
    ch.plots[0].gap_width = 70
    pts = ch.plots[0].series[0].points
    for idx, c in enumerate([MUTED, EOSIN, HEMA]):
        pts[idx].format.fill.solid()
        pts[idx].format.fill.fore_color.rgb = rgb(c)
    text(s, 6.9, 1.75, 5.8, 0.45, "Regressor: image → 50-gene expression (MAE, z-scored)",
         size=15, bold=True, color=INK)
    text(s, 6.9, 6.25, 5.8, 0.5, "MAE on real images: 0.853 (64×64) / 0.851 (224×224). Lower = better.",
         size=11, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Two independent networks trained only on real data. The classifier predicts cancer type from the image; "
        "the regressor predicts the 50-gene expression vector. Both are applied to the synthetic images.")

    # ---------------- Capacity
    s = new()
    kicker(s, "3 · Capacity")
    title(s, "Capacity: UNet 454M vs 114M")
    rows = [("", "Small", "Large"), ("Parameters", "113.7M", "454.5M"),
            ("block_out_channels", "128 … 512", "256 … 1024"),
            ("Training / epoch (estimate)", "~11 min", "~40 min"),
            ("Samples for FID", "1,000 / type", "200 / type")]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.6), Inches(1.9), Inches(6.4), Inches(2.6))
    tbl.name = "Small vs large table"
    t = tbl.table
    t.columns[0].width = Inches(2.8)
    t.columns[1].width = Inches(1.8)
    t.columns[2].width = Inches(1.8)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(HEMA if r == 0 else (TINT if r % 2 else WHITE))
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.size = Pt(14)
                    run.font.name = BODY
                    run.font.bold = (r == 0 or c == 2)
                    run.font.color.rgb = rgb(WHITE if r == 0 else INK)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cd = CategoryChartData()
    cd.categories = ["8", "16", "32", "64"]
    cd.add_series("ms / step", (167, 262, 535, 1070))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.4), Inches(2.25), Inches(5.3), Inches(3.3), cd)
    gf.name = "Large model local benchmark chart"
    ch = gf.chart
    style_chart(ch, legend=False)
    quiet_axes(ch, val_min=0, num_fmt="#,##0")
    color_series(ch, [CLASS], num_fmt="#,##0")
    ch.plots[0].series[0].points[3].format.fill.solid()
    ch.plots[0].series[0].points[3].format.fill.fore_color.rgb = rgb(EOSIN)
    ch.plots[0].gap_width = 60
    text(s, 7.4, 1.75, 5.3, 0.45, "Local benchmark (RTX 5070 Ti 16 GB), ms/step by batch size",
         size=15, bold=True)
    text(s, 7.4, 5.55, 5.3, 0.9,
         "8/16/32 measured; 64 is extrapolated from that trend (~3.6×, matching the 4× param "
         "count) -- the raw batch=64 reading (18,652ms) was a VRAM-exhaustion artifact, not real "
         "compute cost. Large model, measured locally; the cluster (A100 40 GB) was not benchmarked.",
         size=12, italic=True, color=MUTED)
    box(s, 0.6, 4.8, 6.4, 1.85, fill=TINT2)
    text(s, 0.85, 4.9, 5.95, 1.65, [
        [("Sample sizes differ", {"bold": True, "color": EOSIN})],
        [("FID samples: 2,000 (large) vs 10,000 (small). "
          "Large-model FID: to be added.", {})],
    ], size=14, space_after=4, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "TODO: add the large model's FID here (not recorded in the repo). "
        "The large model differs from the small one only in channel width. Sampling was 200 images/type because of the "
        "5h job limit.")

    # ============================================================ PART 4
    divider("4", "Main experiment", "Paper-scale multi-cancer reproduction of RNA-CDM on spatial data", EOSIN)

    # ---------------- Pipeline
    s = new()
    kicker(s, "4 · Pipeline")
    title(s, "From a Visium spot's expression to a 256×256 image")
    steps = [
        ("Expression", "14,368 genes per spot (shared across 5 types + QC)", "data", MUTED),
        ("β-VAE", "6000 → 4000 → latent 200\n~220M parameters", "trained", TEAL),
        ("Stage 1 DDPM", "64×64, cross-attention\non the latent (dim 256)", "training", AMBER),
        ("Stage 2 SR", "blur(64×64) + latent → 256×256 (dim 128)", "code ready", CLASS),
    ]
    w, gap, y = 2.65, 0.5, 2.15
    for i, (head, body, status, scol) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        box(s, x, y, w, 2.1, fill=TINT2 if i in (1, 2, 3) else TINT, name=f"Pipeline step {i + 1}")
        tile(s, x + 0.25, y + 0.25, 0.6, str(i + 1), fill=EOSIN if i else HEMA, size=18)
        text(s, x + 1.0, y + 0.25, w - 1.1, 0.6, head, size=18, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.25, y + 1.05, w - 0.45, 1.3, body, size=14, color=INK, line_spacing=1.05)
        b = box(s, x + 0.25, y + 2.4, w - 0.5, 0.45, fill=scol, radius=0.5)
        tf = b.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = status
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.name = BODY
        r.font.color.rgb = rgb(WHITE)
        if i < 3:
            arrow(s, x + w + 0.07, y + 1.05, x + w + gap - 0.07, y + 1.05)
    text(s, 0.6, 5.6, 12.1, 0.6, "Same architecture as RNA-CDM, but every tile is conditioned on its own spot. At sampling "
         "time the stage-1 output is Gaussian-blurred and fed to stage 2.", size=14, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Three models in sequence, as in RNA-CDM. The VAE is trained and evaluated, stage 1 is training now on anakin, "
        "and stage 2 is implemented but has not run yet.")

    # ---------------- Data
    s = new()
    kicker(s, "4 · Data")
    title(s, "5 cancer types, 47 WSIs, 154,866 patches")
    cd = CategoryChartData()
    cd.categories = ["IDC", "COAD", "SCCRCC", "EPM", "PRAD"]
    cd.add_series("Patches", (35041, 34013, 31564, 27515, 26733))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.8), Inches(6.2), Inches(4.6), cd)
    gf.name = "Patches per cancer type chart"
    ch = gf.chart
    style_chart(ch, legend=False)
    quiet_axes(ch, val_min=0, val_max=42000, num_fmt="#,##0")
    color_series(ch, [HEMA], num_fmt="#,##0")
    ch.category_axis.reverse_order = True
    ch.plots[0].gap_width = 55
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    text(s, 0.6, 6.4, 6.2, 0.4, "Patches per type after the 4,000/slide cap (WSIs: 10/10/10/10/7); 1.31× range",
         size=11, italic=True, color=MUTED)
    items = [
        ("Types chosen by availability", "The 5 types with the most human Visium samples in HEST-1k."),
        ("Bug 1: LIHB was 100% mouse", "0 shared genes. Replaced by PRAD; species = Homo sapiens filter everywhere."),
        ("Bug 2: one targeted panel (TENX28)", "1,142 genes collapsed the intersection to 1,064 → --min-genes 15,000 → 14,368."),
    ]
    for i, (h, b) in enumerate(items):
        y = 1.85 + i * 1.55
        box(s, 7.2, y, 5.5, 1.35, fill=TINT2 if i else TINT)
        text(s, 7.45, y + 0.12, 5.05, 0.4, h, size=15, bold=True, color=EOSIN if i else HEMA)
        text(s, 7.45, y + 0.52, 5.05, 0.8, b, size=13, color=INK)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Both issues were found while building the gene panel: the cross-type gene intersection came out at 0, "
        "then at 1,064 genes.")

    # ---------------- Decisions
    s = new()
    kicker(s, "4 · Decisions")
    title(s, "Data and training decisions vs the paper")
    grid = [
        ("128 µm → 256 px", "0.5 µm/px as in the paper; PRAD/EPM get upsampled ~1.5–1.8× (accepted limitation)."),
        ("≤ 4,000 tiles / slide", "Uniform random subsample with seed 0, matching the paper's wording."),
        ("60 / 20 / 20 split", "Patient-level for SCCRCC/EPM; WSI-level fallback for COAD/PRAD/IDC (no patient IDs)."),
        ("Weighted sampler", "Inverse type frequency instead of discarding data to fix the 1.31× imbalance."),
        ("Genes: shared + QC, no HVG", "HVG keeps 2,913 genes but only 22% of variance; QC keeps 99.9% of UMIs."),
        ("VAE recipe", "BatchNorm + LeakyReLU, β = 0.005, lr 3e-3 with warmup + cosine, early stopping."),
    ]
    cw, chh = 3.85, 2.2
    for i, (h, b) in enumerate(grid):
        x = 0.6 + (i % 3) * (cw + 0.275)
        y = 1.85 + (i // 3) * (chh + 0.3)
        box(s, x, y, cw, chh, fill=TINT)
        tile(s, x + 0.25, y + 0.25, 0.5, str(i + 1), fill=EOSIN if i % 2 else HEMA, size=15)
        text(s, x + 0.9, y + 0.22, cw - 1.05, 0.55, h, size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.25, y + 0.95, cw - 0.45, 1.2, b, size=13, color=INK)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Every decision and its rationale is logged in DECISIONS.md.")

    # ---------------- VAE
    s = new()
    kicker(s, "4 · VAE")
    title(s, "β-VAE: training curve and test reconstruction")
    losses = json.load(open(os.path.join(MC, "vae_multicancer_losses.json")))
    cd = CategoryChartData()
    cd.categories = [str(i) for i in range(len(losses["train"]))]
    cd.add_series("Train", [round(v, 4) for v in losses["train"]])
    cd.add_series("Validation", [round(v, 4) for v in losses["val"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.6), Inches(1.75), Inches(7.0), Inches(4.65), cd)
    gf.name = "VAE loss curve chart"
    ch = gf.chart
    style_chart(ch)
    lo = min(min(losses["train"]), min(losses["val"]))
    hi = max(max(losses["train"]), max(losses["val"]))
    quiet_axes(ch, val_min=round(lo - 0.01, 2), val_max=round(hi + 0.01, 2), num_fmt="0.00")
    ch.category_axis.tick_labels.font.size = Pt(10)
    cax = ch.category_axis._element
    anchor_el = cax.find(qn("c:noMultiLvlLbl"))
    for tag, val in (("c:tickLblSkip", "5"), ("c:tickMarkSkip", "5")):
        el = etree.Element(qn(tag))
        el.set("val", val)
        if anchor_el is not None:
            anchor_el.addprevious(el)
        else:
            cax.append(el)
    for ser, c in zip(ch.plots[0].series, [CLASS, EOSIN]):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(2.25)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.NONE
    text(s, 0.6, 6.4, 7.0, 0.4, "Loss per epoch (run 2, batch 128). Best val 0.8482 at epoch 4; early stop at 44.",
         size=11, italic=True, color=MUTED)
    x0 = 8.1
    stat(s, x0, 1.85, 2.2, "0.936", "RMSE (paper 0.248)", color=EOSIN, vsize=34)
    stat(s, x0 + 2.4, 1.85, 2.2, "0.514", "MAE (paper 0.147)", color=EOSIN, vsize=34)
    box(s, x0, 3.55, 4.6, 2.85, fill=TINT)
    text(s, x0 + 0.25, 3.7, 4.1, 2.6, [
        [("Training runs", {"bold": True, "color": HEMA})],
        [("Run 1: batch 256, early stop at epoch 27, best val 0.8576 (epoch 7).", {})],
        [("Run 2: batch 128, early stop at epoch 44, best val 0.8482 (epoch 4).", {})],
        [("Checkpoint used downstream: run 2, epoch 4.", {})],
        [("Test-set z-scores: 0.45% of values have |z| > 5, max |z| = 106.9.", {"color": MUTED})],
    ], size=13, space_after=6)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Two runs (batch 256 and 128, different LR schedules). Test-set RMSE 0.9355 and MAE 0.5136; the paper reports "
        "0.2475 and 0.1471 on bulk data.")

    # ---------------- Stage 1 recipe
    s = new()
    kicker(s, "4 · Stage 1")
    title(s, "Stage 1: training recipe")
    recipe = [("Conditioning", "Cross-attention (dim 256) + latent in the timestep embedding"),
              ("Noise schedule", "Linear, 1,000 timesteps"),
              ("Batch / budget", "256 (= the paper's 4 × 64) · 79k steps"),
              ("EMA", "decay 0.9995 with warmup min(d, (t+1)/(t+10))"),
              ("Sampler", "Karras Alg. 2, 32 steps, S_churn = 80, ρ = 7")]
    tbl = s.shapes.add_table(len(recipe), 2, Inches(0.6), Inches(1.9), Inches(5.6), Inches(3.9))
    tbl.name = "Stage 1 recipe table"
    t = tbl.table
    t.columns[0].width = Inches(1.75)
    t.columns[1].width = Inches(3.85)
    for r, (k, v) in enumerate(recipe):
        for c, val in enumerate((k, v)):
            cell = t.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(TINT if r % 2 == 0 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(14)
                    run.font.name = BODY
                    run.font.bold = (c == 0)
                    run.font.color.rgb = rgb(HEMA if c == 0 else INK)
    pic = s.shapes.add_picture(os.path.join(MC, "preview_step999_compare.png"), Inches(6.6), Inches(1.85),
                               width=Inches(6.1))
    pic.name = "Step 999 real vs EMA vs raw samples"
    ph = pic.height / 914400
    text(s, 6.6, 1.85 + ph + 0.1, 6.1, 0.9,
         "Step 999 samples: real (top), EMA weights (middle), raw weights (bottom). The EMA warmup schedule was "
         "added after this check.", size=12, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Settings matched to the paper's Methods. The sampler is the stochastic Algorithm 2 from Karras et al. "
        "The EMA decay rate is not stated in the paper; 0.9995 is our choice.")

    # ---------------- Stage 1 FID
    s = new()
    kicker(s, "4 · Stage 1 FID")
    title(s, "Stage 1: FID at the first checkpoints (training in progress)")
    cd = CategoryChartData()
    cd.categories = ["999", "1,999", "5,999", "8,999", "15,999"]
    cd.add_series("FID", (201.41, 164.66, 88.40, 98.44, 43.82))
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.75), Inches(12.1), Inches(4.55), cd)
    gf.name = "Stage 1 FID progress chart"
    ch = gf.chart
    style_chart(ch, legend=False)
    quiet_axes(ch, val_min=0, val_max=240, num_fmt="0")
    ser = ch.plots[0].series[0]
    ser.format.line.color.rgb = rgb(EOSIN)
    ser.format.line.width = Pt(2.75)
    ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.CIRCLE
    ser.marker.size = 10
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = rgb(EOSIN)
    ser.marker.format.line.color.rgb = rgb(WHITE)
    pl = ch.plots[0]
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.0"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.ABOVE
    pl.data_labels.font.size = Pt(12)
    pl.data_labels.font.bold = True
    pl.data_labels.font.color.rgb = rgb(INK)
    text(s, 0.6, 6.35, 12.1, 0.45, "Early checkpoints of a run that is still training (target 79k steps). 5,000 paired "
         "samples, raw weights, same sampler settings at every checkpoint. Not the paper's FID50k protocol.",
         size=12, italic=True, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Checkpoints from the first part of stage-1 training. The run continues; these values will be updated.")

    # ---------------- Infra
    s = new()
    kicker(s, "4 · Infrastructure")
    title(s, "Graceful shutdown at SLURM's 5h limit")
    tl = [
        ("1", "USR1 @ 300 s", "5-minute warning. No checkpoint saved; the job FAILED with no traceback.", CLASS, "failed"),
        ("2", "USR1 @ 900 s + workers", "15 minutes, plus DataLoader workers ignoring the signal. Same failure.", CLASS, "failed"),
        ("3", "while kill -0; wait", "The trap interrupted wait, and set -e killed the script while Python was saving.", TEAL, "fixed"),
    ]
    y = 2.0
    for i, (num, head, body, col, status) in enumerate(tl):
        x = 0.6 + i * 4.1
        box(s, x, y, 3.8, 3.4, fill=TINT)
        tile(s, x + 0.3, y + 0.3, 0.7, num, fill=TEAL if i == 2 else HEMA, size=22)
        text(s, x + 1.2, y + 0.3, 2.45, 0.7, head, size=16, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, y + 1.25, 3.25, 1.6, body, size=14, color=INK)
        b = box(s, x + 0.3, y + 2.8, 1.6, 0.4, fill=col, radius=0.5)
        tf = b.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = status
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.name = BODY
        r.font.color.rgb = rgb(WHITE)
        if i < 2:
            arrow(s, x + 3.85, y + 1.7, x + 4.05, y + 1.7)
    text(s, 0.6, 5.8, 12.1, 0.8, "Verified with manual scancel --signal=USR1 tests and on two real overnight 5h cutoffs, "
         "with no lost steps.", size=15, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Training runs across chained jobs because of the 5h limit per job.")

    # ============================================================ Next steps
    s = new(dark=True)
    text(s, 0.6, 0.32, 9, 0.3, "NEXT STEPS", size=11, bold=True, color="F2A7BF")
    title(s, "Next steps", dark=True)
    nxt = [
        ("Finish stage 1", "Train to 79k steps; evaluate with EMA weights."),
        ("Start stage 2 (SR 256×256)", "Validate batch 64, blur kernel 9 / σ 3 and memory on anakin."),
        ("Full cascade", "First end-to-end sampling run, evaluated with the FID50k protocol."),
        ("Open question", "Is the paper's 79k steps per stage or in total?"),
    ]
    for i, (h, b) in enumerate(nxt):
        x = 0.6 + (i % 2) * 6.2
        y = 1.9 + (i // 2) * 2.35
        tile(s, x, y, 0.8, str(i + 1), fill=EOSIN if i < 3 else CLASS, size=24)
        text(s, x + 1.05, y - 0.02, 4.85, 0.5, h, size=19, font=HEAD, bold=True, color=WHITE)
        text(s, x + 1.05, y + 0.5, 4.85, 1.3, b, size=15, color="D8D0E3")
    footer(s, n, dark=True)
    s.notes_slide.notes_text_frame.text = (
        "Remaining work for the main experiment.")

    prs.save(OUT)
    print("Saved", OUT, "slides:", n)


if __name__ == "__main__":
    main()
