"""MedBox Reader — local demo (Gradio Blocks), v2 by default, v1 selectable for comparison.

    python app.py      # then open the local URL shown in the terminal

What the user sees, in reading order:
  1. a verdict card: trade name (green), molecule only (blue, v2), or refused (orange), with the reason;
  2. a 4-step tracker showing where the image stopped;
  3. what the model saw: the photo with the boxes, the exact crop given to TrOCR, and the scores
     drawn against the thresholds.
Same readers as the evaluation (one version of the code). No image is stored.
Colours: soft, low-glare palette with a light AND a dark version (follows the system setting), WCAG-readable text.
"""

import random
from html import escape
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from medbox.reader import Reader, load_config
from medbox.reader_v2 import ReaderV2, load_v2_config

READER = Reader(load_config())          # v1 (YOLO); models are loaded once, on the first request
THRESHOLD = READER.config["threshold"]
READER_V2 = ReaderV2(load_v2_config())  # v2 (text lines + tallest text + brands and DCI)
T_BRAND, T_DCI = READER_V2.config["T_brand"], READER_V2.config["T_dci"]

# ---- theme: muted teal accent, off-white / deep-slate surfaces (no pure white, no pure black) -----------
THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="slate", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f4f6f8", body_background_fill_dark="#111827",
    block_background_fill="#ffffff", block_background_fill_dark="#1a2232",
    block_border_color="#e3e8ee", block_border_color_dark="#273246",
    block_shadow="0 1px 3px rgba(15,23,42,.06)", block_shadow_dark="none",
    body_text_color="#1f2937", body_text_color_dark="#e5e7eb",
    block_label_background_fill="#eef6f5", block_label_background_fill_dark="#1f2d3a",
    block_label_text_color="#0f5f58", block_label_text_color_dark="#99e2d8",
    button_primary_background_fill="#2a9d8f", button_primary_background_fill_dark="#2a9d8f",
    button_primary_background_fill_hover="#238a7e", button_primary_background_fill_hover_dark="#33b3a3",
    button_primary_text_color="#ffffff",
)

# All custom colours are CSS variables with a light and a dark value, so every text stays readable.
CSS = """
:root{
  --ok-bg:#ecf8f3; --ok-bd:#9fd9c3; --ok-tx:#14532d; --ok-ac:#2f9e72;
  --no-bg:#fdf5ea; --no-bd:#efcd9a; --no-tx:#6b3f0b; --no-ac:#d08a2c;
  --mo-bg:#eef4fc; --mo-bd:#a9c6ec; --mo-tx:#1e3a66; --mo-ac:#4a7fc9;
  --mute:#64748b; --track:#e5e9ef; --chip:#eef2f6; --chip-tx:#334155;
}
.dark{
  --ok-bg:rgba(47,158,114,.12); --ok-bd:rgba(110,205,165,.35); --ok-tx:#c9f0de; --ok-ac:#6fcfa6;
  --no-bg:rgba(208,138,44,.12); --no-bd:rgba(232,180,110,.35); --no-tx:#f5dfbf; --no-ac:#e8b46e;
  --mo-bg:rgba(74,127,201,.13); --mo-bd:rgba(140,180,235,.38); --mo-tx:#d3e3f8; --mo-ac:#8cb4eb;
  --mute:#94a3b8; --track:#2a3446; --chip:#222c3c; --chip-tx:#cbd5e1;
}
.gradio-container{max-width:1180px !important; margin:auto !important;}
#hdr{text-align:center; padding:6px 0 2px}
#hdr h1{font-size:2rem; font-weight:700; margin:0; letter-spacing:-.01em}
#hdr p{color:var(--mute) !important; margin:6px 0 10px}
.chips{display:flex; gap:8px; justify-content:center; flex-wrap:wrap}
.chip{background:var(--chip); color:var(--chip-tx) !important; border-radius:999px; padding:4px 12px; font-size:.82rem}

.verdict{border-radius:14px; padding:18px 20px; border:1.5px solid; line-height:1.55}
.verdict *{color:inherit !important}
.verdict .label{font-size:.8rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; opacity:.85}
.verdict .title{font-size:1.55rem; font-weight:700; margin:2px 0 6px; display:block}
.verdict .note{font-size:.88rem; opacity:.85; margin-top:6px; display:block}
.v-ok{background:var(--ok-bg); border-color:var(--ok-bd); color:var(--ok-tx) !important}
.v-no{background:var(--no-bg); border-color:var(--no-bd); color:var(--no-tx) !important}
.v-mo{background:var(--mo-bg); border-color:var(--mo-bd); color:var(--mo-tx) !important}
.v-empty{border-style:dashed; border-color:var(--track); color:var(--mute) !important}

.steps{display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:12px}
.step{border-radius:12px; padding:10px 12px; background:var(--chip); border:1px solid var(--track)}
.step *{color:var(--chip-tx) !important}
.step .h{font-weight:600; font-size:.9rem}
.step .d{font-size:.78rem; opacity:.85; margin-top:2px; word-break:break-word}
.step.done{border-color:var(--ok-bd)} .step.done .h::before{content:"✓ "; color:var(--ok-ac) !important}
.step.stop{border-color:var(--no-bd)} .step.stop .h::before{content:"✕ "; color:var(--no-ac) !important}
.step.skip{opacity:.5} .step.skip .h::before{content:"– "}

.kv{width:100%; border-collapse:collapse; font-size:.92rem}
.kv, .kv tr, .kv td{border:none !important; background:transparent !important}
.kv td{padding:7px 4px; border-bottom:1px solid var(--track) !important}
.kv td:first-child{color:var(--mute) !important; width:42%}
.kv code{font-size:.92rem}
.bar{position:relative; height:12px; border-radius:999px; background:var(--track); margin:14px 0 4px}
.bar .fill{height:100%; border-radius:999px}
.bar .thr{position:absolute; top:-5px; width:2px; height:22px; background:var(--chip-tx)}
.barlegend{display:flex; justify-content:space-between; font-size:.78rem; color:var(--mute) !important}
.dev{font-size:.8rem; color:var(--mute) !important; margin-top:8px}
.foot{text-align:center; font-size:.82rem; color:var(--mute) !important; margin-top:6px}
"""

EMPTY = ("<div class='verdict v-empty'><span class='label'>Waiting</span>"
         "<span class='title'>Add a photo of a medicine box</span>"
         "The analysis starts as soon as the photo is added.</div>")
REASONS = {
    "no_detection": "The detector found no drug-name region on this photo.",
    "below_threshold": "The text read is not close enough to any Tunisian trade name.",
}


# ---- drawing --------------------------------------------------------------------------------------------
def draw_box(image: Image.Image, box) -> Image.Image:
    """Photo with the YOLO box and a readable label (font size scales with the image)."""
    out = image.copy()
    if box is None:
        return out
    x1, y1, x2, y2, conf = box
    d = ImageDraw.Draw(out)
    s = min(out.size)
    width, size = max(3, round(s / 160)), max(14, round(s / 28))
    font = ImageFont.load_default(size=size)
    colour = (225, 85, 105)                                          # soft rose: visible on any box colour
    d.rectangle([x1, y1, x2, y2], outline=colour, width=width)
    label = f"drug-name  {conf:.2f}"
    l, t, r, b = d.textbbox((0, 0), label, font=font)
    pad = size // 4
    ty = y1 - (b - t) - 2 * pad if y1 - (b - t) - 2 * pad > 0 else y2  # above the box, or below if no room
    d.rounded_rectangle([x1, ty, x1 + (r - l) + 2 * pad, ty + (b - t) + 2 * pad], radius=pad, fill=colour)
    d.text((x1 + pad, ty + pad - t), label, fill="white", font=font)
    return out


# ---- HTML pieces ----------------------------------------------------------------------------------------
def verdict_html(r: dict) -> str:
    if r["status"] == "ok":
        return (f"<div class='verdict v-ok'><span class='label'>Recognized</span>"
                f"<span class='title'>{escape(r['name'])}</span>"
                f"Match score {r['score']} (threshold {THRESHOLD})."
                f"<span class='note'>Always check the name printed on the box.</span></div>")
    return (f"<div class='verdict v-no'><span class='label'>Refused</span>"
            f"<span class='title'>Box not recognized</span>"
            f"{REASONS.get(r['reason'], '')} Please type the drug name."
            f"<span class='note'>Refusing is safer than guessing a wrong medicine.</span></div>")


def steps_html(r: dict, box) -> str:
    """4-step tracker: where did the image stop?"""
    if box is None:
        s = [("stop", "Detect", "no name region found"), ("skip", "Crop", ""), ("skip", "Read", ""),
             ("skip", "Match", "")]
    else:
        ok = r["status"] == "ok"
        s = [("done", "Detect", f"confidence {box[4]:.2f}"),
             ("done", "Crop", f"rotation kept {r['rotation']}°"),
             ("done" if r["ocr_text"] else "stop", "Read", escape(repr(r["ocr_text"]))),
             ("done" if ok else "stop", "Match", f"score {r['score']} / {THRESHOLD}")]
    cells = "".join(f"<div class='step {c}'><div class='h'>{h}</div><div class='d'>{d}</div></div>" for c, h, d in s)
    return f"<div class='steps'>{cells}</div>"


def reading_html(r: dict, box) -> str:
    """Key facts + the match score drawn against the threshold."""
    if box is None:
        return "<p style='color:var(--mute)'>Nothing was cropped, so nothing was read.</p>"
    score = float(r["score"])
    colour = "var(--ok-ac)" if r["status"] == "ok" else "var(--no-ac)"
    rows = [("Text read by TrOCR", f"<code>{escape(r['ocr_text']) or '—'}</code>"),
            ("Closest Tunisian name", escape(str(r["candidate"]))),
            ("Rotation kept", f"{r['rotation']}°"),
            ("Detector confidence", f"{box[4]:.2f}")]
    table = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f"<table class='kv'>{table}</table>"
            f"<div class='bar'><div class='fill' style='width:{score}%; background:{colour}'></div>"
            f"<div class='thr' style='left:{THRESHOLD}%'></div></div>"
            f"<div class='barlegend'><span>match score {score}</span><span>threshold {THRESHOLD} ▲</span></div>")


# ---- v2 -------------------------------------------------------------------------------------------------
V2_REASON = "No text line on the box is close enough to a Tunisian trade name or molecule."
DEV_NOTE = (f"<p class='dev'>v2 thresholds ({T_BRAND} trade name, {T_DCI} molecule) were tuned on 13 development "
            "photos; an independent test on new photos is pending.</p>")


def draw_lines(image: Image.Image, cands: list[dict], win: dict | None) -> Image.Image:
    """Every text line that gave a reading (grey), the winning line highlighted (teal) with its label."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    s = min(out.size)
    width, size = max(2, round(s / 220)), max(14, round(s / 30))
    font = ImageFont.load_default(size=size)
    seen = set()
    for c in cands:
        b = tuple(c["box"])
        if b not in seen:
            seen.add(b)
            d.rectangle(b, outline=(148, 163, 184), width=width)    # soft slate: candidate lines
    if win:
        x1, y1, x2, y2 = win["box"]
        colour = (42, 157, 143)                                     # same teal as the accent
        d.rectangle([x1, y1, x2, y2], outline=colour, width=width + 2)
        label = f" tallest text · rot {win['rotation']}° "
        l, t, r, b = d.textbbox((0, 0), label, font=font)
        pad = size // 4
        ty = y1 - (b - t) - 2 * pad if y1 - (b - t) - 2 * pad > 0 else y2
        d.rounded_rectangle([x1, ty, x1 + (r - l) + 2 * pad, ty + (b - t) + 2 * pad], radius=pad, fill=colour)
        d.text((x1 + pad, ty + pad - t), label, fill="white", font=font)
    return out


def verdict_v2(dec: dict) -> str:
    if dec["status"] == "brand":
        return (f"<div class='verdict v-ok'><span class='label'>Trade name recognized</span>"
                f"<span class='title'>{escape(dec['answer'])}</span>Match score {dec['score']} "
                f"(threshold {T_BRAND}).<span class='note'>Always check the name printed on the box.</span>"
                f"{DEV_NOTE}</div>")
    if dec["status"] == "molecule":
        return (f"<div class='verdict v-mo'><span class='label'>Molecule recognized · brand not identified</span>"
                f"<span class='title'>{escape(dec['answer'])}</span>Match score {dec['score']} "
                f"(threshold {T_DCI}). The generic name was read, but no trade name was close enough."
                f"<span class='note'>A safe partial answer instead of guessing a brand.</span>{DEV_NOTE}</div>")
    return (f"<div class='verdict v-no'><span class='label'>Refused</span><span class='title'>Box not recognized</span>"
            f"{V2_REASON} Please type the drug name."
            f"<span class='note'>Refusing is safer than guessing a wrong medicine.</span>{DEV_NOTE}</div>")


def steps_v2(dec: dict, cands: list[dict]) -> str:
    n_lines = len({tuple(c["box"]) for c in cands})
    if not cands:
        s = [("stop", "Find text", "no readable line"), ("skip", "Pick tallest", ""), ("skip", "Read", ""),
             ("skip", "Match", "")]
    else:
        w = dec["from"] or cands[0]
        s = [("done", "Find text", f"{n_lines} line(s) read"),
             ("done", "Pick tallest", f"height {w['height']} px"),
             ("done", "Read", escape(repr(w["text"]))),
             ("done" if dec["status"] != "refused" else "stop", "Match", dec["status"])]
    cells = "".join(f"<div class='step {c}'><div class='h'>{h}</div><div class='d'>{d}</div></div>" for c, h, d in s)
    return f"<div class='steps'>{cells}</div>"


def bar(score: float, thr: float, colour: str, label: str) -> str:
    return (f"<div class='bar'><div class='fill' style='width:{score}%; background:{colour}'></div>"
            f"<div class='thr' style='left:{thr}%'></div></div>"
            f"<div class='barlegend'><span>{label} {score}</span><span>threshold {thr} ▲</span></div>")


def reading_v2(dec: dict, cands: list[dict]) -> str:
    if not cands:
        return "<p style='color:var(--mute)'>No text line gave a reading of 4 characters or more.</p>"
    b, m = ReaderV2.best(cands, "brand"), ReaderV2.best(cands, "dci")
    rows = [("Text read", f"<code>{escape((dec['from'] or b)['text'])}</code>"),
            ("Closest trade name", f"{escape(str(b['brand']))}"),
            ("Closest molecule", f"{escape(str(m['dci']))}"),
            ("Lines read", str(len({tuple(c['box']) for c in cands})))]
    table = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f"<table class='kv'>{table}</table>"
            + bar(b["brand_score"], T_BRAND, "var(--ok-ac)", "trade-name score")
            + bar(m["dci_score"], T_DCI, "var(--mo-ac)", "molecule score"))


def analyze_v2(image: Image.Image):
    r = READER_V2.identify(image)                                     # same function as the v2 evaluation
    cands, win = r["candidates"], r["from"]
    crop = None
    if win:
        crop = image.crop(tuple(win["box"]))
        if win["rotation"]:
            crop = crop.rotate(win["rotation"], expand=True)          # shown as it was read
    raw = {k: r[k] for k in ("status", "answer", "score")} | {"candidates": cands}
    return verdict_v2(r), steps_v2(r, cands), draw_lines(image, cands, win), crop, reading_v2(r, cands), raw


# ---- callback -------------------------------------------------------------------------------------------
def analyze(image, version):
    if image is None:                                                 # back to the empty state: hide the panels
        return EMPTY, "", None, None, "", {}, gr.Column(visible=False)
    image = image.convert("RGB")
    if version.startswith("v2"):
        return (*analyze_v2(image), gr.Column(visible=True))
    box = READER.detect(image)                                        # v1 step 1: where is the name?
    r = READER.identify(image)                                        # steps 2-4: crop, read, match, decide
    crop = None
    if box is not None:
        crop = READER.crop(image, box)                                # the exact crop given to TrOCR
        if r["rotation"]:
            crop = crop.rotate(r["rotation"], expand=True)            # shown as it was read
    return (verdict_html(r), steps_html(r, box), draw_box(image, box), crop, reading_html(r, box), r,
            gr.Column(visible=True))                                  # show the result panels


# ---- examples: 3 Kaggle test images shipped in examples/ (CC BY 4.0, see examples/ATTRIBUTION.md) ----
# Fallback: the same seed-42 draw from the local Kaggle copy, if examples/ is empty
_shipped = sorted(Path("examples").glob("*.jpg"))
if _shipped:
    EXAMPLES = [[str(p)] for p in _shipped]
else:
    _kaggle = sorted(Path("data/raw/kabul/test/images").glob("*.jpg"))
    EXAMPLES = [[str(p)] for p in random.Random(42).sample(_kaggle, 3)] if len(_kaggle) >= 3 else None

with gr.Blocks(title="MedBox Reader") as demo:
    gr.HTML("<div id='hdr'><h1>💊 MedBox Reader</h1>"
            "<p>Photo of a Tunisian medicine box → trade name, or a safe refusal</p>"
            f"<div class='chips'><span class='chip'>Text detection + tallest line</span><span class='chip'>TrOCR reader</span>"
            f"<span class='chip'>2,160 trade names · 728 molecules</span><span class='chip'>refuses when unsure</span>"
            "</div></div>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            inp = gr.Image(type="pil", label="Photo of a medicine box", height=380, sources=["upload", "clipboard", "webcam"])
            version = gr.Radio(["v2 · tallest text (default)", "v1 · YOLO detector"], value="v2 · tallest text (default)",
                               label="Reader version", info="Compare both versions on the same photo")
            btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Markdown("<span style='color:var(--mute)'>Best results: brand name clearly visible, "
                        "photo not too small, box filling most of the frame.</span>")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Try an image from the Kaggle test split (CC BY 4.0)")
        with gr.Column(scale=6, min_width=420):
            verdict = gr.HTML(EMPTY)
            with gr.Column(visible=False) as details:                   # hidden until there is a result
                steps = gr.HTML("")
                with gr.Tabs():
                    with gr.Tab("What the model saw"):
                        with gr.Row(equal_height=False):
                            annotated = gr.Image(label="1 · Where the model looked", height=300,
                                                 interactive=False, scale=3)
                            with gr.Column(scale=2):
                                crop_img = gr.Image(label="2 · Crop given to TrOCR", height=130, interactive=False)
                                reading = gr.HTML("")
                    with gr.Tab("Raw output"):
                        raw = gr.JSON(label="Raw result")

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            f"**v2 (default)** — 1. **Find text**: a text detector (CRAFT, via EasyOCR) finds every text line. "
            f"2. **Pick tallest**: the 5 tallest lines are kept — on a medicine box, the brand is usually the "
            f"biggest text. 3. **Read**: TrOCR reads each line at 0°, 90°, 180° and 270°; readings shorter than "
            f"4 characters are dropped. 4. **Match**: the text is compared with 2,160 trade names and 728 molecules. "
            f"Trade name if score ≥ {T_BRAND}; otherwise molecule if score ≥ {T_DCI}; otherwise **refused**. "
            f"Development results on 13 own photos (thresholds tuned on them): 6/8 trade names, 1/8 molecule, "
            f"0 wrong answers. Independent test pending.\n\n"
            f"**v1** — 1. **Detect** — YOLOv8n (fine-tuned on 1,276 images; mAP@0.5 = 0.667 on a held-out test split) "
            f"finds the drug-name region and keeps the most confident box.\n"
            f"2. **Crop** — the box is enlarged by 5 % and cut out, so TrOCR reads a single line.\n"
            f"3. **Read** — TrOCR reads the crop at 0°, 90° and 270°; the best-matching reading is kept.\n"
            f"4. **Match** — rapidfuzz compares the text with 2,160 Tunisian trade names; "
            f"score ≥ {THRESHOLD} shows the name, otherwise the system **refuses**.\n\n"
            f"Pilot results (14 web images): 0 wrong names, 5/5 non-drug objects refused, 0/9 boxes recognized. "
            f"Known limit: the detector often frames the molecule (generic name) instead of the brand.")
    gr.HTML("<p class='foot'>Prototype for research and learning — not a medical tool, does not replace a "
            "pharmacist. No image is stored.</p>")

    outputs = [verdict, steps, annotated, crop_img, reading, raw, details]
    btn.click(analyze, inputs=[inp, version], outputs=outputs)
    inp.change(analyze, inputs=[inp, version], outputs=outputs)       # runs on upload, paste, webcam or example
    version.change(analyze, inputs=[inp, version], outputs=outputs)   # switching version re-analyses the photo

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)                                 # Gradio 6: theme and css go to launch()
