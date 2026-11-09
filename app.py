"""MedBox Reader — local demo (Gradio). Same `identify` function as the evaluation: one version of the code.

    python app.py      # then open the local URL shown in the terminal

The demo stores no image: the uploaded picture is only kept in memory for the request.
"""

import gradio as gr

from medbox.reader import identify


def demo(image):
    if image is None:
        return {"status": "refused", "reason": "no image"}
    result = identify(image)                  # dict: status, name, score, ocr_text, box, rotation...
    if result["status"] == "refused":
        result["message"] = "Box not recognized. Please type the drug name."
    return result


gr.Interface(
    fn=demo,
    inputs=gr.Image(type="pil", label="Photo of a medicine box"),
    outputs=gr.JSON(label="Result"),
    title="MedBox Reader (prototype, not medical advice)",
    description=(
        "Reads the trade name of a Tunisian medicine box, or refuses. "
        "Pilot results: it refused all test images and gave 0 wrong names. "
        "Not a medical tool; it does not replace a pharmacist."
    ),
).launch()
