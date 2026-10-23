"""Gradio web interface for the Kepler transit classifier.

Loads models/random_forest.joblib and models/preprocessor.joblib (produced by
src/train.py) and predicts CONFIRMED vs FALSE POSITIVE for one candidate.

Usage (from the project root):
    python app.py
"""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from src.data_loader import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

# Label 1 = CONFIRMED, 0 = FALSE POSITIVE (see data_loader.py).
LABELS = {1: "CONFIRMED", 0: "FALSE POSITIVE"}

# Input ranges and defaults per feature: (label, min, max, step, default, unit).
FEATURE_SPECS = {
    "koi_period": ("Orbital period", 0.0, 1100.0, 0.1, 10.0, "days"),
    "koi_duration": ("Transit duration", 0.0, 140.0, 0.1, 3.0, "hours"),
    "koi_depth": ("Transit depth", 0.0, 1500000.0, 1.0, 500.0, "ppm"),
    "koi_prad": ("Planetary radius", 0.0, 200000.0, 0.1, 2.3, "Earth radii"),
    "koi_teq": ("Equilibrium temperature", 0.0, 15000.0, 1.0, 800.0, "K"),
}

# Fail fast with a clear message if the pipeline has not been trained yet.
for _path in (MODEL_PATH, PREPROCESSOR_PATH):
    if not _path.exists():
        raise FileNotFoundError(
            f"{_path} not found. Run `python src/train.py` first."
        )

MODEL = joblib.load(MODEL_PATH)
PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)


def predict(*values: float) -> tuple[str, dict[str, float]]:
    """Return the predicted class and per-class probabilities for one candidate."""
    row = pd.DataFrame([dict(zip(FEATURE_COLUMNS, values))], columns=FEATURE_COLUMNS)
    X = PREPROCESSOR.transform(row)
    proba = MODEL.predict_proba(X)[0]

    # Map model.classes_ to labels so the output does not depend on column order.
    probs = {LABELS[int(c)]: float(p) for c, p in zip(MODEL.classes_, proba)}
    prediction = max(probs, key=probs.get)
    return prediction, probs


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Kepler Transit Classifier") as demo:
        gr.Markdown(
            "# Kepler Transit Classifier\n"
            "Enter the five transit features for one candidate. The model predicts "
            "whether it is a **CONFIRMED** planet or a **FALSE POSITIVE**. "
            "The model was trained on synthetic data, so treat predictions as a demo."
        )

        inputs = []
        with gr.Row():
            for col in FEATURE_COLUMNS:
                label, lo, hi, step, default, unit = FEATURE_SPECS[col]
                with gr.Column():
                    slider = gr.Slider(
                        minimum=lo,
                        maximum=hi,
                        step=step,
                        value=default,
                        label=f"{label} ({unit})",
                    )
                    number = gr.Number(value=default, label=f"{label} ({unit}) – exact value")
                    # Keep slider and number box in sync; .input fires only on user edits,
                    # which avoids feedback loops.
                    slider.input(lambda v: v, inputs=slider, outputs=number)
                    number.input(
                        lambda v, lo=lo, hi=hi: min(max(v, lo), hi) if v is not None else lo,
                        inputs=number,
                        outputs=slider,
                    )
                    inputs.append(number)

        with gr.Row():
            predict_button = gr.Button("Classify", variant="primary")
        with gr.Row():
            prediction_box = gr.Textbox(label="Prediction", interactive=False)
            confidence = gr.Label(label="Confidence", num_top_classes=2)

        predict_button.click(fn=predict, inputs=inputs, outputs=[prediction_box, confidence])

    return demo


if __name__ == "__main__":
    build_interface().launch()
