from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


ROOT = Path(__file__).parent
ARTIFACT_PATH = ROOT / "models" / "ploidy_random_forest.joblib"
BACKGROUND_PATH = ROOT / "data" / "lime_background.csv"
IMPORTANCE_PATH = ROOT / "data" / "global_feature_importance.csv"

CLASS_NAMES = ["Aneuploidy", "Euploidy"]


artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
imputer = artifact["imputer"]
FEATURES = artifact["features"]
THRESHOLD = artifact["threshold"]
MEDIANS = artifact["medians"]

background = pd.read_csv(BACKGROUND_PATH)
importance = pd.read_csv(IMPORTANCE_PATH)

explainer = LimeTabularExplainer(
    background.values,
    feature_names=FEATURES,
    class_names=CLASS_NAMES,
    mode="classification",
)


def predict_proba_from_raw(raw_values):
    frame = pd.DataFrame([raw_values], columns=FEATURES)
    imputed = pd.DataFrame(imputer.transform(frame), columns=FEATURES)
    return model.predict_proba(imputed), imputed


def predict_proba_from_imputed_array(values):
    frame = pd.DataFrame(values, columns=FEATURES)
    return model.predict_proba(frame)


def make_lime_plot(explanation):
    rows = explanation.as_list(label=1)
    labels = [row[0] for row in rows][::-1]
    values = [row[1] for row in rows][::-1]
    colors = ["#008b8b" if value > 0 else "#b23a48" for value in values]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Contribution toward Euploidy")
    ax.set_title("Local LIME Explanation")
    fig.tight_layout()
    return fig


def infer(*values):
    raw_values = dict(zip(FEATURES, values))
    proba, imputed = predict_proba_from_raw(raw_values)
    euploidy_probability = float(proba[0, 1])
    aneuploidy_probability = float(proba[0, 0])
    prediction = "Euploidy" if euploidy_probability >= THRESHOLD else "Aneuploidy"

    explanation = explainer.explain_instance(
        imputed.iloc[0].values,
        predict_proba_from_imputed_array,
        num_features=min(10, len(FEATURES)),
        labels=(1,),
    )

    probabilities = {
        "Euploidy": euploidy_probability,
        "Aneuploidy": aneuploidy_probability,
    }
    summary = (
        f"Prediction: {prediction}\n\n"
        f"Euploidy probability: {euploidy_probability:.3f}\n\n"
        f"Decision threshold: {THRESHOLD:.6f}"
    )
    local_table = pd.DataFrame(
        explanation.as_list(label=1),
        columns=["Feature condition", "Contribution toward Euploidy"],
    )
    global_table = importance.head(12).reset_index(drop=True)

    return probabilities, summary, make_lime_plot(explanation), local_table, global_table


def default_value(feature):
    value = MEDIANS.get(feature, 0)
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


inputs = [
    gr.Number(label=feature, value=default_value(feature), precision=3)
    for feature in FEATURES
]

with gr.Blocks(title="XAI Ploidy Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# XAI Ploidy Prediction")
    gr.Markdown(
        "Random Forest model for embryo ploidy prediction with LIME explanation."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input Features")
            for component in inputs:
                component.render()
            submit = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("## Prediction")
            label = gr.Label(label="Class probabilities")
            summary = gr.Textbox(label="Result", lines=5)
            lime_plot = gr.Plot(label="Local explanation")

    with gr.Row():
        local_table = gr.Dataframe(label="LIME contribution table")
        global_table = gr.Dataframe(label="Global feature importance")

    submit.click(
        infer,
        inputs=inputs,
        outputs=[label, summary, lime_plot, local_table, global_table],
    )

    demo.load(
        infer,
        inputs=inputs,
        outputs=[label, summary, lime_plot, local_table, global_table],
    )


if __name__ == "__main__":
    demo.launch()
