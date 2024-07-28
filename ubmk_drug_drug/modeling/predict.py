from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import typer
import xgboost as xgb

from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from train import load_data

from ubmk_drug_drug.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def xgboost(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        input_path: Path = PROCESSED_DATA_DIR / "drug_drug_without_name.csv",
        model_path: Path = MODELS_DIR / "model.pt",
        # -----------------------------------------
):
    data = load_data(input_path)
    model = torch.load(model_path)




    with torch.no_grad():
        out, x = model.encode(data.x, data.edge_index)

    out = out.cpu()

    existing_links = [(data.edge_index[0, i].item(), data.edge_index[1, i].item()) for i in
                      range(data.edge_index.shape[1])]
    all_possible_links = list(combinations(range(len(out)), 2))

    missing_links = [link for link in all_possible_links if link not in existing_links]

    # ----------------positives-----------------
    positive_examples = [(out[i], out[j]) for i, j in existing_links]
    positive_labels = [1] * len(positive_examples)

    # ----------------negatives-----------------
    np.random.seed(42)
    negative_sample_indices = np.random.choice(len(missing_links), len(existing_links), replace=False)
    negative_examples = [(out[i], out[j]) for i, j in np.array(missing_links)[negative_sample_indices]]
    negative_labels = [0] * len(negative_examples)

    examples = positive_examples + negative_examples
    labels = positive_labels + negative_labels

    X = [np.concatenate([u, v]) for u, v in examples]
    y = labels

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgboost(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")

    predicted_links = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            node_pair = negative_examples[i]
            predicted_links.append((node_pair[0], node_pair[1]))


if __name__ == "__main__":
    app()
