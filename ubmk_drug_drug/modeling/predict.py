from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import typer
import xgboost as xgb

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, log_loss, precision_score, \
    precision_recall_curve, auc
from sklearn.model_selection import train_test_split

from train import load_data

from ubmk_drug_drug.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def xgboost(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=5)
    model.fit(X_train, y_train)
    return model


def support_vector_machines(X_train, y_train):
    model = svm.SVC(kernel="rbf", probability=True)
    return model.fit(X_train, y_train)


def logistic_regression(X_train, y_train):
    model = LogisticRegression()
    return model.fit(X_train, y_train)


@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        input_path: Path = PROCESSED_DATA_DIR / "drug_drug_without_name.csv",
        model_path: Path = MODELS_DIR / "model_mac.pt",
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
    # Accuracy: 0.872478206724782
    # ROC AUC: 0.9418855600354954
    # F1: 0.8748778103616813
    # Log Loss: 0.30125885801621655
    # Precision: 0.8651522474625423
    # Recall: 0.8848245180425112
    # AUPR: 0.9040045720426886
    # 2069 number of links predicted

    # model = random_forest(X_train, y_train)
    # Accuracy: 0.8445828144458282
    # ROC AUC: 0.9102828764390235
    # F1: 0.8462296697880729
    # Log Loss: 1.630960832981541
    # Precision: 0.8437346437346437
    # Recall: 0.8487394957983193
    # AUPR: 0.8843441681475526
    # 2035 number of links predicted

    # model = support_vector_machines(X_train, y_train)
    # Accuracy: 0.7785803237858032
    # ROC AUC: 0.8572371542522041
    # F1: 0.7709353259469209
    # Log Loss: 0.48196850413185144
    # Precision: 0.8051668460710442
    # Recall: 0.7394957983193278
    # AUPR: 0.8379602138514749
    # 1858 number of links predicted

    # model = logistic_regression(X_train, y_train)
    # Accuracy: 0.7519302615193026
    # ROC AUC: 0.8390187045760897
    # F1: 0.7549212598425197
    # Log Loss: 0.49728007284287823
    # Precision: 0.7515923566878981
    # Recall: 0.7582797825012358
    # AUPR: 0.8158327072035334
    # 2041 number of links predicted

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    log_loss_val = log_loss(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    precision2, recall2, thresholds = precision_recall_curve(y_test, y_pred)
    auc_precision_recall = auc(recall2, precision2)

    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"F1: {f1}")
    print(f"Log Loss: {log_loss_val}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUPR: {auc_precision_recall}")

    predicted_links = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            node_pair = negative_examples[i]
            predicted_links.append((node_pair[0], node_pair[1]))

    print(f"{len(predicted_links)} number of links predicted")


if __name__ == "__main__":
    app()
