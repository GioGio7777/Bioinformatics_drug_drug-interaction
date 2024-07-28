from pathlib import Path

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics.classification import BinaryAUROC
from ubmk_drug_drug.modeling.train import load_data

import typer
from loguru import logger
from tqdm import tqdm

from ubmk_drug_drug.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


def tsne_vis(embeddings_2d,out):
    out = str(Path(out))
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', marker='o', edgecolor='k')
    plt.title("t-SNE ile Embedding Görselleştirme")
    plt.xlabel("Boyut 1")
    plt.ylabel("Boyut 2")
    plt.grid(True)
    plt.savefig(out+'/tsne.png')

@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        input_path: Path = PROCESSED_DATA_DIR / "drug_drug_without_name.csv",
        model_path: Path = MODELS_DIR / "model.pt",
        output_path: Path = FIGURES_DIR,
        # -----------------------------------------
):
    auroc = BinaryAUROC()

    model = torch.load(model_path)
    data = load_data(input_path)
    with torch.no_grad():
        out, x = model.encode(data.x, data.edge_index)

    out = out.cpu()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(out)

    tsne_vis(embeddings_2d,output_path)

    logger.success("Plot Created")


if __name__ == "__main__":
    app()
