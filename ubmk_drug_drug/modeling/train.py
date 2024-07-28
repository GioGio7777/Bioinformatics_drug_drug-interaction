from pathlib import Path

import pandas as pd
import torch

import typer
from loguru import logger
from tqdm import tqdm

from models.model import GCN
from torch_geometric.nn import GAE

from ubmk_drug_drug.features import toAdjencyMatrix

from ubmk_drug_drug.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_data(data_path):
    df = pd.read_csv(data_path)
    return toAdjencyMatrix(df)


def train(model, optimizer, data, loss):
    output, x = model.encode(data.x, data.edge_index)
    optimizer.zero_grad()
    loss = model.recon_loss(output, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        input_path: Path = PROCESSED_DATA_DIR / "drug_drug_without_name.csv",
        model_path: Path = MODELS_DIR / "model.pt"
        # -----------------------------------------
):
    logger.info(f"torch version: {torch.__version__}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Device is {device}")

    out_channels = 2
    num_features = 1
    m = GCN(num_features, out_channels)
    model = GAE(m)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    data = load_data(input_path)

    for epoch in tqdm(range(500), desc="Training Progress"):
        loss = train(model, optimizer, data, loss)
        if epoch % 10 == 0:
            logger.success(f'Epoch {epoch}, Loss: {loss}', style="braces")

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        logger.info(f"Node Embeddings: {z} ")

    torch.save(model, model_path)


if __name__ == "__main__":
    app()
