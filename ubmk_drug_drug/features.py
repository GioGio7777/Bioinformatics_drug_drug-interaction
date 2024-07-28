import pandas
import torch
import typer
from torch_geometric.data import Data
from tqdm import tqdm

import numpy as np

app = typer.Typer()


def toAdjencyMatrix(df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_of_row = df.shape[0]
    num_of_column = df.shape[1] - 1

    adj_matrix = np.zeros((num_of_row, num_of_column))

    for i in range(num_of_row):
        for x in range(num_of_column):
            if df.iloc[i, x] == 1:
                adj_matrix[i, x] = 1


    row, col = np.where(adj_matrix == 1)
    edge_index = np.vstack((row, col))
    edge_index = torch.tensor(edge_index, dtype=torch.int32)

    data_x = np.ones((df.shape[0], 1))
    data_x = torch.tensor(data_x, dtype=torch.float32)

    data = Data(x=data_x, edge_index=edge_index).to(device)

    return data
