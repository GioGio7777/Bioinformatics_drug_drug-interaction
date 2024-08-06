from pathlib import Path

import pandas as pd

import typer
from loguru import logger

from ubmk_drug_drug.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def mat_drug_drug_wo_names(input_path, output_path):
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))
    dr_dr = pd.read_csv(input_path + "/mat_drug_drug.txt", header=None,delim_whitespace=True)
    dr_dr.to_csv(str(output_path + "/drug_drug_without_name.csv"), index=False)

    return logger.success("Dataset Processed and Saved in" + output_path)


@app.command()
def main(

        input_path: Path = RAW_DATA_DIR / "",
        output_path: Path = PROCESSED_DATA_DIR,

):
    logger.info("Processing dataset drug-drug interaction ")
    mat_drug_drug_wo_names(input_path, output_path)


if __name__ == "__main__":
    app()
