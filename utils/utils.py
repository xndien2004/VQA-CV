import json
import pandas as pd
import ast

def countTrainableParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def countAllParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def is_json(data_path):
    path = str(data_path).lower()
    return "vitextvqa" in path or "viocrvqa" in path

def load_json(data_path):
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    df_img = pd.DataFrame(raw["images"])
    df_ann = pd.DataFrame(raw["annotations"])

    df = df_ann.merge(df_img, left_on="image_id", right_on="id")

    df["answer"] = df["answers"].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]
    )

    return df

def load_csv(data_path):
    return pd.read_csv(data_path)

def load_data(data_path):
    if is_json(data_path):
        return load_json(data_path)
    return load_csv(data_path)