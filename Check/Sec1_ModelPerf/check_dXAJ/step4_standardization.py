import pandas as pd
import numpy as np

base_project_name = "anhui21_797_PET_Anhui"
model_name = "Anhui_dPL"
combined_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}_metrics.csv"
final_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}.csv"

def clean_and_merge_metrics(df):
    df = df.copy()
    metrics = ["NSE", "KGE", "RMSE", "Corr"]
    mask_use_train = pd.Series(False, index=df.index)
    train_col, valid_col = "NSE_Train", "NSE_Validation"
    for col in [train_col, valid_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace("#NAME?", np.nan), errors="coerce"
            ).apply(lambda x: x if abs(x) < 1e6 else np.nan)
    if train_col in df.columns and valid_col in df.columns:
        mask_use_train = df[valid_col].isna()
        df["NSE"] = df[valid_col].combine_first(df[train_col])
    for metric in metrics[1:]:
        train_col, valid_col = f"{metric}_Train", f"{metric}_Validation"
        for col in [train_col, valid_col]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].replace("#NAME?", np.nan), errors="coerce"
                ).apply(lambda x: x if abs(x) < 1e6 else np.nan)
        if train_col in df.columns and valid_col in df.columns:
            df[metric] = df[valid_col].combine_first(df[train_col])
    for metric in ["PFE", "PTE"]:
        train_col, valid_col = f"{metric}_Train", f"{metric}_Validation"

        if train_col in df.columns and valid_col in df.columns:
            df[train_col] = pd.to_numeric(df[train_col], errors="coerce")
            df[valid_col] = pd.to_numeric(df[valid_col], errors="coerce")

            df[metric] = np.where(mask_use_train, df[train_col], df[valid_col])

    cols_to_keep = (
        ["Basin_ID", "Train_Loss", "Validation_Loss", "Epoch"]
        + metrics
        + ["PFE", "PTE"]
    )
    return df[[col for col in cols_to_keep if col in df.columns]]


if __name__ == "__main__":
    df = pd.read_csv(combined_csv)
    cleaned_df = clean_and_merge_metrics(df)
    cleaned_df.to_csv(final_csv, index=False)