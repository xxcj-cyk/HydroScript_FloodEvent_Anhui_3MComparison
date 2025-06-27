import os
import xarray as xr
import pandas as pd

base_project_name = "anhui21_797_PET_Anhui"
project_name = f"{base_project_name}_train"
model_name = "Anhui_EnLoss-LSTM"
root_dir = f"./Result/{model_name}/{project_name}"
output_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{project_name}_metrics.csv"
valid_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}_valid_metrics.csv"
combined_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}_metrics.csv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Load the single training metric file
train_metric_file = os.path.join(root_dir, "metric_streamflow.csv")
if os.path.isfile(train_metric_file):
    train_combined_df = pd.read_csv(train_metric_file)

    # Load basin order from .nc file
    nc_file = os.path.join(root_dir, "epoch_best_model.pth_flow_pred.nc")
    if os.path.isfile(nc_file):
        nc_data = xr.open_dataset(nc_file)

        # Assuming 'basin' is a coordinate or variable that gives the basin order
        basin_order = nc_data["basin"].values
        nc_data.close()

        # Replace 'basin_id' in the CSV with the ordered basins from the .nc file
        train_combined_df["basin_id"] = basin_order

        # Rename columns for consistency
        train_combined_df.rename(
            columns={
                "basin_id": "Basin_ID",
                "NSE": "NSE_Train",
                "KGE": "KGE_Train",
                "RMSE": "RMSE_Train",
                "Corr": "Corr_Train",
                "PFE": "PFE_Train",
                "PTE": "PTE_Train",
                "trainloss": "Train_Loss",
            },
            inplace=True,
        )

        # Round specified columns to match step1 format
        # NSE and KGE: 3 decimal places
        for col in ["NSE_Train", "KGE_Train"]:
            train_combined_df[col] = train_combined_df[col].round(3)
        # RMSE and Corr: 2 decimal places
        for col in ["RMSE_Train", "Corr_Train"]:
            train_combined_df[col] = train_combined_df[col].round(2)
        # PFE: 1 decimal place
        train_combined_df["PFE_Train"] = train_combined_df["PFE_Train"].round(1)
        # PTE: integer (safely handle NA values)
        train_combined_df["PTE_Train"] = train_combined_df["PTE_Train"].fillna(0).round(0)
        train_combined_df["PTE_Train"] = train_combined_df["PTE_Train"].astype(int, errors='ignore')

        # Sort by 'Basin_ID'
        train_combined_df.sort_values(by="Basin_ID", inplace=True)

        # Save training data to CSV
        train_combined_df.to_csv(output_csv, index=False)
        print(f"Training data saved to {output_csv}")
    else:
        print("NC file for basin order not found.")
else:
    print("Training metric file not found.")

# Check if validation file exists and merge if present
if not os.path.isfile(valid_csv):
    print("Validation data file not found, merging process ended.")
else:
    # Read validation data
    valid_df = pd.read_csv(valid_csv)

    # Merge training and validation data on Basin_ID
    merged_df = pd.merge(train_combined_df, valid_df, on="Basin_ID", how="inner")

    # Set the desired column order
    desired_columns_order = [
        "Basin_ID",
        "Train_Loss",
        "Validation_Loss",
        "Epoch",
        "NSE_Train",
        "NSE_Validation",
        "KGE_Train",
        "KGE_Validation",
        "RMSE_Train",
        "RMSE_Validation",
        "Corr_Train",
        "Corr_Validation",
        "PFE_Train",
        "PFE_Validation",
        "PTE_Train",
        "PTE_Validation",
    ]
    merged_df = merged_df[desired_columns_order]

    # Save merged data to CSV with correct number format
    for col in merged_df.columns:
        if "NSE" in col or "KGE" in col:
            merged_df[col] = merged_df[col].round(3)
        elif "RMSE" in col or "Corr" in col:
            merged_df[col] = merged_df[col].round(2)
        elif "PFE" in col:
            merged_df[col] = merged_df[col].round(1)
        elif "PTE" in col:
            # 安全处理可能存在的NA值
            merged_df[col] = merged_df[col].fillna(0).round(0)
            merged_df[col] = merged_df[col].astype(int, errors='ignore')
    
    # Save to CSV
    merged_df.to_csv(combined_csv, index=False)
    print(f"Training and validation data merged and saved to {combined_csv}")
