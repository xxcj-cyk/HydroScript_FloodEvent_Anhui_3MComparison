import os
import xarray as xr
import pandas as pd

base_project_name = "anhui_21basin_797flood"
project_name = f"{base_project_name}_train"
model_name = "Anhui_LSTM16"
root_dir = f"./results/{model_name}/{project_name}"
output_csv = f"./visualizations/evaluation_metrics/{model_name}/{base_project_name}/{model_name}_{project_name}_metrics.csv"
valid_csv = f"./visualizations/evaluation_metrics/{model_name}/{base_project_name}/{model_name}_{base_project_name}_valid_metrics.csv"
combined_csv = f"./visualizations/evaluation_metrics/{model_name}/{base_project_name}/{model_name}_{base_project_name}_metrics.csv"

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
                "FHV": "FHV_Train",
                "FLV": "FLV_Train",
                "trainloss": "TrainLoss",
            },
            inplace=True,
        )

        # Round specified columns to 3 decimal places
        for col in [
            "NSE_Train",
            "KGE_Train",
            "RMSE_Train",
            "Corr_Train",
            "FHV_Train",
            "FLV_Train",
        ]:
            train_combined_df[col] = train_combined_df[col].round(3)

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
        "RMSE_Train",
        "RMSE_Validation",
        "Corr_Train",
        "Corr_Validation",
        "KGE_Train",
        "KGE_Validation",
        "FHV_Train",
        "FHV_Validation",
        "FLV_Train",
        "FLV_Validation",
    ]
    merged_df = merged_df[desired_columns_order]

    # Save merged data to CSV with 3 decimal places
    merged_df.to_csv(combined_csv, index=False, float_format="%.3f")
    print(f"Training and validation data merged and saved to {combined_csv}")
