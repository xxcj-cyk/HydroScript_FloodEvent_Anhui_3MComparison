import os
import json
import glob
import xarray as xr
import matplotlib.pyplot as plt
import csv

# Define project and directories
base_project_name = "anhui21_797_PET_Anhui"
model_name = "Anhui_EnLoss-dPL"
root_dir = f"./Result/{model_name}/{base_project_name}"
figure_dir = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}"
csv_file = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}_valid_metrics.csv"

# Read basin IDs order from epochbest_model.pthflow_pred.nc file
nc_file_path = os.path.join(root_dir, "epoch_best_flow_pred.nc")
nc_data = xr.open_dataset(nc_file_path)

# Assuming the .nc file has a variable 'basin_id' with the correct order of basin IDs
basin_ids = list(nc_data["basin"].values)

# Ensure the directory for saving figures exists
os.makedirs(figure_dir, exist_ok=True)

# Initialize the CSV file and write the header
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Basin_ID",
            "Epoch",
            "Train_Loss",
            "Validation_Loss",
            "NSE_Validation",
            "RMSE_Validation",
            "Corr_Validation",
            "KGE_Validation",
            "FHV_Validation",
            "FLV_Validation",
        ]
    )

# Use glob to find matching JSON files
json_files = glob.glob(os.path.join(root_dir, "*_2025*.json"))

# Process each JSON file
for json_file in json_files:
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Initialize tracking for each basin's minimum validation loss and corresponding metrics
    min_loss_metrics = {
        basin: {
            "epoch": 0,
            "train_loss": float("inf"),
            "validation_loss": float("inf"),
            "NSE": None,
            "RMSE": None,
            "Corr": None,
            "KGE": None,
            "FHV": None,
            "FLV": None,
        }
        for basin in basin_ids
    }

    # Extract epoch-wise data for plotting
    epoch_data = {
        basin: {
            "epochs": [],
            "train_losses": [],
            "validation_losses": [],
            "nse_values": [],
        }
        for basin in basin_ids
    }

    # Iterate through each epoch
    for run in data["run"]:
        epoch = run["epoch"]
        train_loss = float(run["train_loss"])
        validation_loss_str = run["validation_loss"]
        validation_loss = float(
            validation_loss_str.split("(")[1].split(",")[0]
        )  # Extract validation loss

        # Check each basin's metrics at this epoch
        for i, basin in enumerate(basin_ids):
            nse = run["validation_metric"]["NSE of streamflow"][i]
            rmse = run["validation_metric"]["RMSE of streamflow"][i]
            corr = run["validation_metric"]["Corr of streamflow"][i]
            kge = run["validation_metric"]["KGE of streamflow"][i]
            fhv = run["validation_metric"]["FHV of streamflow"][i]
            flv = run["validation_metric"]["FLV of streamflow"][i]

            # Store data for plotting
            epoch_data[basin]["epochs"].append(epoch)
            epoch_data[basin]["train_losses"].append(train_loss)
            epoch_data[basin]["validation_losses"].append(validation_loss)
            epoch_data[basin]["nse_values"].append(nse)

            # If this epoch's validation loss for this basin is the smallest so far, update the metrics
            if validation_loss < min_loss_metrics[basin]["validation_loss"]:
                min_loss_metrics[basin].update(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "NSE": nse,
                        "RMSE": rmse,
                        "Corr": corr,
                        "KGE": kge,
                        "FHV": fhv,
                        "FLV": flv,
                    }
                )

    # Write min-loss metrics to CSV for each basin
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for basin, metrics in sorted(min_loss_metrics.items()):
            writer.writerow(
                [
                    basin,
                    metrics["epoch"],  # Keep epoch as is
                    f"{metrics['train_loss']:.3f}",
                    f"{metrics['validation_loss']:.3f}",
                    f"{metrics['NSE']:.3f}" if metrics["NSE"] is not None else None,
                    f"{metrics['RMSE']:.3f}" if metrics["RMSE"] is not None else None,
                    f"{metrics['Corr']:.3f}" if metrics["Corr"] is not None else None,
                    f"{metrics['KGE']:.3f}" if metrics["KGE"] is not None else None,
                    f"{metrics['FHV']:.3f}" if metrics["FHV"] is not None else None,
                    f"{metrics['FLV']:.3f}" if metrics["FLV"] is not None else None,
                ]
            )
