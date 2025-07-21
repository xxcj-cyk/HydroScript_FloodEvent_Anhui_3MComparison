import os
import json
import glob
import xarray as xr
import csv
import pandas as pd
import numpy as np

# 定义项目和路径
base_project_name = "anhui21_797_PET_Anhui_lr"
model_name = "Anhui_LSTM"
root_dir = f"./Result/{model_name}/{base_project_name}"
output_dir = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}"
csv_file = f"{output_dir}/{model_name}_{base_project_name}_valid_metrics.csv"

# 从NetCDF文件中读取流域ID顺序
nc_file_path = os.path.join(root_dir, "epoch_best_flow_pred.nc")
nc_data = xr.open_dataset(nc_file_path)
basin_ids = list(nc_data["basin"].values)

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 初始化CSV文件并写入表头
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Basin_ID",
            "Epoch",
            "Train_Loss",
            "Validation_Loss",
            "NSE_Validation",
            "KGE_Validation",
            "RMSE_Validation",
            "Corr_Validation",
            "PFE_Validation",
            "PTE_Validation",
        ]
    )

# 使用glob查找匹配的JSON文件
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
            "KGE": None,
            "RMSE": None,
            "Corr": None,
            "PFE": None,
            "PTE": None,
        }
        for basin in basin_ids
    }

    # 迭代每个epoch
    for run in data["run"]:
        epoch = run["epoch"]
        train_loss = float(run["train_loss"])

        # 提取验证损失
        validation_loss_str = run["validation_loss"]
        validation_loss = float(
            validation_loss_str.split("(")[1].split(",")[0]
        )  # Check each basin's metrics at this epoch
        for i, basin in enumerate(basin_ids):
            nse = run["validation_metric"]["NSE of streamflow"][i]
            kge = run["validation_metric"]["KGE of streamflow"][i]
            rmse = run["validation_metric"]["RMSE of streamflow"][i]
            corr = run["validation_metric"]["Corr of streamflow"][i]
            pfe = run["validation_metric"]["PFE of streamflow"][i]
            pte = run["validation_metric"]["PTE of streamflow"][i]

            # If this epoch's validation loss for this basin is the smallest so far, update the metrics
            if validation_loss < min_loss_metrics[basin]["validation_loss"]:
                min_loss_metrics[basin].update(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "NSE": nse,
                        "KGE": kge,
                        "RMSE": rmse,
                        "Corr": corr,
                        "PFE": pfe,
                        "PTE": pte,
                    }
                )

    # 将最小损失指标写入CSV（每个流域）
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for basin, metrics in sorted(min_loss_metrics.items()):
            # 准备行数据，处理可能的NaN值
            row_data = [
                basin,
                metrics["epoch"],
                f"{metrics['train_loss']:.3f}",
                f"{metrics['validation_loss']:.3f}",
            ]
            
            # 处理可能的NaN值
            for metric_name, format_spec in [
                ("NSE", ".3f"), 
                ("KGE", ".3f"), 
                ("RMSE", ".2f"), 
                ("Corr", ".2f"), 
                ("PFE", ".1f")
            ]:
                value = metrics[metric_name]
                if value is not None and not np.isnan(value):
                    row_data.append(f"{value:{format_spec}}")
                else:
                    row_data.append(None)
            
            # 特殊处理PTE，它需要转换为整数
            pte_value = metrics["PTE"]
            if pte_value is not None and not np.isnan(pte_value):
                row_data.append(f"{int(pte_value)}")
            else:
                row_data.append(None)
                
            # 写入行数据
            writer.writerow(row_data)
