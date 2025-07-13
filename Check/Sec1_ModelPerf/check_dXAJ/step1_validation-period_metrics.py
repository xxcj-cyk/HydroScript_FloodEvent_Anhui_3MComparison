import os
import json
import glob
import xarray as xr
import csv

# 定义项目和路径
base_project_name = "anhui18_691_PET_Anhui"
model_name = "Anhui_dPL"
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

# 处理每个JSON文件
for json_file in json_files:
    # 读取JSON文件
    with open(json_file, "r") as f:
        data = json.load(f)

    # 初始化每个流域的最小验证损失和相应指标跟踪
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
        )   # 检查每个流域在此epoch的指标
        for i, basin in enumerate(basin_ids):
            nse = run["validation_metric"]["NSE of streamflow"][i]
            kge = run["validation_metric"]["KGE of streamflow"][i]
            rmse = run["validation_metric"]["RMSE of streamflow"][i]
            corr = run["validation_metric"]["Corr of streamflow"][i]
            pfe = run["validation_metric"]["PFE of streamflow"][i]
            pte = run["validation_metric"]["PTE of streamflow"][i]

            # 如果此epoch的验证损失是目前为止最小的，则更新指标
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
            writer.writerow(
                [
                    basin,
                    metrics["epoch"],
                    f"{metrics['train_loss']:.3f}",
                    f"{metrics['validation_loss']:.3f}",
                    f"{metrics['NSE']:.3f}" if metrics["NSE"] is not None else None,
                    f"{metrics['KGE']:.3f}" if metrics["KGE"] is not None else None,
                    f"{metrics['RMSE']:.2f}" if metrics["RMSE"] is not None else None,
                    f"{metrics['Corr']:.2f}" if metrics["Corr"] is not None else None,
                    f"{metrics['PFE']:.1f}" if metrics["PFE"] is not None else None,
                    f"{int(metrics['PTE'])}" if metrics["PTE"] is not None else None,
                ]
            )
