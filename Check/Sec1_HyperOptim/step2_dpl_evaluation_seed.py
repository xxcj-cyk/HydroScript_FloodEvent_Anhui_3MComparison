# -*- coding: utf-8 -*-
# @Author: Yikai CHAI
# @Date:   2025-08-31 15:15:09
# @Last Modified by:   Yikai CHAI
# @Last Modified time: 2025-09-04 09:35:36
import os
import json
import glob
import xarray as xr
import pandas as pd
import numpy as np
import csv
import math
from hydrodata_china.settings.datasets_dir import DATASETS_DIR
from hydromodel_dl.configs.config import default_config_file, update_cfg, cmd
from hydromodel_dl.trainers.trainer import train_and_evaluate


# 全局配置
model_name = "Anhui_dPL"

# 支持多个seed、batch_size、forecast_length批量评估
seeds = [1111, 2222, 3333, 4444, 5555]
batch_size_list = [5]
forecast_length_list = [240, 120, 72, 24]
base_prefix = "anhui_70114100_33"

# 生成所有组合的项目名称和配置
project_list = []
for seed in seeds:
    for batch_size in batch_size_list:
        for forecast_length in forecast_length_list:
            project_name = f"{base_prefix}_b{batch_size:04d}_fl{forecast_length:03d}_lr005_seed{seed}"
            project_list.append(
                {
                    "project_name": project_name,
                    "csv_path": "./Data/Select/anhui_70114100_33.csv",
                    "batch_size": batch_size,
                    "forecast_length": forecast_length,
                    "seed": seed,
                }
            )


def load_basin_ids(csv_path):
    """从CSV文件加载流域ID"""
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()


def step1_extract_validation_metrics(current_project_name):
    """步骤1: 从JSON文件提取验证期指标"""
    print(f"步骤1: 提取验证期指标... (项目: {current_project_name})")

    root_dir = f"./Result/{model_name}/{current_project_name}"
    output_dir = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}"
    csv_file = f"{output_dir}/{model_name}_{current_project_name}_valid_metrics.csv"

    # 从NC文件读取流域ID顺序
    nc_file_path = os.path.join(root_dir, "epoch_best_flow_pred.nc")
    nc_data = xr.open_dataset(nc_file_path)
    basin_ids = list(nc_data["basin"].values)
    nc_data.close()

    os.makedirs(output_dir, exist_ok=True)

    # 初始化CSV表头
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

    # 查找JSON日志文件
    json_files = glob.glob(os.path.join(root_dir, "*_2025*.json"))

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # 每个流域的最小验证损失跟踪
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

        for run in data["run"]:
            epoch = run["epoch"]
            train_loss = float(run["train_loss"])
            validation_loss_str = run["validation_loss"]
            validation_loss = float(validation_loss_str.split("(")[1].split(",")[0])

            for i, basin in enumerate(basin_ids):
                nse = run["validation_metric"]["NSE of streamflow"][i]
                kge = run["validation_metric"]["KGE of streamflow"][i]
                rmse = run["validation_metric"]["RMSE of streamflow"][i]
                corr = run["validation_metric"]["Corr of streamflow"][i]
                pfe = run["validation_metric"]["PFE of streamflow"][i]
                pte = run["validation_metric"]["PTE of streamflow"][i]

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

        # 写入最优指标
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
                        (
                            f"{metrics['RMSE']:.2f}"
                            if metrics["RMSE"] is not None
                            else None
                        ),
                        (
                            f"{metrics['Corr']:.2f}"
                            if metrics["Corr"] is not None
                            else None
                        ),
                        (
                            f"{metrics['PFE']:.1f}"
                            if metrics["PFE"] is not None
                            and not math.isnan(metrics["PFE"])
                            else None
                        ),
                        (
                            f"{int(metrics['PTE'])}"
                            if metrics["PTE"] is not None
                            and not math.isnan(metrics["PTE"])
                            else None
                        ),
                    ]
                )

    print(f"验证期指标已保存到: {csv_file}")


def dxaj_hydrodataset_args(
    basin_ids, current_project_name, batch_size, forecast_length
):
    """配置DXAJ水文数据集参数"""
    project_name = os.path.join(model_name, f"{current_project_name}_train")
    train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]

    model_path = os.path.join(
        "Result", model_name, current_project_name, "best_model.pth"
    )
    stat_file_path = os.path.join(
        "Result", model_name, current_project_name, "dapengscaler_stat.json"
    )

    return cmd(
        sub=project_name,
        ctx=[2],
        gage_id=basin_ids,
        source_cfgs={
            "dataset_type": "CHINA",
            "source_name": "Anhui_1H",
            "source_path": DATASETS_DIR["Anhui_1H"]["EXPORT_DIR"],
            "time_unit": ["1h"],
        },
        dataset="FloodEventDplDataset",
        min_time_unit="h",
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=batch_size,
        var_t=["p_anhui", "pet_anhui"],
        var_c=[
            "sgr_dk_sav",
            "p_mean",
            "lit_cl_smj",
            "wet_cl_smj",
            "glc_cl_smj",
            "Area",
            "pet_mm_syr",
            "aet_mm_syr",
            "ele_mt_sav",
            "lka_pc_sse",
            "for_pc_sse",
            "slp_dg_sav",
            "lkv_mc_usu",
            "ero_kh_sav",
            "tmp_dc_syr",
            "riv_tc_usu",
            "gdp_ud_sav",
            "soc_th_sav",
            "kar_pc_sse",
            "ppd_pk_sav",
            "nli_ix_sav",
            "pst_pc_sse",
            "pac_pc_sse",
            "snd_pc_sav",
            "pop_ct_usu",
            "swc_pc_syr",
            "ria_ha_usu",
            "slt_pc_sav",
            "cly_pc_sav",
            "crp_pc_sse",
            "inu_pc_slt",
            "cmi_ix_syr",
            "snw_pc_syr",
            "ari_ix_sav",
            "ire_pc_sse",
            "rev_mc_usu",
            "inu_pc_smn",
            "urb_pc_sse",
            "prm_pc_sse",
            "gla_pc_sse",
        ],
        var_out=["streamflow", "flood_event"],
        n_output=1,
        forecast_history=0,
        forecast_length=forecast_length,
        which_first_tensor="sequence",
        target_as_input=0,
        constant_only=0,
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": ["p_anhui"],
            "pbm_norm": True,
        },
        model_name="DplLstmXaj",
        model_type="Normal",
        model_hyperparam={
            "n_input_features": 42,
            "n_output_features": 15,
            "n_hidden_states": 16,
            "kernel_size": 15,
            "warmup_length": 240,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
        },
        train_epoch=20,
        save_epoch=1,
        warmup_length=240,
        train_mode=0,
        stat_dict_file=stat_file_path,
        opt="Adam",
        opt_param={"lr": 0.005},
        lr_scheduler={"lr": 0.005, "lr_factor": 0.95},
        loss_func="RMSEFlood",
        model_loader={"load_way": "pth", "pth_path": model_path},
        fill_nan=["no"],
        metrics=["NSE", "KGE", "RMSE", "Corr", "PFE", "PTE"],
        evaluator={"eval_way": "1pace", "pace_idx": -1},
    )


def step2_run_training_evaluation(
    current_csv_path, current_project_name, batch_size, forecast_length
):
    """步骤2: 运行训练期评估"""
    print(f"步骤2: 运行训练期评估... (项目: {current_project_name})")

    cfg = default_config_file()
    basin_ids = load_basin_ids(current_csv_path)
    args_ = dxaj_hydrodataset_args(
        basin_ids, current_project_name, batch_size, forecast_length
    )
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print("训练期评估完成!")


def step3_merge_training_validation_data(current_project_name):
    """步骤3: 合并训练和验证数据"""
    print(f"步骤3: 合并训练和验证数据... (项目: {current_project_name})")

    project_name = f"{current_project_name}_train"
    root_dir = f"./Result/{model_name}/{project_name}"
    output_csv = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}/{model_name}_{project_name}_metrics.csv"
    valid_csv = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}/{model_name}_{current_project_name}_valid_metrics.csv"
    combined_csv = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}/{model_name}_{current_project_name}_metrics.csv"

    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 加载单个训练指标文件
    train_metric_file = os.path.join(root_dir, "metric_streamflow.csv")
    if os.path.isfile(train_metric_file):
        train_combined_df = pd.read_csv(train_metric_file)

        # 从NC文件加载流域顺序
        nc_file = os.path.join(root_dir, "epoch_best_model.pth_flow_pred.nc")
        if os.path.isfile(nc_file):
            nc_data = xr.open_dataset(nc_file)

            # 假设'basin'是给出流域顺序的坐标或变量
            basin_order = nc_data["basin"].values
            nc_data.close()

            # 用NC文件中的有序流域替换CSV中的basin_id
            train_combined_df["basin_id"] = basin_order

            # 重命名列以保持一致性
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

            # 将指定列四舍五入以匹配step1格式
            # NSE和KGE: 3位小数
            for col in ["NSE_Train", "KGE_Train"]:
                train_combined_df[col] = train_combined_df[col].round(3)
            # RMSE和Corr: 2位小数
            for col in ["RMSE_Train", "Corr_Train"]:
                train_combined_df[col] = train_combined_df[col].round(2)
            # PFE: 1位小数
            train_combined_df["PFE_Train"] = train_combined_df["PFE_Train"].round(1)
            # PTE: 整数（安全处理NA值）
            train_combined_df["PTE_Train"] = (
                train_combined_df["PTE_Train"].fillna(0).round(0)
            )
            train_combined_df["PTE_Train"] = train_combined_df["PTE_Train"].astype(
                int, errors="ignore"
            )

            # 按Basin_ID排序
            train_combined_df.sort_values(by="Basin_ID", inplace=True)

            # 保存训练数据到CSV
            train_combined_df.to_csv(output_csv, index=False)
            print(f"训练数据已保存到 {output_csv}")
        else:
            print("未找到用于流域顺序的NC文件。")
    else:
        print("未找到训练指标文件。")

    # 检查验证文件是否存在并合并
    if not os.path.isfile(valid_csv):
        print("未找到验证数据文件，合并过程结束。")
    else:
        # 读取验证数据
        valid_df = pd.read_csv(valid_csv)

        # 在Basin_ID上合并训练和验证数据
        merged_df = pd.merge(train_combined_df, valid_df, on="Basin_ID", how="inner")

        # 设置期望的列顺序
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

        # 保存合并数据到CSV，使用正确的数字格式
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
                merged_df[col] = merged_df[col].astype(int, errors="ignore")

        # 保存到CSV
        merged_df.to_csv(combined_csv, index=False)
        print(f"训练和验证数据已合并并保存到 {combined_csv}")


def step4_standardize_data(current_project_name):
    """步骤4: 数据标准化和清理（合并原clean_and_merge_metrics功能）"""
    print(f"步骤4: 数据标准化和清理... (项目: {current_project_name})")

    combined_csv = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}/{model_name}_{current_project_name}_metrics.csv"
    final_csv = f"./Visualization/Sec1_HyperOptim/{model_name}/{current_project_name}/{model_name}_{current_project_name}.csv"

    df = pd.read_csv(combined_csv).copy()

    # 定义需要处理的指标
    metrics = ["NSE", "KGE", "RMSE", "Corr", "PFE", "PTE"]

    if {"NSE_Train", "NSE_Validation"} <= set(df.columns):
        df["NSE_Train"] = pd.to_numeric(
            df["NSE_Train"].replace("#NAME?", np.nan), errors="coerce"
        )
        df["NSE_Validation"] = pd.to_numeric(
            df["NSE_Validation"].replace("#NAME?", np.nan), errors="coerce"
        )
        df["NSE_Train"] = df["NSE_Train"].apply(lambda x: x if abs(x) < 1e6 else np.nan)
        df["NSE_Validation"] = df["NSE_Validation"].apply(
            lambda x: x if abs(x) < 1e6 else np.nan
        )
        mask_use_train = df["NSE_Validation"].isna()
        df["NSE"] = df["NSE_Validation"].combine_first(df["NSE_Train"])
    else:
        mask_use_train = pd.Series(False, index=df.index)

    for metric in metrics[1:]:
        train_col, valid_col = f"{metric}_Train", f"{metric}_Validation"
        if {train_col, valid_col} <= set(df.columns):
            df[train_col] = pd.to_numeric(
                df[train_col].replace("#NAME?", np.nan), errors="coerce"
            )
            df[valid_col] = pd.to_numeric(
                df[valid_col].replace("#NAME?", np.nan), errors="coerce"
            )
            df[train_col] = df[train_col].apply(lambda x: x if abs(x) < 1e6 else np.nan)
            df[valid_col] = df[valid_col].apply(lambda x: x if abs(x) < 1e6 else np.nan)

            if metric in ["PFE", "PTE"]:
                df[metric] = np.where(mask_use_train, df[train_col], df[valid_col])
            else:
                df[metric] = df[valid_col].combine_first(df[train_col])

    cols_to_keep = ["Basin_ID", "Train_Loss", "Validation_Loss", "Epoch"] + metrics
    cleaned_df = df[[col for col in cols_to_keep if col in df.columns]]

    cleaned_df.to_csv(final_csv, index=False)
    print(f"标准化后的数据已保存到: {final_csv}")


def main():
    """主函数：批量执行多个组合的评估流水线"""
    print("开始批量执行 dPL 模型性能评估流水线...")
    print(f"模型名称: {model_name}")
    print(f"组合总数: {len(project_list)}")
    print("-" * 50)

    try:
        for i, proj in enumerate(project_list):
            current_project = proj["project_name"]
            current_csv = proj["csv_path"]
            batch_size = proj["batch_size"]
            forecast_length = proj["forecast_length"]

            print(f"\n处理项目 {i+1}/{len(project_list)}: {current_project}")
            print(f"CSV文件: {current_csv}")
            print(f"batch_size={batch_size}, forecast_length={forecast_length}")
            print("-" * 30)

            # 执行步骤
            step1_extract_validation_metrics(current_project)
            step2_run_training_evaluation(
                current_csv, current_project, batch_size, forecast_length
            )
            step3_merge_training_validation_data(current_project)
            step4_standardize_data(current_project)

            print(f"项目 {current_project} 处理完成！")
            print(
                f"结果文件: ./Visualization/Sec1_HyperOptim/{model_name}/{current_project}/{model_name}_{current_project}.csv"
            )

        print("-" * 50)
        print("所有组合项目处理完成！")

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
