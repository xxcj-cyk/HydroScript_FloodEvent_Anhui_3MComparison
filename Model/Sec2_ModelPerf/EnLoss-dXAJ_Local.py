# -*- coding: utf-8 -*-
# @Author: Yikai CHAI
# @Date:   2025-06-22 08:34:40
# @Last Modified by:   Yikai CHAI
# @Last Modified time: 2025-09-06 20:16:03
import os
import pandas as pd
import multiprocessing as mp
from hydromodel_dl.datasets.data_readers import DATASETS_DIR_CHINA as DATASETS_DIR
from hydromodel_dl.configs.config import default_config_file, update_cfg, cmd
from hydromodel_dl.trainers.trainer import train_and_evaluate

csv_paths = [
    "./Data/Select/anhui_50406910_20.csv",
    "./Data/Select/anhui_50501200_34.csv",
    "./Data/Select/anhui_50701100_41.csv",
    "./Data/Select/anhui_50913900_24.csv",
    "./Data/Select/anhui_51004350_18.csv",
    "./Data/Select/anhui_62549024_78.csv",
    "./Data/Select/anhui_62700110_27.csv",
    "./Data/Select/anhui_62700700_38.csv",
    "./Data/Select/anhui_62802400_17.csv",
    "./Data/Select/anhui_62802700_61.csv",
    "./Data/Select/anhui_62803300_87.csv",
    "./Data/Select/anhui_62906900_38.csv",
    "./Data/Select/anhui_62907100_25.csv",
    "./Data/Select/anhui_62907600_15.csv",
    "./Data/Select/anhui_62907601_14.csv",
    "./Data/Select/anhui_62909400_62.csv",
    "./Data/Select/anhui_62911200_43.csv",
    "./Data/Select/anhui_62916110_20.csv",
    "./Data/Select/anhui_70112150_10.csv",
    "./Data/Select/anhui_70114100_33.csv",
]


def load_basin_ids(csv_path):
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()


def get_project_name(csv_path):
    basename = os.path.basename(csv_path)
    filename = os.path.splitext(basename)[0]
    # 创建项目名称
    return os.path.join("Anhui_EnLoss-dPL", f"{filename}_b0100_fl240_lr005_w1_seed1111")


def dxaj_hydrodataset_args(basin_ids, project_name):
    train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    return cmd(
        # 1. 项目和基础配置
        sub=project_name,
        ctx=[2],
        gage_id=basin_ids,
        # 2. 数据源配置
        source_cfgs={
            "dataset_type": "CHINA",
            "source_name": "Anhui_1H",
            "source_path": DATASETS_DIR["Anhui_1H"]["EXPORT_DIR"],
            "time_unit": ["1h"],
        },
        # 3. 数据集配置
        dataset="FloodEventDplDataset",
        min_time_unit="h",
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=100,
        # 4. 特征和预测设置
        var_t=[
            "p_anhui",
            "pet_anhui",
        ],
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
        forecast_length=240,
        which_first_tensor="sequence",
        target_as_input=0,
        constant_only=0,
        # 5. 数据缩放器配置
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "p_anhui",
            ],
            "pbm_norm": True,
        },
        # 6. 模型配置
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
        # 7. 训练配置
        train_epoch=20,
        save_epoch=1,
        warmup_length=240,
        # 8. 优化器配置
        opt="Adam",
        opt_param={
            "lr": 0.005,
        },
        lr_scheduler={
            "lr": 0.005,
            "lr_factor": 0.95,
        },
        # 9. 损失函数配置
        loss_func="HybridFlood",
        loss_param={
            "mae_weight": 1,
        },
        # 10. 评估配置
        model_loader={"load_way": "best"},
        fill_nan=["no"],
        metrics=["NSE", "KGE", "RMSE", "Corr", "PFE", "PTE"],
        evaluator={"eval_way": "1pace", "pace_idx": -1},
    )


def run_single_dpl_exp(csv_path):
    """处理单个CSV文件的训练过程"""
    print(f"开始处理: {csv_path}")
    cfg = default_config_file()
    basin_ids = load_basin_ids(csv_path)
    project_name = get_project_name(csv_path)
    args_ = dxaj_hydrodataset_args(basin_ids, project_name)
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print(f"完成处理: {csv_path}")
    return True


def run_parallel_dpl_exp(csv_paths, num_processes=4):
    """并行处理多个CSV文件的训练过程"""
    print(f"开始并行处理 {len(csv_paths)} 个流域，使用 {num_processes} 个进程")
    # 创建进程池
    pool = mp.Pool(processes=num_processes)
    # 提交所有任务到进程池
    results = pool.map(run_single_dpl_exp, csv_paths)
    # 关闭进程池
    pool.close()
    pool.join()
    # 统计成功和失败的任务
    success_count = sum(results)
    failed_count = len(results) - success_count
    print(f"所有任务已完成！成功: {success_count}, 失败: {failed_count}")


if __name__ == "__main__":
    # 设置并行进程数，这里使用4个进程
    num_processes = 5
    # 运行并行训练
    run_parallel_dpl_exp(csv_paths, num_processes)
