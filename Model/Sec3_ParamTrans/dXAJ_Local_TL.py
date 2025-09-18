# -*- coding: utf-8 -*-
# @Author: Yikai CHAI
# @Date:   2025-06-11 15:33:56
# @Last Modified by:   Yikai CHAI
# @Last Modified time: 2025-09-05 14:15:53
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

# 流域文件名到代码的映射
basin_name_to_code = {
    "anhui_50406910": "A01",
    "anhui_50501200": "A02",
    "anhui_50701100": "A03",
    "anhui_50913900": "A04",
    "anhui_51004350": "A05",
    "anhui_62549024": "A06",
    "anhui_62700110": "A07",
    "anhui_62700700": "A08",
    "anhui_62802400": "A09",
    "anhui_62802700": "A10",
    "anhui_62803300": "A11",
    "anhui_62906900": "A12",
    "anhui_62907100": "A13",
    "anhui_62907600": "A14",
    "anhui_62907601": "A15",
    "anhui_62909400": "A16",
    "anhui_62911200": "A17",
    "anhui_62916110": "A18",
    "anhui_70112150": "A19",
    "anhui_70114100": "A20"
}

# 目标流域到最佳源流域的映射
target_to_source = {
    "A01": "A05",
    "A02": "A10",
    "A03": "A10",
    "A04": "A05",
    "A05": "A18",
    "A06": "A07",
    "A07": "A16",
    "A08": "A15",
    "A09": "A07",
    "A10": "A02",
    "A11": "A02",
    "A12": "A13",
    "A13": "A12",
    "A14": "A15",
    "A15": "A14",
    "A16": "A07",
    "A17": "A07",
    "A18": "A05",
    "A19": "A20",
    "A20": "A19"
}

# 流域代码到文件名的反向映射
basin_code_to_name = {v: k for k, v in basin_name_to_code.items()}

def load_basin_ids(csv_path):
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()

def get_basin_code_from_path(csv_path):
    """从CSV路径中提取流域文件名并映射为代码"""
    basename = os.path.basename(csv_path)
    filename = os.path.splitext(basename)[0]
    # 移除末尾的数字（如_20, _34等）
    basin_name = "_".join(filename.split("_")[:-1])
    return basin_name_to_code.get(basin_name, basin_name)

def get_source_basin_filename(target_basin_code):
    """获取目标流域对应的源流域完整文件名（包含数字后缀）"""
    source_basin_code = target_to_source.get(target_basin_code)
    if source_basin_code:
        source_basin_name = basin_code_to_name.get(source_basin_code)
        if source_basin_name:
            # 从csv_paths中找到对应的完整文件名
            for csv_path in csv_paths:
                if source_basin_name in csv_path:
                    basename = os.path.basename(csv_path)
                    return os.path.splitext(basename)[0]
    return "unknown"

def get_project_name(csv_path, target_basin_code):
    """生成项目名称，包含迁移学习标识（使用源流域完整文件名）"""
    basename = os.path.basename(csv_path)
    filename = os.path.splitext(basename)[0]  # 目标流域完整文件名
    source_filename = get_source_basin_filename(target_basin_code)  # 源流域完整文件名
    return os.path.join("Anhui_dPL_TL", f"{filename}_b0500_fl240_lr005_seed1111_tl_{source_filename}")

def get_model_and_stat_paths(target_basin_code):
    """根据目标流域代码获取模型和统计文件的路径"""
    source_basin_code = target_to_source.get(target_basin_code)
    if not source_basin_code:
        raise ValueError(f"找不到目标流域 {target_basin_code} 对应的源流域")
    
    # 获取源流域的文件名
    source_basin_name = basin_code_to_name.get(source_basin_code)
    if not source_basin_name:
        raise ValueError(f"找不到流域代码 {source_basin_code} 对应的文件名")
    
    # 构建源流域的完整文件名（添加数字后缀）
    # 需要根据实际情况确定数字后缀，这里假设可以从已有的csv_paths中匹配
    source_csv_path = None
    for csv_path in csv_paths:
        if source_basin_name in csv_path:
            source_csv_path = csv_path
            break
    
    if not source_csv_path:
        raise ValueError(f"找不到流域 {source_basin_name} 对应的CSV文件")
    
    # 从CSV路径中提取完整的文件名（包含数字后缀）
    source_basename = os.path.basename(source_csv_path)
    source_filename = os.path.splitext(source_basename)[0]
    
    # 构建模型和统计文件路径
    base_dir = "Result/Sec2_ModelPerf/Anhui_dPL"
    model_dir = f"{source_filename}_b0500_fl240_lr005_seed1111"
    
    model_path = os.path.join(base_dir, model_dir, "best_model.pth")
    stat_file_path = os.path.join(base_dir, model_dir, "dapengscaler_stat.json")
    
    return model_path, stat_file_path

def dxaj_hydrodataset_args(basin_ids, project_name, model_path, stat_file_path):
    train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
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
        batch_size=500,
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
        train_mode=False,
        stat_dict_file=stat_file_path,
        rs=1111,
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
        loss_func="RMSEFlood",
        # 10. 评估配置
        model_loader={"load_way": "best"},
        fill_nan=["no"],
        metrics=["NSE", "KGE", "RMSE", "Corr", "PFE", "PTE"],
        evaluator={"eval_way": "1pace", "pace_idx": -1},
    )


def run_single_dpl_exp(csv_path):
    print(f"开始处理: {csv_path}")
    # 获取目标流域代码
    target_basin_code = get_basin_code_from_path(csv_path)
    print(f"目标流域代码: {target_basin_code}")
    # 获取对应的模型和统计文件路径
    model_path, stat_file_path = get_model_and_stat_paths(target_basin_code)
    print(f"使用模型路径: {model_path}")
    print(f"使用统计文件路径: {stat_file_path}")
    cfg = default_config_file()
    basin_ids = load_basin_ids(csv_path)
    project_name = get_project_name(csv_path, target_basin_code)
    args_ = dxaj_hydrodataset_args(basin_ids, project_name, model_path, stat_file_path)
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print(f"完成处理: {csv_path}")
    return True


def run_sequential_lstm_exp(csv_paths):
    print(f"开始顺序处理 {len(csv_paths)} 个流域")
    success_count = 0
    failed_count = 0
    
    for csv_path in csv_paths:
        try:
            result = run_single_dpl_exp(csv_path)
            if result:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"处理 {csv_path} 时发生错误: {e}")
            failed_count += 1
    
    print(f"所有任务已完成！成功: {success_count}, 失败: {failed_count}")

if __name__ == "__main__":
    run_sequential_lstm_exp(csv_paths)