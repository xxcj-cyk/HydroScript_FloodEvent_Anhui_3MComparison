import os
import pandas as pd
import multiprocessing as mp
from hydrodata_china.settings.datasets_dir import DATASETS_DIR
from hydromodel_dl.configs.config import default_config_file, update_cfg, cmd
from hydromodel_dl.trainers.trainer import train_and_evaluate
import glob

# 定义所有CSV文件的路径列表
csv_paths = [
    "./data/anhui_50406910_28.csv",
    "./data/anhui_50501200_36.csv",
    "./data/anhui_50701100_44.csv",
    "./data/anhui_50913900_37.csv",
    "./data/anhui_51004350_21.csv",
    "./data/anhui_62549024_80.csv",
    "./data/anhui_62700110_27.csv",
    "./data/anhui_62700700_41.csv",
    "./data/anhui_62802400_19.csv",
    "./data/anhui_62802700_62.csv",
    "./data/anhui_62803300_87.csv",
    "./data/anhui_62902000_48.csv",
    "./data/anhui_62906900_40.csv",
    "./data/anhui_62907100_26.csv",
    "./data/anhui_62907600_16.csv",
    "./data/anhui_62907601_14.csv",
    "./data/anhui_62909400_62.csv",
    "./data/anhui_62911200_43.csv",
    "./data/anhui_62916110_21.csv",
    "./data/anhui_70112150_10.csv",
    "./data/anhui_70114100_35.csv",
]

# 或者可以使用glob自动获取所有CSV文件
# csv_paths = glob.glob("./data/anhui_*.csv")

def load_basin_ids(csv_path):
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()

def get_project_name(csv_path):
    """根据CSV文件路径生成项目名称"""
    # 从路径中提取文件名（不含扩展名）
    basename = os.path.basename(csv_path)
    filename = os.path.splitext(basename)[0]
    # 创建项目名称
    return os.path.join("Anhui_dPL", f"{filename}rainfall_16")

def dpl_selfmadehydrodataset_args(basin_ids, project_name):
    train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "dataset_type": "CHINA",
            "source_name": "Anhui_1H",
            "source_path": DATASETS_DIR["Anhui_1H"]["EXPORT_DIR"],
            "time_unit": ["1h"],
        },
        # model_type="MTL",
        model_type="Normal",
        ctx=[2],
        model_name="DplLstmXaj",
        
        model_hyperparam={
            "n_input_features": 42,
            "n_output_features": 15,
            "n_hidden_states": 16,#128
            "kernel_size": 15,
            "warmup_length": 240,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
        },
        # loss_func="RMSESum",
        loss_func="MultiOutLoss",  # 替换损失函数
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [2],
            "item_weight": [1, 0],
            "limit_part": [1],
        },  # 添加参数
        dataset="DPLDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "P_Anhui",
            ],
            "pbm_norm": True,
        },
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=500,#可改小，精度可能能变高
        min_time_unit="h",  # 后续自行添加
        forecast_history=0,
        forecast_length=240,#warmup_length相同
        var_t=[
            "P_Anhui",
            "PET_Anhui",
        ],
        var_c=[
            "sgr_dk_sav",  # 河流坡度
            "p_mean",  # 年平均降雨
            "lit_cl_smj",  # 土壤地质
            "wet_cl_smj",  # 湿地
            "glc_cl_smj",  # 土地利用
            "Area",  # shp area calculated by the shapefile
            "pet_mm_syr",  # Potential Evapotranspiration
            "aet_mm_syr",  # Actual Evapotranspiration
            "ele_mt_sav",  # Elevation
            "lka_pc_sse",  # Limnicity (Percent Lake Area)
            "for_pc_sse",  # Forest Cover Extent
            "slp_dg_sav",  # Terrain Slope
            "lkv_mc_usu",  # Lake Volume
            "ero_kh_sav",  # Soil Erosion
            "tmp_dc_syr",  # Air Temperature
            "riv_tc_usu",  # River Volume
            "gdp_ud_sav",  # Gross Domestic Product
            "soc_th_sav",  # Organic Carbon Content in Soil
            "kar_pc_sse",  # Karst Area Extent
            "ppd_pk_sav",  # Population Density
            "nli_ix_sav",  # Nighttime Lights
            "pst_pc_sse",  # Pasture Extent
            "pac_pc_sse",  # Protected Area Extent
            "snd_pc_sav",  # Sand Fraction in Soil
            "pop_ct_usu",  # Population Count
            "swc_pc_syr",  # Soil Water Content
            "ria_ha_usu",  # River Area
            "slt_pc_sav",  # Silt Fraction in Soil
            "cly_pc_sav",  # Clay Fraction in Soil
            "crp_pc_sse",  # Cropland Extent
            "inu_pc_slt",  # Inundation Extent
            "cmi_ix_syr",  # Climate Moisture Index
            "snw_pc_syr",  # Snow Cover Extent
            "ari_ix_sav",  # Global Aridity Index
            "ire_pc_sse",  # Irrigated Area Extent (Equipped)
            "rev_mc_usu",  # Reservoir Volume
            "inu_pc_smn",  # Inundation Extent
            "urb_pc_sse",  # Urban Extent
            "prm_pc_sse",  # Permafrost Extent
            "gla_pc_sse",  # Glacier Extent
        ],
        var_out=["streamflow", "PET_Anhui"],  # 添加参数
        n_output=2,  # 添加参数
        fill_nan=["no", "no"],  # 添加参数
        target_as_input=0,
        constant_only=0,
        train_epoch=20,
        save_epoch=1,
        model_loader={"load_way": "best"},
        warmup_length=240,#warmup_length同前
        opt="Adam",
        opt_param={
            "lr": 0.005,
        },
        lr_scheduler={
            "lr": 0.005,
            "lr_factor": 0.95,
        },
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "Corr", "KGE", "FHV", "FLV"],
        gage_id=basin_ids,
    )


def run_single_dpl_exp(csv_path):
    """处理单个CSV文件的训练过程"""
    print(f"开始处理: {csv_path}")
    cfg = default_config_file()
    basin_ids = load_basin_ids(csv_path)
    project_name = get_project_name(csv_path)
    args_ = dpl_selfmadehydrodataset_args(basin_ids, project_name)
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
    num_processes = 4
    # 运行并行训练
    run_parallel_dpl_exp(csv_paths, num_processes)