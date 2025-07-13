import os
import pandas as pd
from hydromodel_dl.datasets.data_readers import DATASETS_DIR_CHINA as DATASETS_DIR
from hydromodel_dl.configs.config import default_config_file, update_cfg, cmd
from hydromodel_dl.trainers.trainer import train_and_evaluate

csv_path = r"./data/anhui_62549024_80.csv"

def load_basin_ids(csv_path):
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()


def dpl_selfmadehydrodataset_args(basin_ids):
    project_name = os.path.join("Anhui_dPL", "TL_anhui_62549024_80rainfall_train")
    # train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    # valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    # test_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]

    model_path = os.path.join("results/Anhui_dPL/anhui_12basin_404rainfall/best_model.pth")
    stat_file_path = os.path.join("results/Anhui_dPL/anhui_12basin_404rainfall/dapengscaler_stat.json")

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
        # train_period=train_period,
        # valid_period=valid_period,
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
            "pet_mm_syr",  # Potential Evapotranspiration: Category = Climate; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {yr} annual average
            "aet_mm_syr",  # Actual Evapotranspiration: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
            "ele_mt_sav",  # Elevation: Category = Physiography; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "lka_pc_sse",  # Limnicity (Percent Lake Area): Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "for_pc_sse",  # Forest Cover Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "slp_dg_sav",  # Terrain Slope: Category = Physiography; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "lkv_mc_usu",  # Lake Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
            "ero_kh_sav",  # Soil Erosion: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "tmp_dc_syr",  # Air Temperature: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
            "riv_tc_usu",  # River Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
            "gdp_ud_sav",  # Gross Domestic Product: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "soc_th_sav",  # Organic Carbon Content in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "kar_pc_sse",  # Karst Area Extent: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "ppd_pk_sav",  # Population Density: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "nli_ix_sav",  # Nighttime Lights: Category = Anthropogenic; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "pst_pc_sse",  # Pasture Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "pac_pc_sse",  # Protected Area Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "snd_pc_sav",  # Sand Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "pop_ct_usu",  # Population Count: Category = Anthropogenic; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
            "swc_pc_syr",  # Soil Water Content: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
            "ria_ha_usu",  # River Area: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
            "slt_pc_sav",  # Silt Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "cly_pc_sav",  # Clay Fraction in Soil: Category = Soils & Geology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "crp_pc_sse",  # Cropland Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "inu_pc_slt",  # Inundation Extent: Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {lt} long-term maximum
            "cmi_ix_syr",  # Climate Moisture Index: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
            "snw_pc_syr",  # Snow Cover Extent: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {yr} annual average
            "ari_ix_sav",  # Global Aridity Index: Category = Climate; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {av} average
            "ire_pc_sse",  # Irrigated Area Extent (Equipped): Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "rev_mc_usu",  # Reservoir Volume: Category = Hydrology; Spatial Extent = {u} in total watershed upstream of sub-basin pour point; Dimensions = {su} sum
            "inu_pc_smn",  # Inundation Extent: Category = Hydrology; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {mn} annual minimum
            "urb_pc_sse",  # Urban Extent: Category = Anthropogenic; Spatial Extent = {s} in reach catchment; Dimensions = {se} spatial extent (%)
            "prm_pc_sse",  # Permafrost Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
            "gla_pc_sse",  # Glacier Extent: Category = Landcover; Spatial Extent = {s} at sub-bsin pour point; Dimensions = {se} spatial extent (%)
        ],
        # var_out=["streamflow"],  # 添加参数
        var_out=["streamflow", "PET_Anhui"],  # 添加参数
        n_output=2,  # 添加参数
        fill_nan=["no", "no"],  # 添加参数
        target_as_input=0,
        constant_only=0,
        train_epoch=10,
        save_epoch=1,
        train_mode=False,
        model_loader={"load_way": "pth", "pth_path": model_path},
        stat_dict_file=stat_file_path,
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


def run_dpl_exp(csv_path):
    cfg = default_config_file()
    basin_ids = load_basin_ids(csv_path)
    args_ = dpl_selfmadehydrodataset_args(basin_ids)
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print("All processes are finished!")

run_dpl_exp(csv_path)