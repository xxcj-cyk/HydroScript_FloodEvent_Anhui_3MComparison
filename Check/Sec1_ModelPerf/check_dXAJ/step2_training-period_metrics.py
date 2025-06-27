import os
import pandas as pd
from hydrodata_china.settings.datasets_dir import DATASETS_DIR
from hydromodel_dl.configs.config import default_config_file, update_cfg, cmd
from hydromodel_dl.trainers.trainer import train_and_evaluate

base_project_name = "anhui21_797_PET_Anhui"  # Change the base_project name(xxx_train)
model_name = "Anhui_EnLoss-dPL"
csv_path = r"./Data/All/anhui21_797.csv"


def load_basin_ids(csv_path):
    basin_data = pd.read_csv(csv_path)
    return basin_data["basin"].tolist()


def dxaj_hydrodataset_args(basin_ids):
    project_name = os.path.join(model_name, f"{base_project_name}_train")
    train_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]
    valid_period = ["2024-08-01 00:00:00", "2024-08-31 23:00:00"]
    test_period = ["2024-07-01 00:00:00", "2024-07-31 23:00:00"]

    model_path = os.path.join(
        "Result", model_name, base_project_name, "best_model.pth"
    )  # Change the model path
    stat_file_path = os.path.join(
        "Result", model_name, base_project_name, "dapengscaler_stat.json"
    )  # Change the stat file path

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
            "n_hidden_states": 16,
            "kernel_size": 15,
            "warmup_length": 240,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
        },
        loss_func="MultiOutLoss",  # 替换损失函数
        # loss_param={
        #     "loss_funcs": "RMSESum",
        #     "data_gap": [0, 0],
        #     "device": [2],
        #     "item_weight": [1, 0],
        #     "limit_part": [1],
        # },
        loss_param = {
            "loss_funcs": "Hybrid",
            "data_gap": [0, 0],
            "device": [2],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
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
        batch_size=500,
        min_time_unit="h",
        forecast_history=0,
        forecast_length=240,#warmup_length相同
        var_t=[
            "P_Anhui",
            "PET_Anhui",
            # "PET_ERA5-Land",
            # "ET_ERA5-Land",
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
        var_out=["streamflow", "PET_Anhui"],  # 添加参数
        n_output=2,  # 添加参数
        fill_nan=["no", "no"],  # 添加参数
        target_as_input=0,
        constant_only=0,
        train_epoch=20,
        save_epoch=1,
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
        train_mode=0,
        model_loader={"load_way": "pth", "pth_path": model_path},
        stat_dict_file=stat_file_path,
    )

def run_dpl_exp(csv_path):
    cfg = default_config_file()
    basin_ids = load_basin_ids(csv_path)
    args_ = dxaj_hydrodataset_args(basin_ids)
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print("All processes are finished!")

run_dpl_exp(csv_path)
