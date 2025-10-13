import os
import pandas as pd
import numpy as np

# 全局配置
model_name = "Anhui_dPL_T"
configuration_name = "_b0500_fl240_lr005_seed1111_t_both"
basin_names = [
    "anhui_50406910_20",
    "anhui_50501200_34",
    "anhui_50701100_41",
    "anhui_50913900_24",
    "anhui_51004350_18",
    "anhui_62549024_78",
    "anhui_62700110_27",
    "anhui_62700700_38",
    "anhui_62802400_17",
    "anhui_62802700_61",
    "anhui_62803300_87",
    "anhui_62906900_38",
    "anhui_62907100_25",
    "anhui_62907600_15",
    "anhui_62907601_14",
    "anhui_62909400_62",
    "anhui_62911200_43",
    "anhui_62916110_20",
    "anhui_70112150_10",
    "anhui_70114100_33",
]
base_project_name = [f"{basin_name}{configuration_name}" for basin_name in basin_names]


def merge_train_valid_metrics(current_project_name):
    """合并训练期和验证期指标数据"""
    print(f"处理项目: {current_project_name}")
    
    # 定义文件路径
    valid_metric_file = f"./Result/{model_name}/{current_project_name}/metric_streamflow.csv"
    train_metric_file = f"./Result/{model_name}/{current_project_name}_train/metric_streamflow.csv"
    
    # 输出目录和文件
    output_dir = f"./Visualization/Sec2_ParamTrans/Transfer/{model_name}/{current_project_name}"
    output_file = f"{output_dir}/{model_name}_{current_project_name}.csv"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取验证期数据
    if not os.path.exists(valid_metric_file):
        print(f"警告: 未找到验证期文件 {valid_metric_file}")
        return None
    
    valid_df = pd.read_csv(valid_metric_file)
    print(f"读取验证期数据: {valid_metric_file}")
    print(f"  - 验证期数据形状: {valid_df.shape}")
    print(f"  - 验证期有效数据: {valid_df['NSE'].notna().sum()} 个流域")
    
    # 读取训练期数据
    if not os.path.exists(train_metric_file):
        print(f"警告: 未找到训练期文件 {train_metric_file}")
        return None
    
    train_df = pd.read_csv(train_metric_file)
    print(f"读取训练期数据: {train_metric_file}")
    print(f"  - 训练期数据形状: {train_df.shape}")
    print(f"  - 训练期有效数据: {train_df['NSE'].notna().sum()} 个流域")
    
    # 合并数据（基于basin_id，使用outer以保留所有流域）
    # 使用suffixes自动给重复的列添加后缀
    merged_df = pd.merge(
        train_df,
        valid_df,
        on='basin_id',
        how='outer',
        suffixes=('_Train', '_Validation')
    )
    
    # 重命名basin_id为Basin_ID
    merged_df = merged_df.rename(columns={'basin_id': 'Basin_ID'})
    
    # 合并指标列（空值会被覆盖）
    # 优先使用验证期数据，如果验证期为空则使用训练期数据
    metrics = ['NSE', 'KGE', 'RMSE', 'Corr', 'PFE', 'PTE']
    for metric in metrics:
        train_col = f"{metric}_Train"
        valid_col = f"{metric}_Validation"
        # combine_first: 如果valid_col是NaN，则用train_col的值替换
        merged_df[metric] = merged_df[valid_col].combine_first(merged_df[train_col])
    
    # 格式化数值
    # NSE和KGE: 3位小数
    for col in ['NSE', 'KGE']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].round(3)
    
    # RMSE和Corr: 2位小数
    for col in ['RMSE', 'Corr']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].round(2)
    
    # PFE: 1位小数
    if 'PFE' in merged_df.columns:
        merged_df['PFE'] = merged_df['PFE'].round(1)
    
    # PTE: 整数
    if 'PTE' in merged_df.columns:
        merged_df['PTE'] = merged_df['PTE'].fillna(0).round(0).astype(int, errors='ignore')
    
    # 只保留Basin_ID和合并后的指标列
    column_order = ['Basin_ID', 'NSE', 'KGE', 'RMSE', 'Corr', 'PFE', 'PTE']
    merged_df = merged_df[[col for col in column_order if col in merged_df.columns]]
    
    # 按Basin_ID排序
    merged_df = merged_df.sort_values('Basin_ID').reset_index(drop=True)
    
    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)
    print(f"合并数据已保存到: {output_file}")
    print(f"共处理 {len(merged_df)} 个流域\n")
    
    return merged_df


def merge_all_projects():
    """合并所有项目的结果"""
    print("=" * 60)
    print("开始合并所有项目的结果")
    print("=" * 60)
    
    all_dfs = []
    
    for project_name in base_project_name:
        df = merge_train_valid_metrics(project_name)
        if df is not None:
            # 添加项目名称列
            df['Project'] = project_name
            all_dfs.append(df)


def main():
    """主函数"""
    print("开始执行迁移学习模型评估数据合并...")
    print(f"模型名称: {model_name}")
    print(f"配置名称: {configuration_name}")
    print(f"共 {len(base_project_name)} 个项目")
    print("-" * 60)
    
    try:
        merge_all_projects()
        print("\n所有处理完成！")
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
