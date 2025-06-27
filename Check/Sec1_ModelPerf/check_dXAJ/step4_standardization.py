import pandas as pd
import numpy as np

def clean_and_merge_nse(df):
    """
    清理并合并NSE_Train和NSE_Validation列，只保留指定列
    
    参数:
        df: 原始DataFrame
        
    返回:
        处理后的DataFrame，只包含Basin_ID, Train_Loss, Validation_Loss, Epoch, NSE
    """
    # 复制原始DataFrame以避免修改原始数据
    df = df.copy()
    
    # 处理NSE_Train列
    df['NSE_Train'] = df['NSE_Train'].replace('#NAME?', np.nan)  # 修正列名拼写错误
    df['NSE_Train'] = pd.to_numeric(df['NSE_Train'], errors='coerce')
    df['NSE_Train'] = df['NSE_Train'].apply(lambda x: x if abs(x) < 1e6 else np.nan)
    
    # 处理NSE_Validation列
    df['NSE_Validation'] = df['NSE_Validation'].replace('#NAME?', np.nan)
    df['NSE_Validation'] = pd.to_numeric(df['NSE_Validation'], errors='coerce')
    df['NSE_Validation'] = df['NSE_Validation'].apply(lambda x: x if abs(x) < 1e6 else np.nan)
    
    # 创建合并列 - 优先使用验证集的值
    df['NSE'] = df['NSE_Validation'].combine_first(df['NSE_Train'])
    
    # 只保留需要的列
    result = df[['Basin_ID', 'Train_Loss', 'Validation_Loss', 'Epoch', 'NSE']]
    
    return result

# 示例使用
base_project_name = "anhui21_797_PET_Anhui"
model_name = "Anhui_EnLoss-dPL"
combined_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}_metrics.csv"
final_csv = f"./Visualization/Sec1_ModelPerf/{model_name}/{base_project_name}/{model_name}_{base_project_name}.csv"
# 假设您的数据已经读入名为df的DataFrame中
df = pd.read_csv(combined_csv)
# 处理数据
cleaned_df = clean_and_merge_nse(df)
# 保存结果
cleaned_df.to_csv(final_csv, index=False)