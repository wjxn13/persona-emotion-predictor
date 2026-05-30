import pandas as pd
import os

data_dir = r"\.kaggle\文本情绪反应预测\基于性格建模的文本情绪反应预测"

df_ann = pd.read_csv(os.path.join(data_dir, "1_dataset_all_annotators.csv"))
df_group = pd.read_csv(os.path.join(data_dir, "2_dataset_group_consensus.csv"))
df_prof = pd.read_csv(os.path.join(data_dir, "3_annotator_profiles.csv"))

print("=== all_annotators 列名 ===")
print(df_ann.columns.tolist())
print(df_ann.head(2))

print("\n=== annotator_profiles 列名 ===")
print(df_prof.columns.tolist())
print(df_prof.head(2))

