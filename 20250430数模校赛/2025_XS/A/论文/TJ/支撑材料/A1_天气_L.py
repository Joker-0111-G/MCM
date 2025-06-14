import pandas as pd
import numpy as np
import os


def process_group(group):
    """处理时间重复的分组数据（增强版）"""
    if len(group) == 1:
        return group.iloc[0]

    # 列分类定义
    mean_cols = ["最高温度", "最低温度", "风速", "湿度"]
    mode_cols = ["当前温度", "天气", "风向"]
    sun_cols = ["日出时间", "日落时间"]

    # ====== 均值列处理 ======
    try:
        # 计算初始均值（至少需要1个有效值）
        initial_mean = group[mean_cols].mean(skipna=True)

        # 填充缺失值并计算距离
        filled_data = group[mean_cols].fillna(initial_mean)
        distances = (filled_data - initial_mean).abs().sum(axis=1)

        # 删除距离最大的行（至少保留1条）
        filtered_group = group.drop(distances.idxmax())

        # 计算最终均值（允许全空值）
        final_mean = filtered_group[mean_cols].mean(skipna=True)
    except Exception as e:
        print(f"处理分组时发生错误（时间：{group.name}）: {str(e)}")
        final_mean = pd.Series([np.nan] * len(mean_cols), index=mean_cols)

    # ====== 众数列处理 ======
    mode_values = {}
    for col in mode_cols:
        try:
            # 使用原始数据计算众数（非处理后的数据）
            non_null = group[col].dropna()
            if len(non_null) > 0:
                mode_val = non_null.mode()
                mode_values[col] = mode_val[0] if not mode_val.empty else np.nan
            else:
                mode_values[col] = np.nan
        except:
            mode_values[col] = np.nan

    # ====== 日出日落时间处理 ======
    sun_values = {}
    for col in sun_cols:
        try:
            # 优先使用处理后的数据，其次用原始数据
            non_null = filtered_group[col].dropna()
            if non_null.empty:
                non_null = group[col].dropna()
            sun_values[col] = non_null.iloc[0] if not non_null.empty else np.nan
        except:
            sun_values[col] = np.nan

    return pd.Series({
        **final_mean.to_dict(),
        **mode_values,
        **sun_values,
        "时间": group.name
    })


# ====== 主处理流程 ======
if __name__ == "__main__":
    # 读取原始数据
    input_path = "电站4天气数据.xlsx"
    origin_df = pd.read_excel(input_path)

    # 时间格式处理
    origin_df["时间"] = pd.to_datetime(origin_df["时间"], errors="coerce")

    # 步骤1：数据拆分
    duplicate_mask = origin_df.duplicated("时间", keep=False)
    duplicate_df = origin_df[duplicate_mask].sort_values("时间")
    unique_df = origin_df[~duplicate_mask]

    # 步骤2：处理重复数据
    processed_data = []
    if not duplicate_df.empty:
        for time, group in duplicate_df.groupby("时间"):
            try:
                processed_row = process_group(group)
                processed_data.append(processed_row)
            except Exception as e:
                print(f"处理时间 {time} 时发生严重错误: {str(e)}")
                continue

    # 创建结果DataFrame
    processed_df = pd.DataFrame(processed_data, columns=origin_df.columns)

    # 步骤3：合并数据
    final_df = pd.concat([unique_df, processed_df], axis=0)

    # 最终排序去重（处理后的数据优先）
    final_df = final_df.sort_values("时间", ascending=True)
    final_df = final_df.drop_duplicates("时间", keep="first")

    # 创建输出目录
    output_dir = "A1_天气数据_重复值整合"
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果（保留原始格式）
    output_path = os.path.join(output_dir, "电站4天气数据整合.xlsx")
    final_df.to_excel(output_path, index=False, engine="openpyxl")

    # 打印验证信息
    print("\n" + "=" * 40)
    print(f"原始数据记录数: {len(origin_df)}")
    print(f"发现重复时间数: {len(duplicate_df.groupby('时间'))}")
    print(f"最终有效记录数: {len(final_df)}")
    print(f"已清除重复记录: {len(origin_df) - len(final_df)}")
    print(f"结果文件已保存至: {os.path.abspath(output_path)}")
    print("=" * 40)