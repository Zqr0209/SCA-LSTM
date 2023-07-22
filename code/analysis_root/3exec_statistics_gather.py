# 指定模型的所有metric的平均值、中位数进行集合

import pandas as pd

from pathlib import Path


def df_gather(statistics_messages, metrics, days, saving_root):
    df = pd.DataFrame()
    index_ls = list()
    for day in range(days):
        for model_name in statistics_messages:
            day_model = list()
            for metric_name in metrics:
                statistics_root = statistics_messages[model_name]
                metric_file = metrics[metric_name]
                metric_path = statistics_root / metric_file
                metric_value = pd.read_csv(metric_path, header=0, names=[metric_name, "value"])
                day_model_metric = metric_value.iloc[day]["value"]
                day_model.append(day_model_metric)
            df[f"{day + 1}_{model_name}"] = day_model
            index_ls.append([f"Day {day + 1}", model_name])
    multi_index = pd.MultiIndex.from_tuples(index_ls, names=["ith-day-ahead", "model"])
    df_multi_index = pd.DataFrame(df.values, index=metrics.keys(), columns=multi_index)
    df_multi_index = df_multi_index.round(4).applymap(lambda item: "%.4f" % item)
    df_multi_index.to_excel(saving_root / "gathered_statistics.xlsx")
    df_multi_index.to_latex(saving_root / "gathered_statistics.tex")


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent

    # messages = {name: root}
    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_[64-4-4-256-0.1]@daymet673_1980~1995#1995~2000#2000~2014@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune",
    #     "LSTM-MSV-S2S": project_root / "runs" / "baseline_lstm_msv_s2s",
    #     "LSTM-S2S": project_root / "runs" / "baseline_lstm_s2s"
    # }
    # saving_dir = Path("./statistics_gathered_1")

    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_[64-4-4-256-0.1]@daymet673_1980~1995#1995~2000#2000~2014@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune",
    #     "RR-Former (Without Pretraining)": project_root / "runs" / "TransformerUnPretrained_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Fine-tuning)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/pretrain_test_single"
    # }
    # saving_dir = Path("./statistics_gathered_2")

    '''''''''
    messages = {
        "RR-Former (Pretrain + Fine-tune)": project_root / "runs" / "Transformer_NAR_[64-4-4-256-0.1]@daymet673_1980~1995#1995~2000#2000~2014@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune",
        "LSTM-MSV-S2S (Pretrain + Fine-tune)": project_root / "runs" / "LSTMMSVS2S_[[22-7-5-128][15-*-1-128][*-7-256-128]-0.2]@daymet673_1980~1995#1995~2000#2000~2014@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune",
        "LSTM-S2S (Pretrain + Fine-tune)": project_root / "runs" / "LSTMS2S_[[22-7-5-128][15-*-1-128][*-7-256-128]-0.2]@daymet673_1980~1995#1995~2000#2000~2014@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune"
    }
    saving_dir = Path("./statistics_gathered_3")
    '''''''''
    messages = {
        "LSTM": project_root / "runs" / "normal"/"LSTM_None_8_1_[6-1-128-2]@114_AUS_basins_list_1977~2008#2008~2011#2011~2014@8|7+1[6+1]@NSELoss_n200_bs2048_lr0.0001_warm_up@seed2333/fine_tune",
        "LSTMATTN": project_root / "runs" / "normal" / "LSTMATTN_None_[[8-1-6-128][*-1-6-128]-0.2]@114_AUS_basins_list_1977~2008#2008~2011#2011~2014@8|7+1[6+1]@NSELoss_n200_bs2048_lr0.0001_warm_up@seed2333/fine_tune",
    }
    saving_dir = Path("/data1/zqr/RRS-Former/analysis_root/AUS_single")

    saving_dir.mkdir(parents=True, exist_ok=True)
    statistics_messages = {model_name: messages[model_name] / "statistics" for model_name in messages}
    # metrics = {name: file_name}
    metrics = {
        "Median of NSE": "nse_median.csv",
        "Mean of NSE": "nse_mean.csv",
        "Median of RMSE": "nrmse_median.csv",
        "Mean of RMSE": "nrmse_mean.csv",
        "Median of ATPE-2%": "tpe[0.02]_median.csv",
        "Mean of ATPE-2%": "tpe[0.02]_mean.csv",
        "Median of Bias": "bias_median.csv",
        "Mean of Bias": "bias_mean.csv"
    }

    df_gather(statistics_messages, metrics, days=1, saving_root=saving_dir)
