import time
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime


def csv_to_array(path, start_date, end_date):
    df = pd.read_csv(path, sep=',', header=0)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%d")
    df = df.set_index("start_date")
    df = df[start_date:end_date]
    return df.values


def visualize_pred_obs(basin, obs_list, pred_list, date_range, messages, saving_root):
    fig, axes = plt.subplots(7, 1, figsize=(18, 24))
    obs_drawn = False
    for obs, pred, message in zip(obs_list, pred_list, messages):
        if obs_drawn is False:
            obs_drawn = True
            for i in range(obs.shape[1]):
                obs_i = obs[:, i]
                axes[i].plot(date_range, obs_i, label=f"obs-{i + 1}")

        for i in range(pred.shape[1]):
            pred_i = pred[:, i]
            axes[i].plot(date_range, pred_i, label=f"{message}-{i + 1}")
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=60)
            axes[i].set_xticks(range(0, len(date_range), 10))
    fig.savefig(saving_root / f"{basin}_obs_pred.jpg")
    plt.close("all")


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent

    region = "01"  # TODO: only will be used in "LSTM-MSV-S2S" and "LSTM-S2S"
    basin_list = ['01013500']
    # basin_list = ['01022500']

    # messages = {name: root}
    messages = {
        "RR-Former": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
        "LSTM-MSV-S2S": project_root / "runs" / "baseline_lstm_msv_s2s" / region,
        "LSTM-S2S": project_root / "runs" / "baseline_lstm_s2s" / region
    }
    saving_root = Path("./visualize_obs_pred_1")
    #
    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Pretraining)": project_root / "runs" / "TransformerUnPretrained_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Fine-tuning)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/pretrain_test_single"
    # }
    # saving_root = Path("./visualize_obs_pred_2")

    saving_root.mkdir(parents=True, exist_ok=True)
    s = time.time()
    start_date = pd.to_datetime("2001-01-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2001-01-30", format="%Y-%m-%d")
    pd_range = pd.date_range(start_date, end_date)
    date_range = [datetime.strftime(d, "%Y-%m-%d") for d in pd_range]
    for basin in basin_list:
        obs_list = list()
        pred_list = list()
        for model_name in messages:
            analyse_root = messages[model_name]
            basin_root = analyse_root / basin / "obs_pred"
            obs = csv_to_array(basin_root / "obs.csv", start_date, end_date)
            pred = csv_to_array(basin_root / "pred.csv", start_date, end_date)
            obs_list.append(obs)
            pred_list.append(pred)
        visualize_pred_obs(basin, obs_list, pred_list, date_range, messages, saving_root)

        print(f"{basin} loaded.")

    e = time.time()
    print(f"used time:{e - s}s.")
