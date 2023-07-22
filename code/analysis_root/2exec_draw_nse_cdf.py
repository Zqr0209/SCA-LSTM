import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plt.style.use("ggplot")


def single_model_nses(nses_path):
    df = pd.read_csv(nses_path, sep=',', header=0, dtype={"basin": str})
    nse_df = df.drop("basin", axis=1)
    other_df = df[["basin"]]

    return nse_df, other_df


def muti_model_nses(nses_paths):
    df = []
    max_days = 0
    for model_number, nses_path in enumerate(nses_paths):
        nse_df, other_df = single_model_nses(nses_path)
        days = nse_df.shape[1]
        max_days = max(max_days, days)
        new_names = [f"{model_number}_nse{day}" for day in range(days)]
        nse_df.columns = new_names
        if model_number == 0:
            df = other_df

        df = pd.concat([df, nse_df], axis=1)

    return df, max_days


def draw_cdf_wide(nses_paths, labels, saving_dir, fontsize):
    total_number = len(nses_paths)
    df, max_days = muti_model_nses(nses_paths)
    cut = 9900
    row = 2
    col = 4
    fig, axes = plt.subplots(row, col, figsize=(24, 10))
    for day in range(max_days):
        if day + 1 == 1:
            title = f"{str(day + 1)}st-day-ahead"
        elif day + 1 == 2:
            title = f"{str(day + 1)}nd-day-ahead"
        elif day + 1 == 3:
            title = f"{str(day + 1)}rd-day-ahead"
        else:
            title = f"{str(day + 1)}th-day-ahead"
        r = day // col
        c = day % col
        axes[r][c].set_title(title, fontsize=fontsize)
        for i, label in zip(range(total_number), labels):
            try:
                nse = df[f"{i}_nse{day}"]
                nse_number = nse.shape[0]
                hist, bin_edges = np.histogram(nse, range=(-100, 1), bins=np.linspace(-99, 1, 10001), density=False)
                hist = np.cumsum(hist) / nse_number
                hist = hist[cut:]
                bin_edges = bin_edges[cut:-1]
                axes[r][c].tick_params(labelsize=fontsize)
                axes[r][c].xaxis.set_ticks(np.arange(-1, 1.01, 0.2))
                axes[r][c].plot(bin_edges, hist, label=label)
                # axes[r][c].legend(fontsize=fontsize)
            except:
                continue
    fig.delaxes(axes[1][3])
    fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(labels), fontsize=fontsize)
    fig.savefig(saving_dir / f"nse_cdf_{max_days}days.eps", format="eps")
    fig.savefig(saving_dir / f"nse_cdf_{max_days}days.jpg", format="jpg")


def draw_cdf_thin(nses_paths, labels, saving_dir, fontsize):
    total_number = len(nses_paths)
    df, max_days = muti_model_nses(nses_paths)
    cut = 9900
    row = 4
    col = 2
    fig, axes = plt.subplots(row, col, figsize=(12, 18))
    for day in range(max_days):
        if day + 1 == 1:
            title = f"{str(day + 1)}st-day-ahead"
        elif day + 1 == 2:
            title = f"{str(day + 1)}nd-day-ahead"
        elif day + 1 == 3:
            title = f"{str(day + 1)}rd-day-ahead"
        else:
            title = f"{str(day + 1)}th-day-ahead"
        r = day // col
        c = day % col
        axes[r][c].set_title(title, fontsize=fontsize)
        for i, label in zip(range(total_number), labels):
            try:
                nse = df[f"{i}_nse{day}"]
                nse_number = nse.shape[0]
                hist, bin_edges = np.histogram(nse, range=(-100, 1), bins=np.linspace(-99, 1, 10001), density=False)
                hist = np.cumsum(hist) / nse_number
                hist = hist[cut:]
                bin_edges = bin_edges[cut:-1]
                axes[r][c].tick_params(labelsize=fontsize)
                axes[r][c].xaxis.set_ticks(np.arange(-1, 1.01, 0.2))
                axes[r][c].plot(bin_edges, hist, label=label)
                # axes[r][c].legend(fontsize=fontsize)
            except:
                continue
    fig.delaxes(axes[3][1])
    fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.1)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=len(labels), fontsize=fontsize)
    fig.savefig(saving_dir / f"nse_cdf_{max_days}days.eps", format="eps")
    fig.savefig(saving_dir / f"nse_cdf_{max_days}days.jpg", format="jpg")


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent

    # messages = {name: root}
    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "LSTM-MSV-S2S": project_root / "runs" / "baseline_lstm_msv_s2s",
    #     "LSTM-S2S": project_root / "runs" / "baseline_lstm_s2s"
    # }
    # saving_dir = Path("./nse_cdf_1")

    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Pretraining)": project_root / "runs" / "TransformerUnPretrained_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Fine-tuning)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/pretrain_test_single"
    # }
    # saving_dir = Path("./nse_cdf_2")

    messages = {
        "RR-Former (With Static)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/pretrain_test_single",
        "RR-Former (Without Static)": project_root / "runs" / "Transformer_NAR_[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[5+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/pretrain_test_single",
        # "LSTMMSVS2S": project_root / "runs" / "LSTMMSVS2S_[[22-7-32-50][15-*-1-20][*-7-70-50]-0.2]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/pretrain_test_single",
        "LSTM (With Static)": project_root / "runs" / "withstatic_lstm_ensemblel@maurer_ext448from531_01~08#99~01#89~99"
    }
    saving_dir = Path("./nse_cdf_3")

    # messages = {
    #     "RR-Former": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (With Static)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet671_1980~1995#1995~2000#2000~2014@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333/fine_tune",
    #     "RR-Former (Without Pretraining)": project_root / "runs" / "TransformerUnPretrained_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/fine_tune",
    #     "RR-Former (Without Fine-tuning)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333/pretrain_test_single"
    # }
    # saving_dir = Path("./nse_cdf_4")

    # messages = {
    #     "RR-Former (With Static)_origin_origin": project_root / "runs" / "withstatic_Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_ext448_01~08#99~01#89~99",
    #     "RR-Former (With Static)_origin": project_root / "runs" / "withstatic_Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_ext448_01~08#99~01#89~99_test/pretrain_test_single",
    #     "RR-Former (With Static)": project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n300_bs512_lr0.001_warm_up@seed2333/pretrain_test_single",
    # }
    # saving_dir = Path("./nse_cdf_5")

    saving_dir.mkdir(parents=True, exist_ok=True)
    nses_paths = [root / "calc_nse.csv" for root in messages.values()]
    # draw_cdf_wide(nses_paths=nses_paths, labels=messages.keys(), saving_dir=saving_dir, fontsize=16)
    draw_cdf_thin(nses_paths=nses_paths, labels=messages.keys(), saving_dir=saving_dir, fontsize=16)
