# 算出指定analyse_root下所有basin的指定metric的平均值、中位数（需要先运行exec_calc_metrics得到指定metric的值）

import time
import pandas as pd
from analysis_config import AnalysisConfig


def calc_metric(analyse_root, saved_metric_name):
    result_root = analyse_root / "statistics"
    result_root.mkdir(exist_ok=True, parents=True)
    path = analyse_root / f"calc_{saved_metric_name}.csv"
    df = pd.read_csv(path, sep=',', header=0, dtype={"basin": str})
    df = df.set_index("basin")
    metrics_mean = df.mean()
    metrics_median = df.median()
    metrics_mean.to_csv(result_root / f"{saved_metric_name}_mean.csv", mode="w")
    metrics_median.to_csv(result_root / f"{saved_metric_name}_median.csv", mode="w")

    print(f"{saved_metric_name} completed.")


if __name__ == '__main__':
    analyse_root = AnalysisConfig.analyse_root
    s = time.time()
    alpha = 0.02
    # metric_name_list = ["nse"]
    metric_name_list = ["nse", "kge", f"tpe[{alpha}]", "bias", "nrmse", "ae", "logNSE"]
    for metric in metric_name_list:
        calc_metric(analyse_root, metric)

    e = time.time()
    print(f"used time:{e - s}s.")
