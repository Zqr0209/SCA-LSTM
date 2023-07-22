# 算出指定analyse_root下所有basin的指定metric的值

import time
import pandas as pd
import numpy as np

from utils.metrics import calc_nse, calc_kge, calc_tpe, calc_bias, cacl_nrmse, cacl_ae, calc_logNSE
from analysis_config import AnalysisConfig


class MetricCalculator:
    @staticmethod
    def csv_to_array(path):
        df = pd.read_csv(path, sep=',', header=0)
        df = df.set_index("start_date")

        return df.values

    def __init__(self, analyse_root, metric_type, *args):
        op_roots = list(analyse_root.glob(f"**/obs_pred"))
        op_roots.sort()
        print("Models total number:", len(op_roots))

        self.analyse_root = analyse_root
        self.op_roots = op_roots
        self.args = args
        if metric_type == "nse":
            self.metric = calc_nse
        elif metric_type == "kge":
            self.metric = calc_kge
        elif metric_type == "tpe":
            self.metric = calc_tpe
        elif metric_type == "bias":
            self.metric = calc_bias
        elif metric_type == "nrmse":
            self.metric = cacl_nrmse
        elif metric_type == "ae":
            self.metric = cacl_ae
        elif metric_type == "logNSE":
            self.metric = calc_logNSE
        else:
            raise RuntimeError(f"No such metric type: {metric_type}")

    def calc_metric(self, obs, pred):
        return self.metric(obs, pred, *self.args)

    def calc_and_save(self, saved_metric_name):
        # basin_mark = "448"
        # basins_file = f"../data/{basin_mark}basins_list.txt"
        # specific_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()

        file_flag = False
        for op_root in self.op_roots:
            basin_root = op_root.parent
            basin = basin_root.name
            # if basin not in specific_basins_list:
            #     continue
            obs = self.csv_to_array(op_root / "obs.csv")
            pred = self.csv_to_array(op_root / "pred.csv")
            obs = np.expand_dims(obs, axis=-1)
            pred = np.expand_dims(pred, axis=-1)
            _, metric_values = self.calc_metric(obs, pred)
            result_path = self.analyse_root / f"calc_{saved_metric_name}.csv"

            if file_flag is False:
                file_flag = True
                metric_comp = [f"{saved_metric_name}{i}" for i in range(len(metric_values))]
                with open(result_path, "wt") as f:
                    f.write(f"basin,{','.join(str(i) for i in metric_comp)}\n")
            with open(result_path, "at") as f:
                f.write(f"{basin},{','.join(str(i) for i in list(metric_values))}\n")

        print(f"{saved_metric_name} completed.")


if __name__ == '__main__':
    analyse_root = AnalysisConfig.analyse_root
    s = time.time()
    tpe_alpha = 0.02
    mc_nse = MetricCalculator(analyse_root, "nse")
    mc_kge = MetricCalculator(analyse_root, "kge")
    mc_tpe = MetricCalculator(analyse_root, "tpe", tpe_alpha)
    mc_bias = MetricCalculator(analyse_root, "bias")
    mc_nrmse = MetricCalculator(analyse_root, "nrmse")
    mc_ae = MetricCalculator(analyse_root, "ae")
    mc_logNSE = MetricCalculator(analyse_root, "logNSE")

    mc_nse.calc_and_save("nse")
    mc_kge.calc_and_save("kge")
    mc_tpe.calc_and_save(f"tpe[{tpe_alpha}]")
    mc_bias.calc_and_save("bias")
    mc_nrmse.calc_and_save("nrmse")
    mc_ae.calc_and_save("ae")
    mc_logNSE.calc_and_save("logNSE")

    # mc_nse.calc_and_save("nse_r", rescaled=True)
    # mc_kge.calc_and_save("kge_r", rescaled=True)
    # mc_tpe.calc_and_save(f"tpe[{tpe_alpha}]_r", rescaled=True)
    # mc_bias.calc_and_save("bias_r", rescaled=True)
    # mc_rmse.calc_and_save("rmse_r", rescaled=True)

    e = time.time()
    print(f"used time:{e - s}s.")