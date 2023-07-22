import pandas as pd
import numpy as np


def cacl_ae(obs, sim):

    ae = np.abs(obs - sim) / obs
    return ae

obs_path = '/data1/zqr/RRS-Former/runs/normal/TransformerS_NAR_[64-4-4-256-0.1]@1_TNH_basins_list_1971~2002#2003~2007#2008~2012@8|7+1[2+1]@MSE_n200_bs4_lr0.0001_warm_up@seed2333/1/obs_pred/obs.csv'
pred_path = '/data1/zqr/RRS-Former/runs/normal/TransformerS_NAR_[64-4-4-256-0.1]@1_TNH_basins_list_1971~2002#2003~2007#2008~2012@8|7+1[2+1]@MSE_n200_bs4_lr0.0001_warm_up@seed2333/1/obs_pred/pred.csv'

obs = pd.read_csv(obs_path, header=0)
pred = pd.read_csv(pred_path, header=0)
obs_0 = obs['obs0']
pred_0 = pred['pred0']
start_date = obs['start_date']
ae = cacl_ae(obs_0, pred_0)
data = pd.concat([start_date, obs_0, pred_0, ae], axis=1)
data.columns = ['start_date', 'obs_0', 'pred_0', 'ae']
outputpath = '/data1/zqr/RRS-Former/runs/normal/TransformerS_NAR_[64-4-4-256-0.1]@1_TNH_basins_list_1971~2002#2003~2007#2008~2012@8|7+1[2+1]@MSE_n200_bs4_lr0.0001_warm_up@seed2333/ae.csv'
data.to_csv(outputpath, sep=',', index=False, header=True)


