import torch
import pandas as pd
import os


def read_attention(basin):
    dir = f"/data1/zqr/RRS-Former/runs/normal/8_LSTMATTN_None_[[8-1-8-128][*-1-8-128]-0.2]@114_AUS_basins_list_1977~2008#2008~2011#2011~2014@8|7+1[8+1]@NSELoss_n200_bs2048_lr0.001_warm_up@seed2333/{basin}/attention.pth"
    attention = torch.load(dir)
    attn_map = torch.mean(attention, dim=0)
    attn_map = attn_map.squeeze().tolist()
    return attn_map

basins_file = f"/data1/zqr/RRS-Former/data/114_AUS_basins_list.txt"
basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
attn = []
for basin in basins_list:
    basin_attn = read_attention(basin)
    basin_attn.append(basin)
    attn.append(basin_attn)

df = pd.DataFrame(attn)
df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', 'basin']
order = ['basin', '1', '2', '3', '4', '5', '6', '7', '8']
df = df[order]

'''''''''
saving_root = "/data1/zqr/RRS-Former/figure/attention/basin_attn.csv"
os.makedirs(os.path.dirname(saving_root), exist_ok=True)
df.to_csv(saving_root, sep=',', index=False, header=True)
'''''''''

best_basins_root = "/data1/zqr/RR-Former/figure/nse.csv"
best_basins = pd.read_csv(best_basins_root)
best_basins_list = best_basins.basin.tolist()
best_basins_attn = df[df['basin'].isin(best_basins_list)]

saving_root = "/data1/zqr/RRS-Former/figure/attention/best_basins_attn.csv"
os.makedirs(os.path.dirname(saving_root), exist_ok=True)
best_basins_attn.to_csv(saving_root, sep=',', index=False, header=True)
