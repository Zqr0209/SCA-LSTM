import os
import pandas as pd
from pathlib import Path

def statistics(name):
    root = Path('/data1/zqr/RRS-Former/figure/nse')
    data_root = root / f"{name}.csv"
    data = pd.read_csv(data_root)
    data = data.rename(columns={'nse0': name})
    return data

trs_day = statistics('trs_day')
sa_day = statistics('sa_day')

trs_week = statistics('trs_week')
sa_week = statistics('sa_week')

trs_tenday = statistics('trs_tenday')
sa_tenday = statistics('sa_tenday')

trs_halfmonth = statistics('trs_halfmonth')
sa_halfmonth = statistics('sa_halfmonth')

trs_month = statistics('trs_month')
sa_month = statistics('sa_month')

pd = pd.concat([trs_day, sa_day, trs_week, sa_week, trs_tenday, sa_tenday, trs_halfmonth, sa_halfmonth, trs_month, sa_month], axis=1)

pd = pd.T[~pd.T.index.duplicated()].T

saving_root = "/data1/zqr/RRS-Former/figure/nse/nse.csv"
os.makedirs(os.path.dirname(saving_root), exist_ok=True)
pd.to_csv(saving_root, sep=',', index=False, header=True)

print(pd)
