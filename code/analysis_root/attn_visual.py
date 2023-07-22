import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8]
basin_attn_root = "/data1/zqr/RRS-Former/figure/attention/basin_attn.csv"
basin_attn = pd.read_csv(basin_attn_root)
data = basin_attn.drop('basin', axis=1)
data_array = np.array(data)
data_list = data_array.tolist()
for i in data_list:
    plt.scatter(x, i, color='lightgreen')

attn_value_root = '/data1/zqr/RRS-Former/figure/attention/attn_value.xlsx'
attn_value = pd.read_excel(attn_value_root)
attn_value = attn_value.iloc[:, 1:]

mean = np.array(attn_value.iloc[0])
mean = mean.tolist()
plt.scatter(x, mean, color='deepskyblue',  marker='s', label='mean')

median = np.array(attn_value.iloc[1])
median = median.tolist()
plt.scatter(x, median, color='yellow',  marker='*', label='median')

plt.legend()

plt.xlabel('LSTM Cell')
plt.ylabel('Weights')
plt.title("SCA-LSTM")

plt.savefig('/data1/zqr/RRS-Former/figure/attention/lstm_attn.eps')

plt.show()