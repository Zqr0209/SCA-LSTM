import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

attn_root = Path('/data1/zqr/RRS-Former/figure/attention/A2390523.pth')
attn_map = torch.load(attn_root)
attn_map = torch.mean(attn_map, dim=0)
print(attn_map)
print(attn_map.shape)

plt.imshow(attn_map, cmap='hot')
plt.xticks(np.arange(0, 8, 1), np.arange(1, 9, 1))
plt.yticks(np.arange(1))
plt.yticks(np.arange(1), np.arange(1, 2, 1))
plt.colorbar()
plt.title('Basin: A2390523', x=0.5, y=4)
plt.savefig('/data1/zqr/RRS-Former/figure/attention/attention.png')