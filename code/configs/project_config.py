import torch
from pathlib import Path
from configs.dataset_config import DatasetConfig


# Project root, computing resources
class ProjectConfig:
    # project_root = Path(__file__).absolute().parent.parent
    project_root = Path("/data1/zqr/RRS-Former")
    single_gpu = 0 # TODO: which gpu to run
    device = torch.device(f"cuda:{single_gpu}")
    # device = torch.device(f"cpu")
    torch.cuda.set_device(device)
    num_workers = 0  # Number of threads for loading data

    # save_dir = f"runoff"
    save_dir = f"normal"
    # save_dir = f"{DatasetConfig.huc}"
    # save_dir = f"{DatasetConfig.huc}" + "_test_search"
    run_root = Path(f"/data1/zqr/RRS-Former/runs/{save_dir}")  # Save each run CHANGE:pub修改了 ORIGIN:Path(f"./runs")

    final_data_root = Path("/data1/zqr/RRS-Former/final_data")  # Cache preprocessed data NOTE:ORIGIN
    # final_data_root = Path("./final_data" + "_test_rrs_norunoff_new") #NOTE：还有一个要测试
