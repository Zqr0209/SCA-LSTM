import pandas as pd
from pathlib import Path


class DatasetConfig:
    camels_root = Path("/data2/zqr/month-CAMELS/CAMELS-AUS")  # your CAMELS dataset root TODO
    # camels_root = Path("/data2/zqr/TangNaiHai")

    # huc = "4huc"  # NOTE:4huc[01,03,11,17]
    # basins_test_mark = "01"  # ADD
    # basin_mark = "t" + basins_test_mark  # NOTE:
    # # basin_mark = basins_test_mark  # NOTE:

    # basins_file = f"data/{huc}/{basin_mark}.txt"  # TEST!!!
    # global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
    # pub_test_basins_list = pd.read_csv(f"data/{huc}/{basins_test_mark}.txt", header=None, dtype=str)[
    #     0].values.tolist()  # ADD

    # basin_mark = "624_BR_basins_list"  # NOTE:upper
    basin_mark = "114_AUS_basins_list"
    # basins_test_mark = "17"  # NOTE:upper

    basins_file = f"/data1/zqr/RRS-Former/data/{basin_mark}.txt"  # TEST!!!
    global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
    # pub_test_basins_list = pd.read_csv(f"data/{huc}/{basins_test_mark}.txt", header=None, dtype=str)[
    #     0].values.tolist()  # ADD

    '''''''''
    # 624_BR 
    train_start = pd.to_datetime("1985-12-16", format="%Y-%m-%d")
    train_end = pd.to_datetime("2000-12-31", format="%Y-%m-%d")
    # val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    # val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    # test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    # test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")
    val_start = pd.to_datetime("2001-01-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2005-12-31", format="%Y-%m-%d")
    test_start = pd.to_datetime("2006-01-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2010-09-29", format="%Y-%m-%d")
    '''''''''

    # day, week AUS 114 TODO
    '''''''''
    train_start = pd.to_datetime("1977-07-15", format="%Y-%m-%d")
    train_end = pd.to_datetime("2008-04-24", format="%Y-%m-%d")
    val_start = pd.to_datetime("2008-04-25", format="%Y-%m-%d")
    val_end = pd.to_datetime("2011-04-24", format="%Y-%m-%d")
    test_start = pd.to_datetime("2011-04-25", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-04-24", format="%Y-%m-%d")
    '''''''''






    # AUS 114 月 半月

    train_start = pd.to_datetime("1977-08-1", format="%Y-%m-%d")
    train_end = pd.to_datetime("2006-12-1", format="%Y-%m-%d")
    val_start = pd.to_datetime("2007-01-1", format="%Y-%m-%d")
    val_end = pd.to_datetime("2009-12-1", format="%Y-%m-%d")
    test_start = pd.to_datetime("2010-01-1", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-04-1", format="%Y-%m-%d")






    # AUS 114 旬
    '''''''''
    train_start = pd.to_datetime("1977-07-21", format="%Y-%m-%d")
    train_end = pd.to_datetime("2008-04-21", format="%Y-%m-%d")
    val_start = pd.to_datetime("2008-05-1", format="%Y-%m-%d")
    val_end = pd.to_datetime("2010-12-21", format="%Y-%m-%d")
    test_start = pd.to_datetime("2011-01-1", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-04-21", format="%Y-%m-%d")
    '''''''''









    # US daymet
    '''''''''
    train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")
    '''''''''

    # 唐乃亥
    '''''''''
    train_start = pd.to_datetime("1971-01-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("2002-12-01", format="%Y-%m-%d")
    val_start = pd.to_datetime("2003-01-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2007-12-01", format="%Y-%m-%d")
    test_start = pd.to_datetime("2008-01-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2012-12-01", format="%Y-%m-%d")
    '''''''''


    dataset_info = f"{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
    # dataset_info = f"{forcing_type}_{huc}_{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
