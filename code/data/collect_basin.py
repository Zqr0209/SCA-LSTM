import os

def collect(basin_number):
    file_dir = f"/data2/zqr/CAMELS/CAMELS-US/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{basin_number}"
    files = os.listdir(file_dir)
    basin_list = []
    for i in files:
        basin_list.append(i[0:8])

    sum = len(basin_list)
    save_file = f"/data1/zqr/RRS-Former/data/{sum}_US_basins_list.txt"

    if os.path.exists(save_file):
        print(f'{save_file} is already exist!')
    else:
        f = open(save_file, "w")
        for line in basin_list:
            f.write(line + '\n')
        f.close()
        print(f'{save_file} save successfully!')

collect("17")