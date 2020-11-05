import shutil
import pandas as pd
from tools.logistics import *
import multiprocessing as mp


def worker_delete_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f'deleted {path}')


def main():
    folder_names_to_delete = [
        # 'ics_train100_test0_bs25',
        # 'backdoored_data_custom-square-size-5_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-10_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-15_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-20_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-25_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-30_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-35_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-40_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-45_backd-original-color_clean-black-color',
        # 'backdoored_data_custom-square-size-50_backd-original-color_clean-black-color',
        'backdoored_data_square-25',
        'backdoored_data_filter_gotham',
        'backdoored_data_filter_kelvin',
        'backdoored_data_filter_lomo',
        'backdoored_data_filter_nashville',
        'backdoored_data_filter_toaster'
    ]

    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-dataset-train')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset')
    root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-train-dataset')
    # root_path = os.path.join(get_project_root_path(), 'TrojAI-data', 'round2-holdout-dataset')

    metadata_path = os.path.join(root_path, 'METADATA.csv')
    metadata = pd.read_csv(metadata_path)

    mp_folders = []
    for index, row in metadata.iterrows():
        model_name = row['model_name']

        # model_architecture = row['model_architecture']
        # model_id_int = int(model_name[3:])

        mp_folders.append(os.path.join(root_path, model_name)) # use this to delete all folders id-0000ABCD

        # for f_name in folder_names_to_delete:
        #     folder = os.path.join(root_path, model_name, f_name)
        #     mp_folders.append(folder)

    cpus = mp.cpu_count() - 4
    print(f'Deleting {len(mp_folders)} datasets using {cpus} CPU cores')
    with mp.Pool(processes=cpus) as pool:
        pool.map(worker_delete_folder, mp_folders)


if __name__ == '__main__':
    main()
    print('script ended')
