import os

from logistics import get_project_root_path
from tools.data import create_backdoored_dataset

def main():
    dir_root = os.path.join(get_project_root_path(), 'TrojAI-data', 'round1-holdout-dataset', 'id-00000009')

    # dir_root = '/mnt/storage/Cloud/MEGA/TrojAI/TrojAI-data/round1-holdout-dataset/id-00000009/'
    dir_example_data = os.path.join(dir_root, 'example_data')
    dir_backdoored_data = os.path.join(dir_root, 'example_data_backdoored')
    filename_trigger = os.path.join(dir_root, 'triggers', 'trigger_9_11.png')

    create_backdoored_dataset(dir_clean_data=dir_example_data,
                              dir_backdoored_data=dir_backdoored_data,
                              filename_trigger=filename_trigger,
                              triggered_fraction=1.0,
                              triggered_classes='all',
                              trigger_target_class=0,
                              trigger_color=(180, 184, 114),
                              trigger_size=31,
                              p_trigger=1.0,
                              keep_original=False)


if __name__ == '__main__':
    main()
    print('script ended')
