import numpy as np
import pandas as pd

from tools.logistics import *

def find_most_frequently_used_trigger(path_root, metadata):
    dict_freq = {}
    for _, row in metadata.iterrows():
        if row['ground_truth']:
            model_name = row['model_name']
            lookup_dir = os.path.join(path_root, model_name, 'triggers')
            if os.path.isdir(lookup_dir):
                triggers = os.listdir(lookup_dir)
                for trig in triggers:
                    if trig not in dict_freq:
                        dict_freq[trig] = [model_name]
                    else:
                        dict_freq[trig].append(model_name)
    best_size = 0
    best_trig = None
    for trig in dict_freq:
        size = len(dict_freq[trig])
        if size > best_size:
            best_size = size
            best_trig = trig
    best_trigger_path = os.path.join(path_root, dict_freq[best_trig][0], best_trig)
    print(dict_freq)
    print(best_trigger_path)
    return best_trigger_path


def main():
    np.random.seed(666)
    project_root_path = get_project_root_path()
    path_root = os.path.join(project_root_path, 'TrojAI-data', 'round1-holdout-dataset')
    path_metadata = os.path.join(path_root, 'METADATA.csv')

    metadata = pd.read_csv(path_metadata)

    find_most_frequently_used_trigger(path_root, metadata)
    exit(666)

    available_models = ['densenet121']

    for _, row in metadata.iterrows():
        model_name = row['model_name']
        num_classes = row['number_classes']
        ground_truth = row['ground_truth']

        if model_name in available_models:
            if ground_truth: # model is backdoored
                pass


if __name__ == '__main__':
    main()
    print('script ended')
