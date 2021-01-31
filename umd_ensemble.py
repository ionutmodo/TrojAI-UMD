from notebooks.methods import keras_load, load_obj
import numpy as np
import os


def general_predict(model, scaler, features: list, arch: list, output_type: str, input_count: int):
    def get_features(f: np.ndarray, a: np.ndarray):
        if input_count == 1:
            return np.hstack((f, a)) # np.array(f.tolist() + a.tolist()).reshape(1, -1)
        if input_count == 2:
            return [f.reshape(1, -1), a.reshape(1, -1)]
        raise RuntimeError('Invalid input_count')

    def get_prediction(f):
        if output_type == 'binary':
            proba = model.predict(f)[0][0]
            label = 1 if proba > UMDEnsemble.BINARY_THRESHOLD else 0
            return label, proba
        if output_type == 'bernoulli':
            prediction = model.predict(f)[0]
            pair_label_prediction = sorted(enumerate(prediction), key=lambda x: -x[1])
            label, proba = pair_label_prediction[0]
            if label == 0:  # clean has max probability => predict 0, 1 - proba
                return 0, 1.0 - proba
            return 1, proba # a backdoored class has max probability => predict 1, proba
        raise RuntimeError('Invalid output_type')

    features = np.array(features).reshape(1, -1)
    arch = np.array(arch).reshape(1, -1)

    if scaler is not None:
        features = scaler.transform(features)
    features = get_features(features, arch)
    backd_proba = get_prediction(features)
    return backd_proba


class UMDEnsemble(object):
    """
        This class implements an ensemble model using many NeuralNetworks saved during testing.
        They should be saved in the path "metamodels/metamodels_ensemble" and its structure should be:
        metamodels/metamodels_ensemble:
        |--- binary_1_input
        |--- binary_2_input
        |--- bernoulli_1_input
        |--- bernoulli_2_input

        Each folder above has the following structure:
        |--- model #1
            |--- model.pkl
            \--- scaler.pkl (optional)
    """

    DEFAULT_PATH = os.path.join('metamodels', 'ensemble_T=0.5')

    """
        The binary threshold is used to decide if the label is 0 or 1 in the case of binary models.
        Basically, if the probability p <= BINARY_THRESHOLD, then the label would be 0, otherwise 1.
        This is useful when I add the probability to specific class bucket (clean, backdoored). 
    """
    BINARY_THRESHOLD = 0.5

    def __init__(self):
        self.models_scalers = {}
        self._load_models()

    def _load_models(self):
        models_count = 0
        for out_type in ['binary', 'bernoulli']:
            for in_count in [1, 2]:
                base_folder = f'{out_type}_{in_count}'
                path = os.path.join(UMDEnsemble.DEFAULT_PATH, base_folder)

                key = (out_type, in_count)
                if key not in self.models_scalers:
                    self.models_scalers[key] = []

                for model_folder in os.listdir(path):
                    full_model_path = os.path.join(path, model_folder)

                    model = keras_load(full_model_path)
                    scaler = load_obj(os.path.join(full_model_path, 'scaler.pkl'))

                    self.models_scalers[key].append((model, scaler))
                    models_count += 1
        print(f'[INFO] UMDEnsemble has {models_count} models')

    def predict(self, features: list, arch: list):
        predictions = {0: [], 1: []}
        for out_type in ['binary', 'bernoulli']:
            for in_count in [1, 2]:
                key = (out_type, in_count)
                for model, scaler in self.models_scalers[key]:
                    label, proba = general_predict(model, scaler, features, arch, out_type, in_count)
                    predictions[label].append(proba)

        print(f'[INFO] bucket clean: {predictions[0]}')
        print(f'[INFO] bucket backd: {predictions[1]}')

        count_0 = len(predictions[0])
        count_1 = len(predictions[1])
        mean_proba = 0.5
        if count_0 > count_1: # voting would predict 'clean'
            mean_proba = sum(predictions[0]) / count_0 # average over clean probabilities
        else: # voting would predict 'backdoored'
            mean_proba = sum(predictions[1]) / count_1 # aberage over backdoored probabilities
        return mean_proba
