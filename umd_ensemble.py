from notebooks.methods import keras_load, load_obj


class UMDModelScaler(object):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler


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

    DEFAULT_PATH = os.path.join('metamodels', 'metamodels_ensemble')

    def __init__(self):
        self.models_scalers = {}
        self._load_models()

    def _load_models(self):
        models_count = 0
        for out_type in ['binary', 'bernoulli']:
            for in_count in [1, 2]:
                base_folder = f'{out_type}_{in_count}_input'
                path = os.path.join(UMDEnsemble.DEFAULT_PATH, base_folder)
                for model_folder in os.listdir(path):
                    full_model_path = os.path.join(path, model_folder)
                    model = keras_load(full_model_path)
                    scaler = load_obj(os.path.join(path_meta_model, 'scaler.pkl'))
                    self.models_scalers[base_folder].append((model, scaler))
                    models_count += 1
        print(f'[INFO] UMDEnsemble has {models_count} models')

    def predict(self):
        pass
