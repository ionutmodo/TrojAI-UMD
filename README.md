# TrojAI repository for the ICSI-UMD team

In order to create a Singularity container to submit on the TrojAI leaderboard,
you need to follow the following three steps:

1. Train an SDN on top of all provided models
2. Generate the features from all provided models
3. Train a binary classifier using the features computed above
4. Create the Singularity container that uses the binary classifier in order to perform a submission

Below we go through each step and provide details such that a solution gets replicated as easy as possible.

## 1. Train an SDN on top of all provided models
For this step we use the file `train_trojai_sdn.py` to train an SDN for each given model in the training set. Make sure you correctly set the variable `root_path` the root of the training dataset folder (it contains `id-0000xyzt` folders).

Use method `train_trojai_sdn` to train SDNs according to the Can's paper or train_trojai_sdn_with_svm to replace them with one-vs-all SVMs.

We use some synthetic data to train the SDNs, to compute the features as well as to predict when we submit the code to the TrojAI server. Each synthetic image is labeled using the original CNN that we want to convert to SDN. If you want to see how we generate this data, you should check out the file `synthetic_data/meta_model_trainer.py`. 

When the `train_trojai_sdn.py` finishes running, in each folder of type `id-0000xyzt` you should have a folder called `ics_synthetic-1000_train100_test0_bs20` that contains the SDN model.

## 2. Generate the features from all provided models
After training the SDN models using the synthetic data, we need to use those SDNs to compute the confusion distribution features using the same synthetic data that we used for training.  For this step we use the file `round_features_sdn.py`.

The following list provides some guidance on the most important variables that you need to set to train all SDNs:
- set batch size accordingly to your hardware (128 worked well for our experiments)
- set variables `experiment_name` and `sdn_name` such that they are significant enough for your experiment and SDN model; they will be used to name folders on the disk
- set the project path accordingly in the method `tools/logistics.py` for each machine you run the code on: we perform the training on OpenLab and we also provide a path for the Windows machine to quickly test the functionality of the  code before starting the job on the OpenLab machine; in that method there might also be more paths for different machines where we tested over time
- the variable `path_report_conf_dist` holds the path to the `df_report_conf_dist` dataframe object. Each line in this dataframe contains the confusion distribution parameters computed on each model. After appending one line, the dataframe is immediately dumped on the disk. Since our OpenLab server has a maximum running time of 8 hours for the processes that use GPU, we implement a mechanism that continues training the SDN from where it left off after the 8h time window expired. For example, if those 8 hours pass and the OpenLab job ends while computing features for model #328, then running the script again without any changes will start by reading the dataframe into memory and continue experiment with the model #328 (the last line would contains features of model #327). 
- make sure you have the metadata file at the specified path
- we iterate through all models, compute confusion distribution and then compute `mean` and `std` of the confusion distribution. When running on OpenLab or Windows 10, we compute the confusion distribution for each backdoored dataset sequentially, but we can also compute for all of them in parallel using Python's multiprocessing module.

When the `round_features_sdn.py` finishes running, the csv file at path `path_report_conf_dist` would have the confusion distribution features computed for all models, using the synthetic data.

## 3. Train a binary classifier using the features computed above
Now that we have the features from all models and now we need to train a meta-model that would use the confusion distribution's mean and std and would try to predict whether the model is backdoored or not. For that, we will train a binary classifier (meta-model) using the `notebooks/analyze-round.ipynb`. We trained the following meta-models:
- fully-connected neural network with 1 and 2 input branches
- logistic regression
- random forest

The meta-models should be placed in the folder `metamodels` and should saved in keras format (two files with extensions h5 and json) or in pickled format if they are models in sklearn.

## 4. Create the Singularity container that uses the binary classifier in order to perform a submission
After training the meta model, it's time to create a container that uses it to perform predictions on the new unseen models on the TrojAI server.

During inference time we need to do all steps above for a single model that we need to classify as clean or backdoored. The steps our container will follow are implemented in the `umd_pipeline.py` file. Note that this file will be run when the container is called.

In order to build the Singularity container to be uploaded, we use the sandbox functionality described in the file `build_spec_sandbox.def` and also contains all required packages to be installed inside the container.

Below we describe some variables that we need to set accordingly before creating the Singularity container:
- `fast_local_test`: set it to TRUE when running tests locally on your machine to check if your meta-model works when you plug it in into the pipeline. **You _MUST_ set it to FALSE when building the container.** Note that setting it to TRUE would actually skip the most time-consuming part: computing training the SDN and building the confusion distribution. Instead of using some real confusion distribution features, it will use zero values.
- `arch_wise_metamodel`: set it to TRUE if you have one meta-model trained on data from a specific model architecture, otherwise set it to FALSE.
- `use_abs_features`: our former experiments used the absolute value of the confusion distribution features. However, we discovered that using absolute value is not beneficial. We did not remove this functionality, but keep it disabled by always setting this parameter to False.
- `add_arch_features`: set it to TRUE if the meta-model uses a specific feature encoding the architecture of the input model. We found this feature decreases the CrossEntropy loss by up to 0.05. Note that this won't work if the input models have some custom architectures.
- `scenario_number`: during our tests we experimented with fully connected based SDNs and SVM based SDNs, as well as simply computing some features on top of the original CNN that we are given. This parameter specifies which scenario (network type, statistics type) we want to use. We suggesst using scenario 1.
- `path_meta_model_binary`: we experimented with fully connected based meta-models that have two types of outputs:
    - **binary output** modelling the probability that the input-model is backdoored, basically P(backdoored|confusion distribution features)$
    - **bernoulli output (multi-label classification)** modelling the independent probability that the model is clean or backdoored with different kinds of backdoors (polygon and all instagram filters). The output of this kind of meta-model is more flexible and allows us develop different heuristics to finally compute the probability that the input model is backdoored
- `sdn_name`: the name of the SDN folder present in each model folder with pattern `id-0000xyzt`

The pipeline contains the following steps:
1. Given the input model, train an SDN using the SAME synthetic dataset that we used in the previous steps.
2. Load the SDN and compute the stats (features of the confusion distribution)
3. Using the stats computed previously, perform the prediction using the meta-model.
4. Write the prediction to the result file
5. Check TrojAI leaderboard for running status