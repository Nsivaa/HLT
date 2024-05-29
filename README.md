# HLT

The dataset directory contains the unpreprocessed datasets of GoEmotions and TwitterData and their cleaned versions (including operations of GoEmotions_data_preparation.ipynb and Twitter_data_preparation.ipynb).

Each model is studied in a separate notebook, with the exception of random forest, that is studied in the same notebook as the decision tree.
These notebooks save their best models in the checkpoint directory (that is provided as empty due to the size of the files).

The directory lib contains python scripts that are used in the notebooks. The scripts contain functions to perform the grid search, load and preprocess datasets, build pytorch models, evaluate models and plot results.

Under results directory, the predictions of llama3, word ranking scores and grid search data is stored. Running grid searches of notebooks will not fit any model since hyperparameters data is stored in this directory.

GoEmotions_data_exploration.ipynb and Twitter_data_exploration.ipynb are the notebooks that contain the data exploration of the datasets.

GoEmotions_emotion_words.ipynb and Twitter_emotion_words.ipynb are the notebooks that contain the emotion words rankings according to a pmi based score.

model_comparison.ipynb is the notebook that contains the comparison of the final best models on the test set.

To build an environment with all dependencies installed, requirements.txt and requirements_conda.txt are provided. The first one is for pip and the second one is for conda and they are equivalent.

HLT_Project_Report.pdf contains the report of the analysis.