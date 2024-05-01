
import os
import itertools
import pandas as pd
from tqdm.notebook import tqdm
from lib.models import SimpleModelInterface

'''
    Performs hold-out cross validation
    Every <checkpoint_interval> models evaluated the results are stored in <res_file>.
    <param_dict> contains for each key a list of values and represents the parameters to do the grid search on.
    Cross validation may be performed incrementally:
    new parameter values may be added in subsequent runs, in this case only the missing data will be computed,
    also new parameters may be added, but a default value for them must be provided for consistency with old data.
    Deletion of values or parameters from <param_dict> will not delete the old data contained in the file.
    <default_params_value> is an optional dictionary used in case new parameters are added or deleted wrt the old result file
    <scores> is a dictionary containing the scores to evaluate the model on, it is assumed that this parameter doesn't change over subsequent runs
'''
class HoldOutCrossValidation:
    def __init__(self, model_class : SimpleModelInterface, scores, train_df, val_df, param_dict, res_file=None, checkpoint_interval=1, default_params_value={}):
        # check if the model class is a subclass of SimpleModelInterface
        assert issubclass(model_class, SimpleModelInterface), 'model_class must be a subclass of SimpleModelInterface'
        self.model_class = model_class
        self.scores = scores
        self.train_df = train_df
        self.val_df = val_df
        self.res_file = res_file
        self.checkpoint_interval = checkpoint_interval
        self.param_dict = param_dict
        self.results = None
        self._init_results(default_params_value)

    def _init_results(self, default_params_value):
        if self.res_file is not None and os.path.exists(self.res_file):
            self.results = pd.read_csv(self.res_file)
            saved_params = list(self.results.columns)
            # remove scores from saved_params
            for score in self.scores.keys():
                saved_params.remove('train_' + score)
                saved_params.remove('val_' + score)
            # check if new params are present in the param_dict
            for param in self.param_dict:
                if param not in saved_params:
                    # previous results should have been done with the default value
                    self.results[param] = default_params_value[param]
            # check for deletion of params
            for param in saved_params:
                if param not in self.param_dict:
                    # add the default value to param_dict
                    self.param_dict[param] = [default_params_value[param]]
        else:
            self.results = pd.DataFrame(columns=list(sorted(self.param_dict.keys())) + ['train_' + score for score in self.scores.keys()] + ['val_' + score for score in self.scores.keys()])

    def _get_param_combinations(self):
        return [dict(zip(self.param_dict.keys(), values)) for values in itertools.product(*self.param_dict.values())]

    def run(self, progress_bar=True):
        ckpt = self.checkpoint_interval
        if progress_bar:
            # compute the total number of iterations
            total_iterations = 0
            for params in self._get_param_combinations():
                if self.results.loc[(self.results[list(params)] == pd.Series(params)).all(axis=1)].shape[0] == 0:
                    total_iterations += 1
            pbar = tqdm(total=total_iterations)
        for params in self._get_param_combinations():
            if self.results.loc[(self.results[list(params)] == pd.Series(params)).all(axis=1)].shape[0] > 0:
                continue
            model = self.model_class(model_params_dict=params)
            model.fit(self.train_df)
            val_scores = model.evaluate(self.val_df, scores=self.scores)
            val_scores = {f'val_{k}': v for k, v in val_scores.items()}
            train_scores = model.evaluate(self.train_df, scores=self.scores)
            train_scores = {f'train_{k}': v for k, v in train_scores.items()}
            self.results = pd.concat([self.results, pd.DataFrame([{**params, **train_scores, **val_scores}])], ignore_index=True)
            ckpt -= 1
            if self.res_file is not None and ckpt == 0:
                # create the directory if it doesn't exist
                if not os.path.exists(os.path.dirname(self.res_file)):
                    os.makedirs(os.path.dirname(self.res_file))
                # delete old backup and backup new csv
                if os.path.exists(self.res_file + '.bak'):
                    os.remove(self.res_file + '.bak')
                if os.path.exists(self.res_file):
                    os.rename(self.res_file, self.res_file + '.bak')
                # save results
                self.results.to_csv(self.res_file, index=False)
                ckpt = self.checkpoint_interval
            if progress_bar:
                pbar.update(1)
        
        if progress_bar:
            pbar.close()
        if self.res_file is not None:
            self.results.to_csv(self.res_file, index=False)
    
    def get_best_info(self, score, is_maximization=True):
        return self.results.sort_values('val_' + score, ascending=not is_maximization).iloc[0].to_dict()
    
    def get_best_params(self, score, is_maximization=True):
        info = self.get_best_info(score, is_maximization)
        # remove scores from info
        for score in self.scores.keys():
            info.pop('train_' + score)
            info.pop('val_' + score)
        return info
    
    def get_results(self):
        return self.results