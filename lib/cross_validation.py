
import os
import itertools
import pandas as pd
from tqdm.notebook import tqdm

'''
    Performs hold-out cross validation
    Every <checkpoint_interval> models evaluated the results are stored in <res_file>.
    <param_dict> contains for each key a list of values and represents the parameters to do the grid search on.
    Cross validation may be performed incrementally:
    new parameter values may be added in subsequent runs, in this case only the missing data will be computed,
    also new parameters may be added, but a default value for them must be provided for consistency with old data.
    Deletion of values or parameters from <param_dict> will not delete the old data contained in the file.
    <default_params_value> is an optional dictionary used in case new parameters are added or deleted wrt the old result file
'''
class HoldOutCrossValidation:
    def __init__(self, model_class, scores, train_df, val_df, param_dict, res_file=None, checkpoint_interval=1, default_params_value={}):
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
        else:
            self.results = pd.DataFrame(columns=['params'] + self.scores)
            # check if new params are present in the param_dict
            for param in self.param_dict:
                if param not in self.results.columns:
                    # previous results should have been done with the default value
                    self.results[param] = default_params_value[param]
            # check for deletion of params
            for param in self.results.columns:
                if param not in self.param_dict:
                    # add the default value to param_dict
                    self.param_dict[param] = [default_params_value[param]]

    def _get_param_combinations(self):
        return [dict(zip(self.param_dict.keys(), values)) for values in itertools.product(*self.param_dict.values())]

    def run(self, progress_bar=True):
        ckpt = self.checkpoint_interval
        if progress_bar:
            # compute the total number of iterations
            total_iterations = 0
            for _ in self._get_param_combinations():
                if self.results.loc[(self.results[list(_)] == pd.Series(_)).all(axis=1)].shape[0] == 0:
                    total_iterations += 1
            pbar = tqdm.tqdm(total=total_iterations)
        for params in self._get_param_combinations():
            if self.results.loc[(self.results[list(params)] == pd.Series(params)).all(axis=1)].shape[0] > 0:
                continue
            model = self.model_class(**params)
            model.fit(self.train_df)
            scores = model.evaluate(self.val_df, scores=self.scores)
            self.results = self.results.append(pd.Series({**params, **scores}), ignore_index=True)
            ckpt -= 1
            if self.res_file is not None and ckpt == 0:
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
        return self.results.sort_values(score, ascending=not is_maximization).iloc[0].to_dict()
    
    def get_results(self):
        return self.results