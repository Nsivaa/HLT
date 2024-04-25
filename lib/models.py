
import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from torch import cuda
from lib.dataset_utils import *
from lib.plot_utils import *
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import matplotlib.pyplot as plt


'''
pytorch Roberta model class
'''
#TODO cross validation su topologia
class RobertaClass(torch.nn.Module):
    def __init__(self, n_classes, frozen_layers=-1):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        if frozen_layers != -1:
            for param in self.l1.embeddings.parameters():
                param.requires_grad = False
            for i in range(frozen_layers):
                for param in self.l1.encoder.layer[i].parameters():
                    param.requires_grad = False
        self.pre_classifier = torch.nn.Linear(768, 768)#TODO add_module?
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]#TODO ???
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
    def get_out_dim(self):
        return self.classifier.out_features
    
'''
utility function to create parameters for the model
'''
def create_model_params(optimizer=torch.optim.Adam,
                        tokenizer=RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True),
                        tokenizer_max_len=None,
                        batch_size=8,
                        loss_function=torch.nn.BCEWithLogitsLoss(),
                        epochs=1,
                        learning_rate=1e-05,
                        val_patience=1,
                        clip_grad_norm=-1):
    return {
        'optimizer': optimizer,
        'tokenizer': tokenizer,
        'tokenizer_max_len': tokenizer_max_len,
        'batch_size': batch_size,
        'loss_function': loss_function,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'val_patience': val_patience,
        'clip_grad_norm': clip_grad_norm
    }

'''
interface of pytorch models useful for cross validation and to automate model construction and evaluation
'''
class SimpleModelInterface:
    def __init__(self, 
                 model: torch.nn.Module, 
                 scores={},
                 model_params_dict=create_model_params()):
        self.model = model
        self.params = model_params_dict
        self.optimizer = model_params_dict['optimizer'](params=model.parameters(), lr=model_params_dict['learning_rate'])
        self.scores = scores
        self.train_scores = {name: [] for name in scores.keys()}
        self.train_loss = []
        self.val_scores = {name: [] for name in scores.keys()}
        self.val_loss = []
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        model.to(self.device)

    def _train(self, training_loader, validation_loader=None, save_path=None):
        self.model.train()
        #TODO usare confusion matrix?
        cur_patience = self.params['val_patience']
        best_val_loss = np.inf
        for _ in range(self.params['epochs']):
            tr_loss = 0
            predictions_acc = []
            targets_acc = []
            for _,data in tqdm(enumerate(training_loader, 0)):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)

                outputs = self.model(ids, mask, token_type_ids)
                loss = self.params['loss_function'](outputs, targets)
                tr_loss += loss.item()
                # append predictions and targets
                predictions_acc.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                targets_acc.extend(targets.detach().cpu().numpy())

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                if self.params['clip_grad_norm'] > -0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip_grad_norm'])
                self.optimizer.step()

            # create numpy arrays
            predictions_acc = np.array(predictions_acc)
            targets_acc = np.array(targets_acc)
            # calculate training scores
            epoch_loss = tr_loss/len(training_loader)
            self.train_loss.append(epoch_loss)
            for name, score in self.scores.items():
                self.train_scores[name].append(score(targets_acc, predictions_acc))

            # calculate validation scores
            if validation_loader is not None:
                val_scores = self.evaluate(validation_loader)
                for name, score in val_scores.items():
                    self.val_scores[name].append(score)
                self.val_loss.append(val_scores['loss'])
                if val_scores['loss'] < best_val_loss:
                    best_val_loss = val_scores['loss']
                    cur_patience = self.params['val_patience']
                    # save model
                    if save_path is not None:
                        torch.save(self.model, save_path)
                else:
                    cur_patience -= 1
                if cur_patience == 0:
                    break

        # restore best model
        if save_path is not None and validation_loader is not None:
            self.model = torch.load(save_path)
            self.model.to(self.device)

    def fit(self, training_df, validation_df=None):
        training_loader = create_data_loader_from_dataframe(training_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=True)
        validation_loader = None
        if validation_df is not None:
            validation_loader = create_data_loader_from_dataframe(validation_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        self._train(training_loader, validation_loader)

    def _predict(self, data_loader, accumulate_targets=False):
        self.model.eval()
        # initialize target and prediction matrices
        predictions_acc = []
        targets_acc = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader, 0)):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
                outputs = self.model(ids, mask, token_type_ids).squeeze()
                # append predictions and targets
                predictions_acc.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                if accumulate_targets:
                    targets_acc.extend(targets.detach().cpu().numpy())
        targets_acc = np.array(targets_acc)
        predictions_acc = np.array(predictions_acc)
        return predictions_acc, targets_acc

    def predict(self, testing_df):
        testing_loader = create_data_loader_from_dataframe(testing_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        predictions, _ =  self._predict(testing_loader)
        return predictions

    def evaluate(self, testing_df, custom_scores=None):
        testing_loader = create_data_loader_from_dataframe(testing_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        predictions, targets = self._predict(testing_loader, accumulate_targets=True)
        # calculate scores
        scores_to_compute = self.scores if custom_scores is None else custom_scores
        scores = {name: score(targets, predictions) for name, score in scores_to_compute.items()}
        return scores

    def get_train_scores(self):
        return self.train_scores
    
    def get_train_loss(self):
        return self.train_loss

    def get_val_scores(self):
        return self.val_scores
    
    def get_val_loss(self):
        return self.val_loss
    
    def save_model(self, model_path, vocabulary_path):
        torch.save(self.model, model_path)
        self.tokenizer.save_vocabulary(vocabulary_path)


'''
function to perform statistical testing to compare 2 models using bootstrap
'''
def bootstrap_test(model1, model2, testing_df, n_tests, sample_size, metric_fun, metric_name):
    # initial evaluation
    score1 = model1.evaluate(testing_df, custom_scores={metric_name: metric_fun})[metric_name]
    score2 = model2.evaluate(testing_df, custom_scores={metric_name: metric_fun})[metric_name]
    print(f'Initial {metric_name}: {score1} {score2}')
    best_model = 1 if score1 > score2 else 2
    if score1 < score2:
        model1, model2 = model2, model1
        score1, score2 = score2, score1
    delta = score1 - score2
    print(f'Best model: {best_model}, with delta: {delta}')
    # perform bootstrap
    successes = 0
    for _ in range(n_tests):
        sample = np.random.choice(testing_df.index, sample_size)
        score1 = model1.evaluate(testing_df.loc[sample], custom_scores={metric_name: metric_fun})[metric_name]
        score2 = model2.evaluate(testing_df.loc[sample], custom_scores={metric_name: metric_fun})[metric_name]
        cur_delta = score1 - score2
        if cur_delta >= 2*delta:
            successes += 1
    print(f'Successes: {successes}/{n_tests}')
    print(f'p-value: {successes/n_tests}')
    return successes/n_tests