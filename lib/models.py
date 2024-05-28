
import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer, AutoTokenizer, AutoModel
import outlines
from torch import cuda
from lib.dataset_utils import *
from lib.plot_utils import *
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os

'''
interface of pytorch models useful for cross validation and to automate model construction and evaluation
'''
class SimpleModelInterface(ABC):
    def __init__(self, 
                 scores={},
                 model_params_dict={},
                 checkpoint=None):
        self.params = self._create_model_params(model_params_dict)
        self.model = self._build_model()
        if checkpoint is not None and os.path.exists(checkpoint):
            self.model = torch.load(checkpoint)
        self.optimizer = self._build_optimizer()
        self.scores = scores
        self.train_scores = {name: [] for name in scores.keys()}
        self.train_loss = []
        self.val_scores = {name: [] for name in scores.keys()}
        self.val_loss = []
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def save_model(self, path):
        torch.save(self.model, path)

    def get_params(self):
        return self.params

    # abstract method to build the model parameters
    @abstractmethod
    def _create_model_params(self):
        pass

    # abstract method to build the torch module, useful to cross validate topology
    @abstractmethod
    def _build_model(self):
        pass

    # may be overridden to build a custom optimizer thus avoiding to rely on fixed parameters
    def _build_optimizer(self):
        return self.params['optimizer'](params=self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['regularization'])

    def _train(self, training_loader, validation_loader=None, save_path=None, progress_bar_epoch=True, progress_bar_step=True, checkpoint_score='loss', checkpoint_score_maximize=False):
        self.model.train()
        self.val_scores['loss'] = []
        cur_patience = self.params['val_patience']
        best_val_performance = -np.inf if checkpoint_score_maximize else np.inf
        for _ in tqdm(range(self.params['epochs']), disable=not progress_bar_epoch):
            if progress_bar_step:
                print(f'Epoch {_+1}/{self.params["epochs"]}')
            tr_loss = 0
            predictions_acc = []
            targets_acc = []
            for data in tqdm(training_loader, disable=not progress_bar_step):
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
                val_scores = self._evaluate(validation_loader, progress_bar=progress_bar_step, compute_loss=True)
                for name, score in val_scores.items():
                    self.val_scores[name].append(score)
                if (val_scores[checkpoint_score] > best_val_performance) == checkpoint_score_maximize:
                    best_val_performance = val_scores[checkpoint_score]
                    cur_patience = self.params['val_patience']
                    # save model
                    if save_path is not None:
                        dir_path = os.path.dirname(save_path)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        torch.save(self.model, save_path)
                else:
                    cur_patience -= 1
                if cur_patience == 0:
                    break

        if validation_loader is not None:
            # append validation loss to the correct list
            for el in self.val_scores['loss']:
                self.val_loss.append(el)
            # delete loss from val_scores
            del self.val_scores['loss']

        # restore best model
        if save_path is not None and validation_loader is not None:
            self.model = torch.load(save_path)
            self.model.to(self.device)

    def fit(self, training_df, validation_df=None, progress_bar_epoch=False, progress_bar_step=False, checkpoint_path=None, checkpoint_score='loss', checkpoint_score_maximize=False, shuffle_training=True):
        training_loader = create_data_loader_from_dataframe(training_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=shuffle_training)
        validation_loader = None
        if validation_df is not None:
            validation_loader = create_data_loader_from_dataframe(validation_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        self._train(training_loader, validation_loader, progress_bar_epoch=progress_bar_epoch, progress_bar_step=progress_bar_step, save_path=checkpoint_path, checkpoint_score=checkpoint_score, checkpoint_score_maximize=checkpoint_score_maximize)

    def _predict(self, data_loader, accumulate_targets=False, progress_bar=True, accumulate_loss=False):
        self.model.eval()
        # initialize target and prediction matrices
        predictions_acc = []
        targets_acc = []
        loss_acc = 0
        with torch.no_grad():
            for data in tqdm(data_loader, disable=not progress_bar):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
                outputs = self.model(ids, mask, token_type_ids).squeeze()
                # append predictions and targets
                predictions_acc.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                if accumulate_targets:
                    targets_acc.extend(targets.detach().cpu().numpy())
                if accumulate_loss:
                    loss_acc += self.params['loss_function'](outputs, targets).item()
        targets_acc = np.array(targets_acc)
        predictions_acc = np.array(predictions_acc)
        return predictions_acc, targets_acc, loss_acc/len(data_loader) if accumulate_loss else None

    def predict(self, testing_df, progress_bar=False):
        testing_loader = create_data_loader_from_dataframe(testing_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        predictions, _, _ =  self._predict(testing_loader, progress_bar=progress_bar)
        return predictions

    def _evaluate(self, test_loader, custom_scores=None, progress_bar=False, compute_loss=False):
        predictions, targets, loss = self._predict(test_loader, accumulate_targets=True, progress_bar=progress_bar, accumulate_loss=compute_loss)
        # calculate scores
        scores_to_compute = self.scores if custom_scores is None else custom_scores
        scores = {name: score(targets, predictions) for name, score in scores_to_compute.items()}
        if compute_loss:
            scores['loss'] = loss
        return scores

    def evaluate(self, testing_df, scores=None, progress_bar=False):
        testing_loader = create_data_loader_from_dataframe(testing_df, self.params['tokenizer'], self.params['tokenizer_max_len'], batch_size=self.params['batch_size'], shuffle=False)
        return self._evaluate(testing_loader, scores, progress_bar=progress_bar)

    def get_train_scores(self):
        return self.train_scores
    
    def get_train_loss(self):
        return self.train_loss

    def get_val_scores(self):
        return self.val_scores
    
    def get_val_loss(self):
        return self.val_loss

'''
function to perform statistical testing to compare 2 models using bootstrap given their predictions
'''
def bootstrap_test(pred_1, pred_2, targets_df, n_tests, sample_size, metric_fun, metric_name, metric_params={}):
    # initial evaluation
    score1 = metric_fun(targets_df, pred_1, **metric_params)
    score2 = metric_fun(targets_df, pred_2, **metric_params)
    print(f'Initial {metric_name}: {score1} {score2}')
    best_model = 'first' if score1 > score2 else 'second'
    if score1 < score2:
        pred_1, pred_2 = pred_2, pred_1
        score1, score2 = score2, score1
    delta = score1 - score2
    print(f'Best model: {best_model}, with delta: {delta}')
    # perform bootstrap
    successes = 0
    for _ in range(n_tests):
        sample = np.random.choice(targets_df.shape[0], sample_size)
        score1 = metric_fun(targets_df.iloc[sample], pred_1[sample], **metric_params)
        score2 = metric_fun(targets_df.iloc[sample], pred_2[sample], **metric_params)
        cur_delta = score1 - score2
        if cur_delta >= 2*delta:
            successes += 1
    p_value = successes/n_tests
    print(f'Successes: {successes}/{n_tests}')
    print(f'p-value: {p_value}')
    return p_value

###########################
# Roberta model
###########################

'''
pytorch Roberta module class
'''
class RobertaModule(torch.nn.Module):
    def __init__(self, n_classes, frozen_layers=-1):
        super(RobertaModule, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        if frozen_layers != -1:
            for param in self.l1.embeddings.parameters():
                param.requires_grad = False
            for i in range(frozen_layers):
                for param in self.l1.encoder.layer[i].parameters():
                    param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
    def get_out_dim(self):
        return self.classifier.out_features
    
ROBERTA_DEFAULT_PARAMS = {
    'optimizer': torch.optim.Adam,
    'tokenizer': RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=False),
    'tokenizer_max_len': None,
    'batch_size': 8,
    'loss_function': torch.nn.BCEWithLogitsLoss(),
    'epochs': 1,
    'learning_rate': 1e-05,
    'regularization': 0,
    'val_patience': np.inf,
    'clip_grad_norm': -1
}

class Roberta(SimpleModelInterface):
    def _build_model(self):
        params = self.get_params()
        if 'n_classes' not in params:
            raise ValueError('Number of classes not specified in model parameters')
        frozen_layers = -1 if 'frozen_layers' not in params else params['frozen_layers']
        return RobertaModule(params['n_classes'], frozen_layers)

    def _create_model_params(self, params_dict):
        params = ROBERTA_DEFAULT_PARAMS.copy()
        params.update(params_dict)
        return params

    def __init__(self, scores={}, model_params_dict={}, checkpoint=None):
        super().__init__(scores, model_params_dict, checkpoint=checkpoint)

###########################
# Bert model
###########################

class BertMultiLabelClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768,n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def get_out_dim(self):
        return self.classifier.out_features
    
BERT_DEFAULT_PARAMS = {
    'optimizer': torch.optim.Adam,
    'tokenizer': BertTokenizer.from_pretrained('bert-base-cased', truncation=True, do_lower_case=False),
    'tokenizer_max_len': None,
    'batch_size': 16,
    'loss_function': torch.nn.BCEWithLogitsLoss(),
    'epochs': 1,
    'learning_rate': 1e-05,
    'regularization': 0,
    'val_patience': np.inf,
    'clip_grad_norm': -1
}

class Bert(SimpleModelInterface):
    def _build_model(self):
        params = self.get_params()
        if 'n_classes' not in params:
            raise ValueError('Number of classes not specified in model parameters')
        return BertMultiLabelClassifier(params['n_classes'])

    def _create_model_params(self, params_dict):
        params = BERT_DEFAULT_PARAMS.copy()
        params.update(params_dict)
        return params

    def __init__(self, scores={}, model_params_dict={}, checkpoint=None):
        super().__init__(scores, model_params_dict, checkpoint=checkpoint)

###########################
# SocBert model
###########################

class SocbertMultiLabelClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(SocbertMultiLabelClassifier, self).__init__()
        self.socbert = AutoModel.from_pretrained("sarkerlab/SocBERT-base")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768,n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.socbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #you get the hidden rappresentation of CLS token
        hidden_state = outputs[0]
        cls_output = hidden_state[:, 0]
        outputs = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
    
    def get_out_dim(self):
        return self.classifier.out_features
    
SOCBERT_DEFAULT_PARAMS = {
    'optimizer': torch.optim.Adam,
    'tokenizer': AutoTokenizer.from_pretrained("sarkerlab/SocBERT-base",truncation=True, do_lower_case=False),
    'tokenizer_max_len': None,
    'batch_size': 16,
    'loss_function': torch.nn.BCEWithLogitsLoss(),
    'epochs': 1,
    'learning_rate': 1e-05,
    'regularization': 0,
    'val_patience': np.inf,
    'clip_grad_norm': -1
}

class Socbert(SimpleModelInterface):
    def _build_model(self):
        params = self.get_params()
        if 'n_classes' not in params:
            raise ValueError('Number of classes not specified in model parameters')
        return SocbertMultiLabelClassifier(params['n_classes'])

    def _create_model_params(self, params_dict):
        params = SOCBERT_DEFAULT_PARAMS.copy()
        params.update(params_dict)
        return params

    def __init__(self, scores={}, model_params_dict={}, checkpoint=None):
        super().__init__(scores, model_params_dict, checkpoint=checkpoint)

###########################
# Llama model
###########################

class Llama3():
    def __init__(self, name,  scores={},  samples = None, mode = "single", emotions = None):
        self.model = outlines.models.transformers(name, model_kwargs ={"torch_dtype": torch.float16},
                                                   device="cuda" if cuda.is_available() else "cpu")
        self.emotions = emotions
        self.mode = mode
        # if multilabel, we ask for "True" or "False" for each emotion, else for emotion name directly. grouped is just to differentiate csv names
        self.choices = ["True", "False"] if self.mode == "multi" or self.mode == "grouped" else self.emotions 

        self.generator = outlines.generate.choice(self.model, self.choices)
        self.scores = scores
        self.samples = samples

    
    def classify(self, data : Llama_EmotionsData, progress_bar = False):
        if self.mode == "single":
            predictions = self.single_predict(data, progress_bar = progress_bar)
        else:
            predictions = self.multi_predict(data, progress_bar = progress_bar)

        self.evaluate(data.targets, predictions)

    def single_predict(self, data, progress_bar = False):
        # classifies single label data  
        predictions = []
        prompt = ""
        if self.samples:
            prompt += f"""Look at these examples: \n {self.samples} \n Now""" 
        try:
            for entry in tqdm(data.text, disable=not progress_bar):
                prompt = f"""Classify the following sentence:\n {entry} \nChoose among the following emotions: {self.emotions}"""
                predictions.append(self.generator(prompt))

        except (ValueError, KeyError): # loop gives error at index of last entry 
            pass

        return predictions
    
    def multi_predict(self, data, progress_bar = False):
        # classifies multilabel data
        # for each sentence, we ask if it evokes each emotion
        # predictions will be a list of lists, each list containing the emotions evoked by the corresponding sentence
        predictions = []
        emotions = self.remove_ekman_prefix(self.emotions) if self.mode == "grouped" else self.emotions #remove "ekman_" prefix if present
        try:
            for entry in tqdm(data.text, disable=not progress_bar):
                sentence_emotions = []
                for emotion in emotions:
                    prompt = f"""Consider the following sentence:\n {entry} \nDoes it evoke the emotion '{emotion}'? Answer with 'True' or 'False'."""
                    response = self.generator(prompt)
                    if response == "True":
                        sentence_emotions.append(emotion)
                predictions.append(sentence_emotions)
        except (ValueError, KeyError): # loop gives error at index of last entry 
            pass

        return predictions
    
    def evaluate(self, targets, predictions):
        # evaluate the model
        if self.mode == "single":
            lb = LabelBinarizer()
        else:
            lb = MultiLabelBinarizer()
        bin_predictions = lb.fit_transform(predictions)
        bin_predictions = pd.DataFrame(bin_predictions, columns = lb.classes_, dtype="int64") 
        n_samples = 0 if self.samples is None else len(self.samples)
        csv_name = "llama_" + self.mode + "_" + str(n_samples) + "_predictions.csv"
        csv_path = './results/llama_predictions/' + csv_name
        bin_predictions.to_csv(csv_path)
        scores = {name: score(targets, bin_predictions) for name, score in self.scores.items()}
        plot_score_barplot(targets, bin_predictions, self.emotions)
        if not self.mode == "grouped": #evaluation is done in model_comparison.ipynb
            scores_dict = get_scores_dict(bin_predictions, targets, self.emotions)
            custom_classification_report(scores_dict, self.emotions)
            if not self.mode == "single":
                plot_multilabel_confusion_heatmap(targets, np.array(bin_predictions), self.emotions, self.emotions, normalize=True)
            return scores

    def remove_ekman_prefix(self, emotions: list, prefix = "ekman_"):
        cleaned_emotions = []
        for text in emotions:
            if text.startswith(prefix):
                cleaned_emotions.append(text[len(prefix):])
        return cleaned_emotions 