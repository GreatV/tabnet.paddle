import paddle
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_multitask_output, check_output_dim
import scipy


class TabNetMultiTaskClassifier(TabModel):
    def __post_init__(self):
        super(TabNetMultiTaskClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = paddle.nn.functional.cross_entropy
        self._default_metric = "logloss"

    def prepare_target(self, y):
        y_mapped = y.copy()
        for task_idx in range(y.shape[1]):
            task_mapper = self.target_mapper[task_idx]
            y_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y[:, task_idx])
        return y_mapped

    def compute_loss(self, y_pred, y_true):
        """
        Computes the loss according to network output and targets

        Parameters
        ----------
        y_pred : list of tensors
            Output of network
        y_true : LongTensor
            Targets label encoded

        Returns
        -------
        loss : torch.Tensor
            output of loss function(s)

        """
        loss = 0
        y_true = y_true.astype(dtype="int64")
        if isinstance(self.loss_fn, list):
            for task_loss, task_output, task_id in zip(
                self.loss_fn, y_pred, range(len(self.loss_fn))
            ):
                loss += task_loss(task_output, y_true[:, task_id])
        else:
            for task_id, task_output in enumerate(y_pred):
                loss += self.loss_fn(task_output, y_true[:, task_id])
        loss /= len(y_pred)
        return loss

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = []
        for i in range(len(self.output_dim)):
            score = np.vstack([x[i] for x in list_y_score])
            score = softmax(score, axis=1)
            y_score.append(score)
        return y_true, y_score

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        output_dim, train_labels = infer_multitask_output(y_train)
        for _, y in eval_set:
            for task_idx in range(y.shape[1]):
                check_output_dim(train_labels[task_idx], y[:, task_idx])
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [
            {class_label: index for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ]
        self.preds_mapper = [
            {str(index): str(class_label) for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ]
        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        results : np.array
            Predictions of the most probable class
        """
        self.network.eval()
        if scipy.sparse.issparse(X):
            dataloader = paddle.io.DataLoader(
                dataset=SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = paddle.io.DataLoader(
                dataset=PredictDataset(X), batch_size=self.batch_size, shuffle=False
            )
        results = {}
        for data in dataloader:
            data = data.astype(dtype="float32")
            output, _ = self.network(data)
            predictions = [
                paddle.argmax(x=paddle.nn.Softmax(axis=1)(task_output), axis=1)
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
                for task_output in output
            ]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        results = [np.hstack(task_res) for task_res in results.values()]
        results = [
            np.vectorize(self.preds_mapper[task_idx].get)(task_res.astype(str))
            for task_idx, task_res in enumerate(results)
        ]
        return results

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : list of np.ndarray

        """
        self.network.eval()
        if scipy.sparse.issparse(X):
            dataloader = paddle.io.DataLoader(
                dataset=SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = paddle.io.DataLoader(
                dataset=PredictDataset(X), batch_size=self.batch_size, shuffle=False
            )
        results = {}
        for data in dataloader:
            data = data.astype(dtype="float32")
            output, _ = self.network(data)
            predictions = [
                paddle.nn.Softmax(axis=1)(task_output).cpu().detach().numpy()
                for task_output in output
            ]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
        res = [np.vstack(task_res) for task_res in results.values()]
        return res
