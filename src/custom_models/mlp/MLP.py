import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
# base directory for importing is the direct running file (run_experiment.py)
# `_shared_imports` and `utils` are at the same directory, and thus no need for `src.`
from _shared_imports import *
from lookahead_optimizer import Lookahead
from utils import detach_tensors, get_labels_lists, reshape_batch


class LitMLP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_rate = args.lr

        self.input_dim = args.num_features
        self.output_dim = args.num_classes
        self.activation = args.mlp_activation
        self.hidden_layer_list = args.mlp_hidden_layer_list
        self.batchnorm = bool(args.batchnorm)

        self.model = Net(input_dim=self.input_dim,
                         output_dim=self.output_dim,
                         activation=self.activation,
                         hidden_layer_list=self.hidden_layer_list,
                         batch_normalization=self.batchnorm)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x: torch.Tensor, y_true: torch.Tensor):
        y_hat = self.forward(x)
        y_pred = y_hat if self.output_dim == 1 else torch.argmax(y_hat, dim=1)

        losses = self.compute_mlp_loss(y_hat, y_true)

        return y_hat, y_pred, losses

    def compute_mlp_loss(self, y_hat, y_true):
        losses = {}
        losses['total'] = torch.zeros(1, device=self.device)
        losses['mse'] = torch.zeros(1, device=self.device)
        losses['cross_entropy'] = torch.zeros(1, device=self.device)

        # compute loss for prediction
        if self.output_dim == 1:
            losses['mse'] = F.mse_loss(input=y_hat.squeeze(-1), target=y_true)
        else:
            losses['cross_entropy'] = F.cross_entropy(input=y_hat,
                                                      target=y_true,
                                                      weight=torch.tensor(self.args.class_weights, device=self.device))

        losses['total'] = losses['mse'] + losses['cross_entropy']

        return losses

    def log_losses(self, losses, key, dataloader_name=""):
        self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
        # loss for basic regression and classification
        self.log(f"{key}/mse_loss{dataloader_name}", losses['mse'].item())
        self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item())

    def log_epoch_metrics(self, outputs, key, dataloader_name=""):
        if self.output_dim == 1:
            return

        y_true, y_pred = get_labels_lists(outputs)
        self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred))
        self.log(f'{key}/F1_weighted{dataloader_name}', f1_score(y_true, y_pred, average='weighted'))
        self.log(f'{key}/precision_weighted{dataloader_name}', precision_score(y_true, y_pred, average='weighted'))
        self.log(f'{key}/recall_weighted{dataloader_name}', recall_score(y_true, y_pred, average='weighted'))
        if self.args.num_classes == 2:
            self.log(f'{key}/AUROC_weighted{dataloader_name}', roc_auc_score(y_true, y_pred, average='weighted'))

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_hat, y_pred, losses = self.compute_loss(x, y_true)
        self.log_losses(losses, key='train')

        return {
            'loss': losses['total'],
            'losses': detach_tensors(losses),
            'y_true': y_true.detach(),
            'y_pred': y_pred.detach(),
        }

    def training_epoch_end(self, outputs):
        if self.trainer.global_step % self.args.log_every_n_steps == 0:
            self.log_epoch_metrics(outputs, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        - dataloader_idx (int) tells which dataloader is the `batch` coming from
        """
        x, y_true = reshape_batch(batch)
        y_hat, y_pred, losses = self.compute_loss(x, y_true)

        return {
            'losses': detach_tensors(losses),
            'y_true': y_true.detach(),
            'y_pred': y_pred.detach(),
        }

    def validation_epoch_end(self, outputs_all_dataloaders):
        """
        - outputs: when no_dataloaders==1 --> A list of dictionaries corresponding to a validation step.
                   when no_dataloaders>1  --> List with length equal to the number of validation dataloaders. Each element is a list with the dictionaries corresponding to a validation step.
        """
        ### Log losses and metrics
        # `outputs_all_dataloaders` is expected to a list of dataloaders.
        # However, when there's only one dataloader, outputs_all_dataloaders is NOT a list.
        # Thus, we transform it in a list to preserve compatibility
        if len(self.args.val_dataloaders_name) == 1:
            outputs_all_dataloaders = [outputs_all_dataloaders]

        for dataloader_id, outputs in enumerate(outputs_all_dataloaders):
            losses = {
                'total': np.mean([output['losses']['total'].item() for output in outputs]),
                'mse': np.mean([output['losses']['mse'].item() for output in outputs]),
                'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
            }
            if dataloader_id == 0:  # original validation dataset
                dataloader_name = ""
            else:
                dataloader_name = f"__{self.args.val_dataloaders_name[dataloader_id]}"

            self.log_losses(losses, key='valid', dataloader_name=dataloader_name)
            self.log_epoch_metrics(outputs, key='valid', dataloader_name=dataloader_name)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = reshape_batch(batch)

        y_hat, y_pred, losses = self.compute_loss(x, y_true)

        return {
            'losses': detach_tensors(losses),
            'y_true': y_true.detach(),
            'y_pred': y_pred.detach(),
            'y_hat': y_hat.detach(),
        }

    def test_epoch_end(self, outputs):
        ### Save losses
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'mse': np.mean([output['losses']['mse'].item() for output in outputs]),
            'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs]),
        }
        self.log_losses(losses, key=self.log_test_key)
        self.log_epoch_metrics(outputs, self.log_test_key)

        # save prediction probabilities
        if outputs[0]['y_hat'] is not None:
            y_list = [output['y_true'].cpu().numpy() for output in outputs]
            y_hat_list = [output['y_hat'].cpu().numpy() for output in outputs]
            y_all = np.concatenate(y_list, axis=0)[:, np.newaxis]
            y_hat_all = np.concatenate(y_hat_list, axis=0)
            if self.output_dim > 1:
                y_hat_all = scipy.special.softmax(y_hat_all, axis=1)
            y_hat_all = np.concatenate([y_hat_all, y_all], axis=1)
            y_hat_all = pd.DataFrame(y_hat_all)

            y_hat_all = wandb.Table(dataframe=y_hat_all)
            wandb.log({f'{self.log_test_key}_y_hat_and_true': y_hat_all})

    def configure_optimizers(self):
        params = self.parameters()

        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params,
                                          lr=self.learning_rate,
                                          weight_decay=self.args.weight_decay,
                                          betas=[0.9, 0.98])
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)

        if self.args.lookahead_optimizer:
            optimizer = Lookahead(optimizer, la_steps=5, la_alpha=0.5)

        if self.args.lr_scheduler == 'none':
            return optimizer
        else:
            if self.args.lr_scheduler == 'plateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                          mode='min',
                                                                          factor=0.5,
                                                                          patience=30,
                                                                          verbose=True)
            elif self.args.lr_scheduler == 'cosine_warm_restart':
                # Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
                # 	splits the scheduler for 6 periods. We split into 6 periods as well.
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.args.cosine_warm_restart_t_0,
                    eta_min=self.args.cosine_warm_restart_eta_min,
                    verbose=True)
            elif self.args.lr_scheduler == 'linear':
                lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                                 start_factor=self.args.lr,
                                                                 end_factor=3e-5,
                                                                 total_iters=self.args.max_steps /
                                                                 self.args.val_check_interval)
            elif self.args.lr_scheduler == 'lambda':

                def scheduler(epoch):
                    if epoch < 500:
                        return 0.995**epoch
                    else:
                        return 0.1

                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)
            else:
                raise Exception()

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': f'valid/{self.args.metric_model_selection}',
                    'interval': 'step',
                    'frequency': self.args.val_check_interval,
                    'name': 'lr_scheduler'
                }
            }


class Net(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str,
                 hidden_layer_list: list,
                 batch_normalization: bool = True) -> None:
        """Prediction Network for classification or regression

        Args:
            output_dim (int): number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            activation (str): activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer for the prediction net, example: [200,200]
        """
        super().__init__()

        self.output_dim = output_dim
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]
        self.fn = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            # use BN after activation has better performance
            if batch_normalization:
                self.fn.add_module('bn{}'.format(i), nn.BatchNorm1d(full_layer_list[i]))
            self.fn.add_module('fn{}'.format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.fn.add_module('act{}'.format(i), self.act)

        self.head = nn.Sequential()
        if batch_normalization:
            self.head.add_module('bn', nn.BatchNorm1d(full_layer_list[-1]))
        self.head.add_module('fn', nn.Linear(full_layer_list[-1], output_dim))
        if self.output_dim != 1:
            self.head.add_module('act', self.act)
            # when using cross-entropy loss in pytorch, we do not need to use softmax.
            # self.head.add_module('softmax', nn.Softmax(-1))

    def forward(self, x):
        x_emb = self.fn(x)
        x = self.head(x_emb)

        return x


class DeactFunc(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def get_activation(value):
    if value == 'relu':
        return torch.nn.ReLU()
    elif value == 'l_relu':
        # set the slope to align tensorflow
        return torch.nn.LeakyReLU(negative_slope=0.2)
    elif value == 'sigmoid':
        return torch.nn.Sigmoid()
    elif value == 'tanh':
        return torch.nn.Tanh()
    elif value == 'none':
        return DeactFunc()
    else:
        raise NotImplementedError('activation for the gating network not recognized')
