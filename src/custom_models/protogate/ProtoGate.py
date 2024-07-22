from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# base directory for importing is the direct running file (run_experiment.py)
# `_shared_imports` and `utils` are at the same directory, and thus no need for `src.`
from _shared_imports import *
from lookahead_optimizer import Lookahead
from utils import detach_tensors, get_f1_score_gate, get_labels_lists

# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class LitProtoGate(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_rate = args.lr

        self.input_dim = args.num_features
        self.output_dim = args.num_classes

        self.feature_selection = args.feature_selection

        self.a = args.protogate_a
        self.sigma = args.protogate_sigma
        self.lam = args.protogate_lam_local

        self.protogate_init_std = args.protogate_init_std

        self.gating_network = GatingNet(input_dim=self.input_dim,
                                        a=args.protogate_a,
                                        sigma=args.protogate_sigma,
                                        activation=args.protogate_activation_gating,
                                        hidden_layer_list=args.protogate_gating_hidden_layer_list)

        self.l1_coef = args.protogate_lam_global

        self.pred_k = args.pred_k
        self.pred_coef = args.pred_coef
        self.proto_layer = KNNNet(k=self.pred_k, tau=args.sorting_tau)
        self.x_neighbour = args.x_train
        self.y_neighbour = args.y_train

        # initialize weights with mannually set parameters
        if self.protogate_init_std > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight.data,
                                        mean=0,
                                        std=self.protogate_init_std,
                                        a=-2 * self.protogate_init_std,
                                        b=2 * self.protogate_init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def compute_loss(self, x: torch.Tensor, y_true: torch.Tensor):
        x_selected, self.alpha, self.stochastic_gate = self.gating_network(x)

        if self.training:
            # use the data from the same batch as neighours for classification
            x_neighbour_selected = x_selected
            y_neighbour = y_true
        else:
            # use the data from the train set as the neighbour for classification
            x_neighbour_selected, _, _ = self.gating_network(self.x_neighbour.to(self.device))
            y_neighbour = self.y_neighbour.to(self.device)

        # omit duplicate one-hot
        y_true = F.one_hot(y_true, num_classes=self.args.num_classes)
        y_neighbour = F.one_hot(y_neighbour, num_classes=self.args.num_classes)

        y_pred, neighbour_list = proto_predict(x_selected, x_neighbour_selected, y_neighbour, self.pred_k)
        losses = self.compute_pred_loss(x_selected, x_neighbour_selected, y_true, y_neighbour)

        return y_pred, losses, neighbour_list

    def compute_pred_loss(self, x_query, x_cand, y_query, y_neighbor):
        losses = {}
        losses['l0_norm'] = torch.zeros(1, device=self.device)
        losses['l1_norm'] = torch.zeros(1, device=self.device)
        losses['pred_loss'] = torch.zeros(1, device=self.device)

        if self.feature_selection:
            losses['l0_norm'] = self.compute_sparsity_loss(self.alpha)
            losses['l1_norm'] = self.l1_coef * torch.norm(self.gating_network.embed.fn0.weight, p=1)

        losses['pred_loss'] = self.pred_k + \
                              pred_loss(self.proto_layer, x_query, x_cand, y_query, y_neighbor)
        losses['pred_loss'] = self.pred_coef * losses['pred_loss']

        losses['total'] = losses['pred_loss'] + losses['l1_norm'] + losses['l0_norm']

        return losses

    def compute_sparsity_loss(self, input2cdf):
        # gates regularization
        reg = 0.5 - 0.5 * torch.erf((-input2cdf) / (self.sigma * np.sqrt(2)))
        loss_reg_gates = self.lam * torch.mean(torch.sum(reg, dim=1))

        return loss_reg_gates

    def log_losses(self, losses, key, dataloader_name=""):
        self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item())
        # regularization for sparsity
        self.log(f"{key}/l0_norm{dataloader_name}", losses['l0_norm'].item())
        self.log(f"{key}/l1_norm{dataloader_name}", losses['l1_norm'].item())
        # loss for prototype-based classification
        self.log(f"{key}/pred_loss{dataloader_name}", losses['pred_loss'].item())

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

    def log_feature_selection(self, key, dataloader_name=""):
        num_open_gate = (self.stochastic_gate > 0).sum(dim=1).float()
        pos_open_gate = (self.stochastic_gate > 0).sum(dim=0).float()
        self.log(f'{key}/median_open_gates{dataloader_name}', torch.median(num_open_gate).item())
        self.log(f'{key}/mean_open_gates{dataloader_name}', torch.mean(num_open_gate).item())
        self.log(f'{key}/std_open_gates{dataloader_name}', torch.std(num_open_gate).item())
        self.log(f'{key}/union_open_gates{dataloader_name}', (pos_open_gate > 0).sum().float().item())

        # display micro F1 score for feature selection when testing
        if 'test' in key and hasattr(self.args, 'test_ids'):
            f1_score_gate = get_f1_score_gate(self.args, self.stochastic_gate)
            self.log(f'{key}/micro_F1_score_of_predicted_feature_gates{dataloader_name}', f1_score_gate)

    def parse_losses(self, outputs):
        losses = {
            'total': np.mean([output['losses']['total'].item() for output in outputs]),
            'l0_norm': np.mean([output['losses']['l0_norm'].item() for output in outputs]),
            'l1_norm': np.mean([output['losses']['l1_norm'].item() for output in outputs]),
            'pred_loss': np.mean([output['losses']['pred_loss'].item() for output in outputs]),
        }

        return losses

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pred, losses, _ = self.compute_loss(x, y_true)
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
            if self.feature_selection:
                self.log_feature_selection(key='train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        - dataloader_idx (int) tells which dataloader is the `batch` coming from
        """
        x, y_true = batch
        y_pred, losses, _ = self.compute_loss(x, y_true)

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
            losses = self.parse_losses(outputs)
            if dataloader_id == 0:  # original validation dataset
                dataloader_name = ""
            else:
                dataloader_name = f"__{self.args.val_dataloaders_name[dataloader_id]}"

            self.log_losses(losses, key='valid', dataloader_name=dataloader_name)
            self.log_epoch_metrics(outputs, key='valid', dataloader_name=dataloader_name)
            if self.feature_selection:
                self.log_feature_selection(key='valid', dataloader_name=dataloader_name)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = batch

        y_pred, losses, neighbour_list = self.compute_loss(x, y_true)

        return {
            'losses': detach_tensors(losses),
            'y_true': y_true.detach(),
            'y_pred': y_pred.detach(),
            'neighbours': neighbour_list.detach(),
        }

    def test_epoch_end(self, outputs):
        ### Save losses
        losses = self.parse_losses(outputs)
        self.log_losses(losses, key=self.log_test_key)
        self.log_epoch_metrics(outputs, self.log_test_key)
        if self.feature_selection:
            self.log_feature_selection(key=self.log_test_key)

        # save idx of instance-wise selected features
        if self.feature_selection == True:
            gate_all = (self.stochastic_gate.detach().cpu() > 0).to(int)
            gate_bin_all = []
            for gate in gate_all:
                gate_binary_str = ''.join(str(int(b)) for b in gate)
                gate_bin_all.append(str(int(gate_binary_str, 2)))
            gate_all = wandb.Table(dataframe=pd.DataFrame(gate_bin_all))
            wandb.log({f'{self.log_test_key}_all_gate': gate_all})

        # save the neighbours for inference
        neighbours_all = np.asarray([out['neighbours'].cpu().numpy() for out in outputs]).reshape(-1, self.pred_k)
        neighbours_all = wandb.Table(dataframe=pd.DataFrame(neighbours_all))
        wandb.log({f'{self.log_test_key}_neighbours': neighbours_all})

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


class GatingNet(nn.Module):

    def __init__(self, input_dim: int, a: float, sigma: float, activation: str, hidden_layer_list: list) -> None:
        """Gating Network for feature selection

        Args:
            input_dim (int): input dimension of the gating network
            a (float): coefficient in hard relu activation function
            sigma (float): std of the gaussion reparameterization noise
            activation (str): activation function of the gating net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer of the gating net, example: [200,200]
        """
        super().__init__()

        self.a = a
        self.sigma = sigma
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]

        self.embed = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            self.embed.add_module('fn{}'.format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.embed.add_module('act{}'.format(i), self.act)

        self.gate = nn.Sequential()
        self.gate.add_module('fn', nn.Linear(full_layer_list[-1], input_dim))
        self.gate.add_module('act', self.act)

    def forward(self, x):
        x_all = x
        x_emb = self.embed(x)
        alpha = self.gate(x_emb)
        stochastic_gate = self.get_stochastic_gate(alpha)
        x_selected = x_all * stochastic_gate

        return x_selected, alpha, stochastic_gate

    def get_stochastic_gate(self, alpha):
        """
        This function replaced the feature_selector function in order to save Z
        """
        # gaussian reparametrization
        noise = self.sigma * torch.randn(alpha.shape, device=alpha.device) \
                if self.training == True else torch.zeros(1, device=alpha.device)
        z = alpha + noise
        stochastic_gate = self.hard_sigmoid(z)

        return stochastic_gate

    def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = self.a * x
        x = torch.clamp(x, 0, 1)

        return x


class KNNNet(torch.nn.Module):

    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1, similarity='euclidean'):
        super(KNNNet, self).__init__()
        self.k = k
        self.soft_sort = HybridSort(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples
        self.similarity = similarity

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, query, neighbors, tau=1.0):
        if self.similarity == 'euclidean':
            diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
            squared_diffs = diffs**2
            l2_norms = squared_diffs.sum(2)
            norms = l2_norms
            scores = -norms  # B * N
        elif self.similarity == 'cosine':
            scores = F.cosine_similarity(query.unsqueeze(1), neighbors.unsqueeze(0), dim=2) - 1
        else:
            raise ValueError('Unknown similarity for KNNNet: {}'.format(self.similarity))

        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)  # B*N*N
            top_k = P_hat[:, :self.k, :].sum(1)  # B*N
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples, ))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k


class HybridSort(torch.nn.Module):

    def __init__(self, tau=1.0, hard=False):
        super(HybridSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        device = scores.device
        one = torch.FloatTensor(dim, 1).fill_(1).to(device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(torch.FloatTensor).to(device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(dim0=1, dim1=0).flatten().type(
                torch.LongTensor).to(device)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().type(torch.LongTensor).to(device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


class PL(Distribution):

    arg_constraints = {'scores': constraints.positive, 'tau': constraints.positive}
    has_rsample = True

    @property
    def mean(self):
        # mode of the PL distribution
        return self.relaxed_sort(self.scores)

    def __init__(self, scores, tau, hard=True, validate_args=None):
        """
        scores. Shape: (batch_size x) n 
        tau: temperature for the relaxation. Scalar.
        hard: use straight-through estimation if True
        """
        self.scores = scores.unsqueeze(-1)
        self.tau = tau
        self.hard = hard
        self.n = self.scores.size()[1]

        if isinstance(scores, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scores.size()
        super(PL, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args:
            if not torch.gt(self.scores, torch.zeros_like(self.scores)).all():
                raise ValueError("PL is not defined when scores <= 0")

    def relaxed_sort(self, inp):
        """
        inp: elements to be sorted. Typical shape: batch_size x n x 1
        """
        bsize = inp.size()[0]
        dim = inp.size()[1]
        one = FloatTensor(dim, 1).fill_(1)

        A_inp = torch.abs(inp - inp.permute(0, 2, 1))
        B = torch.matmul(A_inp, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).type(FloatTensor)
        C = torch.matmul(inp, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(dim0=1,
                                                                                    dim1=0).flatten().type(LongTensor)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().type(LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

    def rsample(self, sample_shape, log_score=True):
        """
        sample_shape: number of samples from the PL distribution. Scalar.
        """
        with torch.enable_grad():  # torch.distributions turns off autograd
            n_samples = sample_shape[0]

            def sample_gumbel(samples_shape, eps=1e-20):
                U = torch.zeros(samples_shape, device='cuda').uniform_()
                return -torch.log(-torch.log(U + eps) + eps)

            if not log_score:
                log_s_perturb = torch.log(self.scores.unsqueeze(0)) + sample_gumbel([n_samples, 1, self.n, 1])
            else:
                log_s_perturb = self.scores.unsqueeze(0) + sample_gumbel([n_samples, 1, self.n, 1])
            log_s_perturb = log_s_perturb.view(-1, self.n, 1)
            P_hat = self.relaxed_sort(log_s_perturb)
            P_hat = P_hat.view(n_samples, -1, self.n, self.n)

            return P_hat.squeeze()

    def log_prob(self, value):
        """
        value: permutation matrix. shape: batch_size x n x n
        """
        permuted_scores = torch.squeeze(torch.matmul(value, self.scores))
        log_numerator = torch.sum(torch.log(permuted_scores), dim=-1)
        idx = LongTensor([i for i in range(self.n - 1, -1, -1)])
        invert_permuted_scores = permuted_scores.index_select(-1, idx)
        denominators = torch.cumsum(invert_permuted_scores, dim=-1)
        log_denominator = torch.sum(torch.log(denominators), dim=-1)
        return (log_numerator - log_denominator)


class DeactFunc(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def proto_predict(query, neighbors, neighbor_labels, k):
    '''
    query: p
    neighbors: n x p
    neighbor_labels: n x num_classes
    '''
    query, neighbors = query.detach(), neighbors.detach()
    diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
    # squared_diffs = diffs**2
    # norms = squared_diffs.sum(-1)
    norms = torch.norm(diffs, p=2, dim=-1)
    indices = torch.argsort(norms, dim=-1).to(neighbor_labels.device)
    labels = neighbor_labels[indices[:, :k]]  # n x k x num_classes
    label_counts = labels.sum(dim=1)  # n x num_classes
    prediction = torch.argmax(label_counts, dim=1)  # n

    return prediction, indices[:, :k]


def pred_loss(proto_layer, query, neighbors, query_label, neighbor_labels, method='deterministic'):
    # query: batch_size x p
    # neighbors: 10k x p
    # query_labels: batch_size x [10] one-hot
    # neighbor_labels: n x [10] one-hot
    if method == 'deterministic':
        # top_k_ness is the sum of top-k row of permutation matrix
        top_k_ness = proto_layer(query, neighbors)  # (B*512, N*512) => (B, N)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)  # (B, N)
        correct_in_top_k = (correct * top_k_ness).sum(-1)  # [B]
        loss = -correct_in_top_k
        # loss = 1 / correct_in_top_k
        return loss.mean()
    elif method == 'stochastic':
        top_k_ness = proto_layer(query, neighbors)
        correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)
        correct_in_top_k = (correct.unsqueeze(0) * top_k_ness).sum(-1)
        loss = -correct_in_top_k
        return loss.mean()
    else:
        raise ValueError(method)


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
