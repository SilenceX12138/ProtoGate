from _shared_imports import *


def get_labels_lists(outputs):
    all_y_true, all_y_pred = [], []
    for output in outputs:
        all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
        all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

    return all_y_true, all_y_pred


def compute_all_metrics(args, y_true, y_pred):
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['F1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    if args.num_classes == 2:
        metrics['AUROC_weighted'] = roc_auc_score(y_true, y_pred, average='weighted')

    return metrics


def detach_tensors(tensors):
    """
	Detach losses 
	"""
    if type(tensors) == list:
        detached_tensors = list()
        for tensor in tensors:
            detach_tensors.append(tensor.detach())
    elif type(tensors) == dict:
        detached_tensors = dict()
        for key, tensor in tensors.items():
            detached_tensors[key] = tensor.detach()
    else:
        raise Exception("tensors must be a list or a dict")

    return detached_tensors


def reshape_batch(batch):
    """
	When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
	
	This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
	"""
    x, y = batch
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)

    return x, y


def get_f1_score_gate(args, stochastic_gate):
    dataset = args.dataset
    test_ids = args.test_ids

    if len(stochastic_gate.shape) == 1:
        stochastic_gate = stochastic_gate.reshape(1, -1).repeat(len(test_ids), axis=0)
    if not (type(stochastic_gate) is np.ndarray):
        stochastic_gate = stochastic_gate.detach().cpu().numpy()
    gate_pred = (stochastic_gate > 0).astype(int)

    if dataset not in ['syn1', 'syn2', 'syn3']:
        return 0

    assert test_ids.shape[0] == gate_pred.shape[0]

    gate_true = np.zeros_like(gate_pred)
    if dataset == 'syn1':
        # 2N = 2000 samples with 11 features
        N1 = 50
        feature_group_1 = [0, 1, 2, 10]
        feature_group_2 = [2, 3, 4, 5, 10]
        feature_group = [feature_group_1, feature_group_2]
    elif dataset == 'syn2':
        N1 = 50
        feature_group_1 = [2, 3, 4, 5, 6, 10]
        feature_group_2 = [6, 7, 8, 9, 10]
        feature_group = [feature_group_1, feature_group_2]
    elif dataset == 'syn3':
        # 2N = 2000 samples with 11 features
        N1 = 50
        feature_group_1 = [0, 1, 8, 10]
        feature_group_2 = [6, 7, 8, 9, 10]
        feature_group = [feature_group_1, feature_group_2]
    else:
        return 0

    for i in range(gate_true.shape[0]):
        gate_true[i, feature_group[(test_ids[i] > N1).astype(int)]] = 1

    score = f1_score(gate_true, gate_pred, average='micro')

    return score


def compute_sparsity(coef: np.ndarray):
    """compute the number of selected features

    Args:
        coef (np.ndarray): weights of the first layer of (n_classes, n_features)
    """

    gate = (np.linalg.norm(coef, ord=2, axis=0) != 0).astype(int)
    num_selected_features = np.sum(gate)

    return num_selected_features


def gate_bin2dec(gate: np.ndarray):
    gate_binary_str = ''.join(str(b) for b in gate)
    gate_dec_str = str(int(gate_binary_str, 2))

    return gate_dec_str


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])
