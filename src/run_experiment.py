import collections
import json
import logging
import pickle
import pprint
import warnings
from dataclasses import dataclass
from pickletools import optimize
from statistics import mode

import lightgbm as lgb
import optuna
import pytorch_lightning
import sklearn
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import wandb
from _config import BASE_DIR, DATA_DIR
from dataset import *
from models import create_model
from param_tune import optuna_objective
from utils import (compute_all_metrics, gate_bin2dec, get_f1_score_gate,
                   get_one_hot)


def get_run_name(args):
    if args.model == 'dnn':
        run_name = 'mlp'
    elif args.model == 'dietdnn':
        run_name = 'mlp_wpn'
    else:
        run_name = args.model

    if args.sparsity_type == 'global':
        run_name += '_SPN_global'
    elif args.sparsity_type == 'local':
        run_name += '_SPN_local'

    return run_name


def create_wandb_logger(args):
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        group=args.group,
        job_type=args.job_type,
        tags=args.tags,
        notes=args.notes,
        # reinit=True,
        log_model=args.wandb_log_model,
        settings=wandb.Settings(start_method="thread"))
    wandb_logger.experiment.config.update(args)  # add configuration file

    return wandb_logger


def run_experiment(args):
    # `repeat_id` and `test_split` are used in grid search (sweep)
    args.suffix_wand_run_name = f"repeat-{args.repeat_id}__test-{args.test_split}"

    # Load dataset
    print(f"\nInside training function")
    print(f"\nLoading data {args.dataset}...")
    data_module = create_data_module(args)
    args.x_train = data_module.train_dataloader().dataset.X
    args.y_train = data_module.train_dataloader().dataset.y

    # below parameters are set when creating data module
    print(f"Train/Valid/Test splits of sizes {args.train_size}, {args.valid_size}, {args.test_size}")
    print(f"Num of features: {args.num_features}")
    if args.full_batch_training == True:
        args.batch_size = args.train_size
        if args.model == 'protogate':
            args.proto_num_train_queries = args.train_size
            args.proto_num_train_neighbors = args.train_size

    # Intialize wandb logging
    wandb_logger = create_wandb_logger(args)
    wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"

    # log the training data of training data for visualisation on MNIST
    if args.dataset == 'mnist':
        train_ids = wandb.Table(dataframe=pd.DataFrame(args.train_ids))
        wandb.log({f'train_ids': train_ids})

    # model training
    # Scikit-learn training
    if args.model in ['lasso', 'rf', 'knn', 'tabnet']:
        # scikit-learn expects class_weights to be a dictionary
        class_weights = {}
        for i, val in enumerate(args.class_weights):
            # handle imbalanced datasets
            class_weights[i] = val
        class_weights_list = [class_weights[i] for i in range(len(class_weights))]

        if args.model == 'lasso':
            model = LogisticRegression(penalty='l1',
                                       C=args.lasso_C,
                                       class_weight=class_weights,
                                       max_iter=10000,
                                       random_state=args.seed_training,
                                       solver='saga',
                                       verbose=True)
            model.fit(data_module.X_train, data_module.y_train)
        elif args.model == 'rf':
            model = SelectFromModel(
                RandomForestClassifier(n_estimators=args.rf_n_estimators,
                                       min_samples_leaf=args.rf_min_samples_leaf,
                                       max_depth=args.rf_max_depth,
                                       class_weight=class_weights,
                                       max_features='sqrt',
                                       random_state=args.seed_training,
                                       verbose=True))
            model.fit(data_module.X_train, data_module.y_train)
        elif args.model == 'knn':
            model = KNeighborsClassifier(n_neighbors=args.knn_k)
            model.fit(data_module.X_train, data_module.y_train)
        elif args.model == 'lgb':
            params = {
                'max_depth': args.lgb_max_depth,
                'learning_rate': args.lgb_learning_rate,
                'min_data_in_leaf': args.lgb_min_data_in_leaf,
                'class_weight': class_weights,
                'n_estimators': 200,
                'objective': 'cross_entropy',
                'num_iterations': 10000,
                'device': 'gpu',
                'feature_fraction': '0.3'
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(data_module.X_train,
                      data_module.y_train,
                      eval_set=[(data_module.X_valid, data_module.y_valid)],
                      callbacks=[lgb.early_stopping(stopping_rounds=100)])
        elif args.model == 'tabnet':
            model = TabNetClassifier(
                n_d=8,
                n_a=
                8,  # The TabNet implementation says "Bigger values gives more capacity to the model with the risk of overfitting"
                n_steps=3,
                gamma=1.5,
                n_independent=2,
                n_shared=2,  # default values
                momentum=0.3,
                clip_value=2.,
                lambda_sparse=args.tabnet_lambda_sparse,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=args.lr),  # the paper sugests 2e-2
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={
                    "gamma": 0.95,
                    "step_size": 20
                },
                seed=args.seed_training)

            class WeightedCrossEntropy(Metric):

                def __init__(self):
                    self._name = "cross_entropy"
                    self._maximize = False

                def __call__(self, y_true, y_score):
                    aux = F.cross_entropy(input=torch.tensor(y_score, device='cuda'),
                                          target=torch.tensor(y_true, device='cuda'),
                                          weight=torch.tensor(args.class_weights,
                                                              device='cuda')).detach().cpu().numpy()

                    return float(aux)

            virtual_batch_size = 5
            if args.dataset == 'lung':
                virtual_batch_size = 6  # lung has training of size 141. With a virtual_batch_size of 5, the last batch is of size 1 and we get an error because of BatchNorm

            batch_size = args.train_size
            model.fit(data_module.X_train,
                      data_module.y_train,
                      eval_set=[(data_module.X_valid, data_module.y_valid)],
                      eval_metric=[WeightedCrossEntropy],
                      loss_fn=torch.nn.CrossEntropyLoss(torch.tensor(args.class_weights, device='cuda')),
                      batch_size=batch_size,
                      virtual_batch_size=virtual_batch_size,
                      max_epochs=5000,
                      patience=100)

        # Log metrics
        if args.model in ['rf']:
            y_pred_train = model.estimator_.predict(data_module.X_train)
            y_pred_valid = model.estimator_.predict(data_module.X_valid)
            y_pred_test = model.estimator_.predict(data_module.X_test)
        else:
            y_pred_train = model.predict(data_module.X_train)
            y_pred_valid = model.predict(data_module.X_valid)
            y_pred_test = model.predict(data_module.X_test)

        train_metrics = compute_all_metrics(args, data_module.y_train, y_pred_train)
        valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid)
        test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test)

        # example for zip() with lists: https://favtutor.com/blogs/zip-two-lists-python
        for metrics, dataset_name in zip([train_metrics, valid_metrics, test_metrics],
                                         ["bestmodel_train", "bestmodel_valid", "bestmodel_test"]):
            for metric_name, metric_value in metrics.items():
                wandb.run.summary[f"{dataset_name}/{metric_name}"] = metric_value

        # recored the feature selection
        if args.model in ['lasso', 'rf', 'tabnet']:
            if args.model == 'lasso':
                gate = (np.linalg.norm(model.coef_, ord=2, axis=0) != 0).astype(int)
                gate_dec_str = gate_bin2dec(gate)
                num_selected_features = gate.sum()
            elif args.model == 'rf':
                gate = np.asarray(model.get_support(), dtype=int)
                gate_dec_str = gate_bin2dec(gate)
                num_selected_features = gate.sum()
            elif args.model == 'tabnet':
                feature_importance, _ = model.explain(data_module.X_test)
                gate_all_mat = np.asarray((feature_importance != 0), dtype=int)
                num_selected_features = gate_all_mat.sum(axis=1).mean()
                gate_all = []
                for gate in gate_all_mat:
                    gate_dec_str = gate_bin2dec(gate)
                    gate_all.append(gate_dec_str)

            wandb.run.summary["{}/{}".format(dataset_name, 'num_selected_features')] \
                                                                             = num_selected_features
            if args.model in ['tabnet']:
                f1_score_gate = get_f1_score_gate(args, gate_all_mat)
                gate_all = wandb.Table(dataframe=pd.DataFrame(gate_all))
            else:
                f1_score_gate = get_f1_score_gate(args, gate)
                gate_all = wandb.Table(dataframe=pd.DataFrame([gate_dec_str]))
            wandb.log({f'gate_all': gate_all})
            if args.dataset in ['syn1', 'syn2', 'syn3']:
                wandb.log({f'bestmodel_test/micro_F1_score_of_predicted_feature_gates': f1_score_gate})

    # Pytorch lightning training
    else:
        # number of output units of the feature extractor
        # Used for convenience when defining the GP
        args.num_tasks = args.feature_extractor_dims[-1]

        if args.max_steps != -1:
            # compute the upper rounded number of epochs to training (used for lr scheduler in DKL)
            steps_per_epoch = max(np.floor(args.train_size / args.batch_size), 1)
            args.max_epochs = int(np.ceil(args.max_steps / steps_per_epoch))
            print(f"Training for max_epochs = {args.max_epochs}")

        # search for the best hyper-parameters
        if args.enable_optuna:
            pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner()
                                                 if args.pruning else optuna.pruners.NopPruner())

            study = optuna.create_study(direction="minimize", pruner=pruner)
            # make this wrapper a function
            objective = lambda trial: optuna_objective(args, data_module, trial)
            study.optimize(objective, n_trials=args.optuna_trial)

            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

        else:
            # Create model
            model = create_model(args, data_module)

            # train model
            trainer, checkpoint_callback = train_model(args, model, data_module, wandb_logger)

            if args.test_only:
                checkpoint_path = args.saved_checkpoint_name

                print(f'Evaluating trained model from {checkpoint_path}')
                wandb.log({"bestmodel/step": checkpoint_path.split("step=")[1].split('.ckpt')[0]})
            elif args.train_on_full_data:
                checkpoint_path = checkpoint_callback.last_model_path
            else:
                checkpoint_path = checkpoint_callback.best_model_path

                print(f"\n\nBest model saved on path {checkpoint_path}\n\n")
                wandb.log({"bestmodel/step": checkpoint_path.split("step=")[1].split('.ckpt')[0]})

            # same the path for checkpoint
            args.checkpoint_path = checkpoint_path

            # Compute metrics for the best model
            model.log_test_key = 'bestmodel_train'
            trainer.test(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)
            model.log_test_key = 'bestmodel_valid'
            trainer.test(model, dataloaders=data_module.val_dataloader()[0], ckpt_path=checkpoint_path)
            model.log_test_key = 'bestmodel_test'
            trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)

    wandb.finish()

    print("\nExiting from train function..")


def train_model(args, model, data_module, wandb_logger=None):
    """
    Return 
    - Pytorch Lightening Trainer
    - checkpoint callback
    """

    # Resume Training
    if args.saved_checkpoint_name:
        # wandb_artifact_path = f'{BASE_DIR}/{args.saved_checkpoint_name}'
        # print(f"\nDownloading artifact: {wandb_artifact_path}...")

        # artifact = wandb.use_artifact(wandb_artifact_path, type='model')
        # artifact_dir = artifact.download()
        # model_checkpoint = torch.load(os.path.join(artifact_dir, 'model.ckpt'))
        model_checkpoint = torch.load(args.saved_checkpoint_name)
        weights = model_checkpoint['state_dict']

        print("\nLoading pretrained weights into model from {}.".format(args.saved_checkpoint_name))
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        print(f"Missing keys: \n")
        print(missing_keys)

        print(f"Unexpected keys: \n")
        print(unexpected_keys)

    # set up training metric
    mode_metric = 'max' if args.metric_model_selection == 'balanced_accuracy' else 'min'
    checkpoint_callback = ModelCheckpoint(monitor=f'valid/{args.metric_model_selection}',
                                          mode=mode_metric,
                                          save_last=True,
                                          verbose=True)

    # add callback functions for training
    callbacks = [checkpoint_callback, RichProgressBar()]
    if args.patience_early_stopping and args.train_on_full_data == False:
        callbacks.append(
            EarlyStopping(
                monitor=f'valid/{args.metric_model_selection}',
                mode=mode_metric,
                patience=args.patience_early_stopping,
            ))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # set up trainer
    pl.seed_everything(args.seed_training, workers=True)
    trainer = pl.Trainer(
        # Training
        max_steps=args.max_steps,
        gradient_clip_val=2.5,

        # logging
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.val_check_interval,
        # val_check_interval=None,
        callbacks=callbacks,

        # miscellaneous
        accelerator=args.accelerator,
        devices="auto",
        detect_anomaly=args.debugging,
        overfit_batches=args.overfit_batches,
        deterministic=args.deterministic,

        # used for debugging
        # fast_dev_run=True,
    )

    model.trainer = trainer

    # train
    trainer.fit(model, data_module)

    return trainer, checkpoint_callback


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    """
    Available datasets
    - toxicity
    - lung
    - metabric-dr__200
    - metabric-pam50__200
    - tcga-2ysurvival__200
    - prostate
    - colon
    """

    ####### Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_size', type=int, help='100, 200, 330, 400, 800, 1600')
    parser.add_argument('--dataset_feature_set',
                        type=str,
                        choices=['hallmark', '8000', '16000'],
                        default='hallmark',
                        help='Note: implemented for Metabric only \
                            hallmark = 4160 common genes \
                            8000 = the 4160 common genes + 3840 random genes \
                            16000 = the 8000 genes above + 8000 random genes')

    ####### Model
    parser.add_argument('--model',
                        type=str,
                        choices=[
                            'knn',
                            'mlp',
                            'lasso',
                            'rf',
                            'tabnet',
                            'protogate',
                        ],
                        default='mlp')
    parser.add_argument(
        '--feature_extractor_dims',
        type=int,
        nargs='+',
        default=[100, 100, 10],  # use last dimnsion of 10 following the paper "Promises and perils of DKL" 
        help='layer size for the feature extractor. If using a virtual layer,\
                              the first dimension must match it.')
    parser.add_argument(
        '--layers_for_hidden_representation',
        type=int,
        default=2,
        help='number of layers after which to output the hidden representation used as input to the decoder \
                              (e.g., if the layers are [100, 100, 10] and layers_for_hidden_representation=2, \
                                  then the hidden representation will be the representation after the two layers [100, 100])'
    )

    parser.add_argument(
        '--batchnorm',
        type=int,
        default=1,
        help='if 1, then add batchnorm layers in the main network. If 0, then dont add batchnorm layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for the main network')
    parser.add_argument('--gamma',
                        type=float,
                        default=0,
                        help='The factor multiplied to the reconstruction error. \
                              If >0, then create a decoder with a reconstruction loss. \
                              If ==0, then dont create a decoder.')
    parser.add_argument('--saved_checkpoint_name',
                        type=str,
                        help='name of the wandb artifact name (e.g., model-1dmvja9n:v0)')
    parser.add_argument('--load_model_weights',
                        action='store_true',
                        dest='load_model_weights',
                        help='True if loading model weights')
    parser.set_defaults(load_model_weights=False)

    ####### Scikit-learn parameters
    parser.add_argument('--lasso_C', type=float, default=1e3, help='lasso regularization parameter')

    parser.add_argument('--rf_n_estimators', type=int, default=500, help='number of trees in the random forest')
    parser.add_argument('--rf_max_depth', type=int, default=5, help='maximum depth of the tree')
    parser.add_argument('--rf_min_samples_leaf', type=int, default=2, help='minimum number of samples in a leaf')

    parser.add_argument('--knn_k', type=int, default=9, help='number of neighbours in KNN')

    parser.add_argument('--tabnet_lambda_sparse',
                        type=float,
                        default=1e-3,
                        help='higher coefficient the sparser the feature selection')

    ####### Sparsity
    parser.add_argument('--sparsity_type',
                        type=str,
                        default=None,
                        choices=['global', 'local'],
                        help="Use global or local sparsity, but WPFS paper only reports the global results")
    parser.add_argument('--sparsity_method',
                        type=str,
                        default='sparsity_network',
                        choices=['learnable_vector', 'sparsity_network'],
                        help="The method to induce sparsity")
    parser.add_argument('--mixing_layer_size', type=int, help='size of the mixing layer in the sparsity network')
    parser.add_argument('--mixing_layer_dropout', type=float, help='dropout rate for the mixing layer')

    parser.add_argument('--sparsity_gene_embedding_type',
                        type=str,
                        default='nmf',
                        choices=['all_patients', 'nmf'],
                        help='It`s applied over data preprocessed using `embedding_preprocessing`')
    parser.add_argument('--sparsity_gene_embedding_size', type=int, default=50)
    parser.add_argument('--sparsity_regularizer', type=str, default='L1', choices=['L1', 'hoyer'])
    parser.add_argument('--sparsity_regularizer_hyperparam',
                        type=float,
                        default=0,
                        help='The weight of the sparsity regularizer (used to compute total_loss)')

    ####### DKL
    parser.add_argument('--grid_bound',
                        type=float,
                        default=5.,
                        help='The grid bound on the inducing points for the GP.')
    parser.add_argument('--grid_size', type=int, default=64, help='Dimension of the grid of inducing points')

    # MLP
    parser.add_argument('--mlp_activation',
                        type=str,
                        default='tanh',
                        choices=['relu', 'l_relu', 'sigmoid', 'tanh', 'none'],
                        help='activation in the MLP layers')
    parser.add_argument('--mlp_hidden_layer_list',
                        type=int,
                        nargs='+',
                        default=[100, 100, 10],
                        help='Number of hidden neurons within MLP')

    # ProtoGate
    parser.add_argument('--feature_selection',
                        action='store_false',
                        dest='feature_selection',
                        help='Whether to enable feature selection in ProtoGate.')
    parser.set_defaults(feature_selection=True)
    parser.add_argument('--protogate_lam_global',
                        type=float,
                        default=2e-4,
                        help='controls the sparsity of the first layer in gating network')
    parser.add_argument('--protogate_lam_local', type=float, default=1e-3, help='controls the local sparsity')
    parser.add_argument('--protogate_gating_hidden_layer_list',
                        type=int,
                        nargs='+',
                        default=[200],
                        help='Number of hidden neurons within the gating network.')
    parser.add_argument('--protogate_a', type=float, default=1)
    parser.add_argument('--protogate_sigma', type=float, default=0.5)
    parser.add_argument('--protogate_init_std', type=float, default=0.015)
    parser.add_argument('--protogate_activation_gating',
                        type=str,
                        default='tanh',
                        choices=['relu', 'l_relu', 'sigmoid', 'tanh', 'none'],
                        help='choose tanh to ensure alpha values in (-1, 1)')
    parser.add_argument("--pred_k", type=int, default=3, help='numer of neighbours for prediction')
    parser.add_argument('--pred_coef', type=float, default=1, help='coefficient for pred loss term.')
    parser.add_argument('--sorting_tau',
                        type=float,
                        default=16.,
                        help='temperature of sorting operator (>0), the smaller the harder permutation')

    ####### Training
    parser.add_argument('--concrete_anneal_iterations',
                        type=int,
                        default=1000,
                        help='number of iterations for annealing the Concrete radnom variables (in CAE and FsNet)')

    parser.add_argument('--max_steps', type=int, default=10000, help='Specify the max number of steps to train.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--full_batch_training', action='store_true', dest='full_batch_training')
    parser.set_defaults(full_batch_training=False)

    parser.add_argument(
        '--patient_preprocessing',
        type=str,
        default='standard',
        choices=['raw', 'standard', 'minmax'],
        help=
        'Preprocessing applied on each COLUMN of the N x D matrix, where a row contains all gene expressions of a patient.'
    )
    parser.add_argument(
        '--embedding_preprocessing',
        type=str,
        default='minmax',
        choices=['raw', 'standard', 'minmax'],
        help=
        'Preprocessing applied on each ROW of the D x N matrix, where a row contains all patient expressions for one gene.'
    )

    ####### Training on the entire train + validation data
    parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
         help='Train on the full data (train + validation), leaving only `--test_split` for testing.')
    parser.set_defaults(train_on_full_data=False)
    parser.add_argument('--path_steps_on_full_data',
                        type=str,
                        default=None,
                        help='Path to the file which holds the number of steps to train.')

    ####### Validation
    parser.add_argument('--metric_model_selection',
                        type=str,
                        default='cross_entropy_loss',
                        choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy', 'mse_loss'])

    parser.add_argument('--patience_early_stopping',
                        type=int,
                        default=500,
                        help='Set number of checks (set by *val_check_interval*) to do early stopping.\
                             It will train for at least   args.val_check_interval * args.patience_early_stopping epochs'
                        )
    parser.add_argument('--log_every_n_steps',
                        type=int,
                        default=500,
                        help='number of steps at which to display the training process')
    parser.add_argument('--val_check_interval',
                        type=int,
                        default=50,
                        help='number of steps at which to check the validation')

    # type of data augmentation
    parser.add_argument('--valid_aug_dropout_p',
                        type=float,
                        nargs="+",
                        help="List of dropout data augmentation for the validation data loader.\
                              A new validation dataloader is created for each value.\
                              E.g., (1, 10) creates a dataloader with valid_aug_dropout_p=1, valid_aug_dropout_p=10\
                              in addition to the standard validation")
    parser.add_argument('--valid_aug_times',
                        type=int,
                        nargs="+",
                        help="Number time to perform data augmentation on the validation sample.")

    ####### Testing
    parser.add_argument('--testing_type',
                        type=str,
                        default='cross-validation',
                        choices=['cross-validation', 'fixed'],
                        help='`cross-validation` performs testing on the testing splits \
                              `fixed` performs testing on an external testing set supplied in a dedicated file')
    parser.add_argument('--test_only',
                        action='store_true',
                        dest='test_only',
                        help='Load a trained model for evaluation.')
    parser.set_defaults(test_only=False)

    ####### Cross-validation
    parser.add_argument('--repeat_id',
                        type=int,
                        default=0,
                        help='each repeat_id gives a different random seed for shuffling the dataset')
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of CV splits")
    parser.add_argument('--test_split',
                        type=int,
                        default=1,
                        help="Index of the valid/test split. It should be smaller than `cv_folds`")
    parser.add_argument('--valid_percentage',
                        type=float,
                        default=0.1,
                        help='Percentage of training data used for validation')

    ####### Evaluation by taking random samples (with user-defined train/valid/test sizes) from the dataset
    parser.add_argument('--evaluate_with_sampled_datasets', action='store_true', dest='evaluate_with_sampled_datasets')
    parser.set_defaults(evaluate_with_sampled_datasets=False)
    parser.add_argument('--custom_train_size', type=int, default=None)
    parser.add_argument('--custom_valid_size', type=int, default=None)
    parser.add_argument('--custom_test_size', type=int, default=None)

    ####### Optimization
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='sgd')
    parser.add_argument('--lr_scheduler',
                        type=str,
                        choices=['plateau', 'cosine_warm_restart', 'linear', 'lambda', 'none'],
                        default='none')
    parser.add_argument('--cosine_warm_restart_eta_min', type=float, default=1e-6)
    parser.add_argument('--cosine_warm_restart_t_0', type=int, default=35)
    parser.add_argument('--cosine_warm_restart_t_mult', type=float, default=1)

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lookahead_optimizer', type=int, default=0, help='Use Lookahead optimizer.')
    parser.add_argument('--class_weight',
                        type=str,
                        choices=['standard', 'balanced'],
                        default='balanced',
                        help="If `standard`, all classes use a weight of 1.\
                              If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)"
                        )

    parser.add_argument('--debugging', action='store_true', dest='debugging')
    parser.set_defaults(debugging=False)
    parser.add_argument('--deterministic', action='store_true', dest='deterministic')
    parser.set_defaults(deterministic=False)

    ####### Others
    parser.add_argument(
        '--overfit_batches',
        type=float,
        default=0,
        help="0 --> normal training. <1 --> overfit on % of the training data. >1 overfit on this many batches")
    parser.add_argument('--accelerator', type=str, default='auto', help='type of accelerator for pytorch lightning')

    # SEEDS
    parser.add_argument('--seed_global', type=int, default=42)
    parser.add_argument('--seed_model_init',
                        type=int,
                        default=42,
                        help='Seed for initializing the model (to have the same weights)')
    parser.add_argument('--seed_training', type=int, default=42, help='Seed for training (e.g., batch ordering)')

    parser.add_argument('--seed_kfold', type=int, help='Seed used for doing the kfold in train/test split')
    parser.add_argument('--seed_validation', type=int, help='Seed used for selecting the validation split.')

    # Dataset loading
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers for loading dataset")
    parser.add_argument('--no_pin_memory',
                        dest='pin_memory',
                        action='store_false',
                        help='dont pin memory for data loaders')
    parser.set_defaults(pin_memory=True)

    ####### Wandb logging
    parser.add_argument('--group', type=str, help="Group runs in wand")
    parser.add_argument('--job_type', type=str, help="Job type for wand")
    parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
    parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
    parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
    parser.add_argument('--wandb_log_model',
                        action='store_true',
                        dest='wandb_log_model',
                        help='True for storing the model checkpoints in wandb')
    parser.set_defaults(wandb_log_model=False)
    parser.add_argument('--disable_wandb',
                        action='store_true',
                        dest='disable_wandb',
                        help='True if you dont want to crete wandb logs.')
    parser.set_defaults(disable_wandb=False)

    # optuna
    parser.add_argument('--enable_optuna', action='store_true', dest='enable_optuna')
    parser.set_defaults(enable_optuna=False)
    parser.add_argument('--optuna_trial',
                        type=int,
                        default=20,
                        help='Numer of trials to find the optimal hyperparameters.')
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.",
    )
    parser.set_defaults(pruning=False)
    parser.add_argument('--metric_param_selection',
                        type=str,
                        default='cross_entropy_loss',
                        choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy', 'mse_loss'])

    return parser.parse_args(args)


if __name__ == "__main__":
    # ignore warnings
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.LightningDeprecationWarning)

    # launch training
    print("Starting...")

    # set up script logger at DEBUG level
    logging.basicConfig(filename=f'{BASE_DIR}/logs_exceptions.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # obtain configuration arguments
    args = parse_arguments()

    # set gloabl random seed
    seed_everything(args)

    # When training without particular validation set, you need to set the step to stop explicitly.
    if args.train_on_full_data and args.model in ['dnn', 'dietdnn']:
        assert args.path_steps_on_full_data

        # retrieve the number of steps to train
        aux = pd.read_csv(args.path_steps_on_full_data, index_col=0)
        conditions = {
            'dataset': args.dataset,
            'model': args.model,
            'sparsity_regularizer_hyperparam': args.sparsity_regularizer_hyperparam,
        }
        temp = aux.loc[(aux[list(conditions)] == pd.Series(conditions)).all(axis=1)].copy()
        assert temp.shape[0] == 1

        args.max_steps = int(temp['median'].values[0])

    # set seeds
    args.seed_kfold = args.repeat_id
    args.seed_validation = args.test_split

    # `val_check_interval` must be less than or equal to the number of the training batches
    # if args.dataset == 'prostate' or args.dataset == 'cll':
    #     args.val_check_interval = 4

    # for all datasets, `custom_train_size` to randomly sample datasets
    if "__" in args.dataset:
        """
        when args.dataset=="metabric-dr__200" split into
        args.dataset = "metabric-dr"
        args.dataset_size = 200
        """
        args.dataset, args.dataset_size = args.dataset.split("__")
        args.dataset_size = int(args.dataset_size)

    # Assert that the dataset is supported
    SUPPORTED_DATASETS = [
        # real-world datasets
        'metabric-pam50',
        # synthesized datasets
        'syn1',
        'syn2',
        'syn3',
    ]
    if args.dataset not in SUPPORTED_DATASETS:
        raise Exception(f"Dataset {args.dataset} not supported. Supported datasets are {SUPPORTED_DATASETS}")

    # Assert custom evaluation with repeated dataset sampling
    if args.evaluate_with_sampled_datasets or args.custom_train_size or args.custom_valid_size or args.custom_test_size:
        assert args.evaluate_with_sampled_datasets
        assert args.custom_train_size
        assert args.custom_test_size
        assert args.custom_valid_size

    # Assert sparsity parameters
    if args.sparsity_type:
        # if one of the sparsity parameters is set, then all of them must be set
        assert args.sparsity_gene_embedding_type
        assert args.sparsity_type
        assert args.sparsity_method
        assert args.sparsity_regularizer
        # assert args.sparsity_regularizer_hyperparam

    # disable wandb when debugging
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    run_experiment(args)
