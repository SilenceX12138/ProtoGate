from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset import *
from models import create_model


def optuna_objective(args, data_module, trial: optuna.trial.Trial) -> float:
    if args.model in ['mlp']:
        args.lr = trial.suggest_float('lr', 1e-3, 1e-1)
        hyperparameters = {'lr': args.lr}

    # initialise model after params are set
    model = create_model(args, data_module)

    return param_tune_experiment(args, data_module, model, hyperparameters)


def param_tune_experiment(args, data_module, model, hyperparameters: dict) -> float:
    # set up training metric
    mode_metric = 'min' if args.metric_model_selection != 'balanced_accuracy' else 'max'
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

    # build dir to save log
    os.makedirs(os.path.join('tune_log'), exist_ok=True)
    os.makedirs(os.path.join('tune_log', args.model), exist_ok=True)
    os.makedirs(os.path.join('tune_log', args.model, args.dataset), exist_ok=True)

    # set up trainer
    pl.seed_everything(args.seed_training, workers=True)
    trainer = pl.Trainer(
        # Training
        max_steps=args.max_steps,
        gradient_clip_val=2.5,

        # logging
        logger=True,
        default_root_dir=os.path.join('tune_log', args.model, args.dataset),
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.val_check_interval,
        callbacks=callbacks,

        # miscellaneous
        accelerator=args.accelerator,
        devices="auto",
        detect_anomaly=args.debugging,
        overfit_batches=args.overfit_batches,
        deterministic=args.deterministic,

        # used for debugging
        fast_dev_run=args.debugging,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    model.trainer = trainer

    # train model
    trainer.fit(model, data_module)

    # retrive the best model
    checkpoint_path = checkpoint_callback.best_model_path
    print(f"\n\nBest model saved on path {checkpoint_path}\n\n")

    # Compute metrics for the best model
    model.log_test_key = 'bestmodel_train'
    trainer.test(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)

    model.log_test_key = 'bestmodel_valid'
    trainer.test(model, dataloaders=data_module.val_dataloader()[0], ckpt_path=checkpoint_path)
    # select the "best" (param_selection) hyper-parameters by evaluating the "another best" (model_seletion) model on the validation set
    valid_metric_value = trainer.logged_metrics[f"bestmodel_valid/{args.metric_param_selection}"].item()

    model.log_test_key = 'bestmodel_test'
    trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)

    return valid_metric_value
