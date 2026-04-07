import argparse
import json
import os

import pytorch_lightning as pl
import torch as T

from src.pag import PAGPipeline, load_binary_table, make_synthetic_binary_table
from src.pag.data import binary_table_probability_table, split_binary_table


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PAG likelihood model on binary tabular data.")
    parser.add_argument('name', help="name of the experiment")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--data-path', help="path to a binary table in csv, npy, npz, pt, or pth format")
    source_group.add_argument('--synthetic-rows', type=int, default=10000, help="number of synthetic rows to generate")

    parser.add_argument('--graph', default='chain', help="PAG structure to use; only 'chain' is supported right now")
    parser.add_argument('--trial', default='1',
                        help="either a trial count (e.g. 3 -> runs 0,1,2) or a comma-separated list of trial indices")
    parser.add_argument('--rerun', '-r', type=int, default=1,
                        help="number of independent reruns to execute for each trial")
    parser.add_argument('--num-vars', type=int, default=3, help="number of variables for synthetic data")
    parser.add_argument('--synthetic-seed', type=int, default=0, help="seed for synthetic data generation")
    parser.add_argument('--val-frac', type=float, default=0.1, help="validation split fraction")
    parser.add_argument('--validation-sample-count', type=int, default=10000,
                        help="number of rows to sample for validation-time probability tables")
    parser.add_argument('--validation-depth', type=int, default=3,
                        help="number of same-U refinement steps during validation sampling")
    parser.add_argument('--print-prob-table-every', type=int, default=1,
                        help="print sampled probability tables every N validation epochs; use 0 to disable")

    parser.add_argument('--batch-size', type=int, default=256, help="training batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--post-freeze-lr', type=float,
                        help="learning rate to switch to immediately after DAG freezing; defaults to lr / 10")
    parser.add_argument('--post-freeze-plateau-factor', type=float, default=0.5,
                        help="ReduceLROnPlateau factor applied to post-freeze lr when post-freeze KL stalls")
    parser.add_argument('--post-freeze-plateau-patience', type=int, default=10,
                        help="number of post-freeze validation epochs with no KL improvement before reducing lr")
    parser.add_argument('--post-freeze-min-lr', type=float, default=1e-6,
                        help="minimum lr allowed for the post-freeze plateau scheduler")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--mc-sample-size', type=int, default=128, help="number of latent samples per row")
    parser.add_argument('--latent-dim', type=int, default=8, help="dimensionality of each latent token")
    parser.add_argument('--embed-dim', type=int, default=64, help="token embedding dimension")
    parser.add_argument('--num-heads', type=int, default=2, help="number of attention heads")
    parser.add_argument('--num-layers', type=int, default=1, help="number of attention blocks")
    parser.add_argument('--ffn-hidden-dim', type=int, default=128, help="hidden dimension in the transformer FFN")
    parser.add_argument('--latent-mlp-hidden-dim', type=int, default=128, help="hidden size of each phi_u MLP")
    parser.add_argument('--latent-mlp-layers', type=int, default=2, help="number of layers in each phi_u MLP")
    parser.add_argument('--head-hidden-dim', type=int, default=64, help="hidden size of each output head")
    parser.add_argument('--mask-structure', default='learnable', choices=['learnable', 'fixed-chain'],
                        help="use either a learnable adjacency mask or a fixed X<-Y->Z constraint mask")
    parser.add_argument('--dagma-s', type=float, default=2.0,
                        help="positive scalar used in the log-determinant DAG penalty")
    parser.add_argument('--mask-init-logit', type=float, default=0,
                        help="initial raw adjacency logit used before the sigmoid edge map is applied")
    parser.add_argument('--mask-init-std', type=float, default=1.0,
                        help="standard deviation of Gaussian noise added to the initial adjacency logits")
    parser.add_argument('--adjacency-normalization', default='none', choices=['none', 'sum'],
                        help="optional normalization for W; 'sum' divides the adjacency by the sum of all entries")
    parser.add_argument('--gate-floor', type=float, default=0.05,
                        help="delta in g = delta + (1 - delta) * sigmoid(a) for post-softmax graph gating")
    parser.add_argument('--gate-renorm-eps', type=float, default=1e-6,
                        help="epsilon used when renormalizing gated attention rows")
    parser.add_argument('--lambda-dag', type=float, default=3,
                        help="coefficient for the DAG penalty")
    parser.add_argument('--lambda-l1', type=float, default=0.0,
                        help="coefficient for the observed-adjacency L1 penalty")
    parser.add_argument('--freeze-dag-when-stable', action='store_true',
                        help="freeze the learned DAG once adjacency changes are small and dag_penalty is below threshold")
    parser.add_argument('--soft-reset-optimizer-when-stable', action='store_true',
                        help="when the learnable graph becomes stable, reset optimizer state but keep the graph learnable")
    parser.add_argument('--soft-reset-max-count', type=int, default=0,
                        help="maximum number of optimizer-state soft resets allowed while the graph stays learnable")
    parser.add_argument('--freeze-dag-min-epochs', type=int, default=20,
                        help="minimum epoch count before DAG freezing is allowed")
    parser.add_argument('--freeze-dag-window', type=int, default=5,
                        help="number of recent epochs used to judge adjacency stability")
    parser.add_argument('--freeze-dag-adj-tol', type=float, default=1e-3,
                        help="maximum allowed entrywise adjacency drift across the stability window")
    parser.add_argument('--freeze-dag-penalty-threshold', type=float, default=1e-4,
                        help="freeze only when dag_penalty is at or below this threshold")
    parser.add_argument('--latent-prior', default='normal', choices=['normal', 'uniform'],
                        help="latent prior used for Monte Carlo marginalization")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout applied inside attention blocks")
    parser.add_argument('--no-residual', action='store_true',
                        help="disable residual connections inside attention blocks")
    parser.add_argument('--print-loss-every', type=int, default=10,
                        help="print train loss and mean log-prob every N batches; use 0 to disable")

    parser.add_argument('--max-epochs', type=int, default=600, help="maximum number of training epochs")
    parser.add_argument('--patience', type=int, default=100, help="early stopping patience")
    parser.add_argument('--gradient-clip-val', type=float, default=5.0,
                        help="global gradient clipping value; use 0 to disable clipping")
    parser.add_argument('--gradient-clip-algorithm', default='norm', choices=['norm', 'value'],
                        help="gradient clipping algorithm passed to the trainer")
    parser.add_argument('--gpu', type=int, help="GPU id to use")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--num-workers', type=int, default=0, help="reserved for future data-loading extensions")
    parser.add_argument('--fast-dev-run', action='store_true', help="run one training and validation batch")
    parser.add_argument('--verbose', action='store_true', help="print dataset and model details")
    return parser.parse_args()


def build_hyperparams(args):
    return {
        'batch-size': args.batch_size,
        'graph': args.graph,
        'lr': args.lr,
        'post-freeze-lr': args.post_freeze_lr,
        'post-freeze-plateau-factor': args.post_freeze_plateau_factor,
        'post-freeze-plateau-patience': args.post_freeze_plateau_patience,
        'post-freeze-min-lr': args.post_freeze_min_lr,
        'weight-decay': args.weight_decay,
        'mc-sample-size': args.mc_sample_size,
        'latent-dim': args.latent_dim,
        'embed-dim': args.embed_dim,
        'num-heads': args.num_heads,
        'num-layers': args.num_layers,
        'ffn-hidden-dim': args.ffn_hidden_dim,
        'latent-mlp-hidden-dim': args.latent_mlp_hidden_dim,
        'latent-mlp-layers': args.latent_mlp_layers,
        'head-hidden-dim': args.head_hidden_dim,
        'mask-structure': args.mask_structure,
        'dagma-s': args.dagma_s,
        'mask-init-logit': args.mask_init_logit,
        'mask-init-std': args.mask_init_std,
        'adjacency-normalization': args.adjacency_normalization,
        'gate-floor': args.gate_floor,
        'gate-renorm-eps': args.gate_renorm_eps,
        'lambda-dag': args.lambda_dag,
        'lambda-l1': args.lambda_l1,
        'freeze-dag-when-stable': args.freeze_dag_when_stable,
        'soft-reset-optimizer-when-stable': args.soft_reset_optimizer_when_stable,
        'soft-reset-max-count': args.soft_reset_max_count,
        'freeze-dag-min-epochs': args.freeze_dag_min_epochs,
        'freeze-dag-window': args.freeze_dag_window,
        'freeze-dag-adj-tol': args.freeze_dag_adj_tol,
        'freeze-dag-penalty-threshold': args.freeze_dag_penalty_threshold,
        'latent-prior': args.latent_prior,
        'dropout': args.dropout,
        'residual': not args.no_residual,
        'print-loss-every': args.print_loss_every,
        'validation-sample-count': args.validation_sample_count,
        'validation-depth': args.validation_depth,
        'print-prob-table-every': args.print_prob_table_every,
        'num-workers': args.num_workers,
        'gradient-clip-val': args.gradient_clip_val,
        'gradient-clip-algorithm': args.gradient_clip_algorithm,
        'rerun': args.rerun,
    }


def get_trial_indices(args):
    trial_arg = str(args.trial).strip()
    normalized_arg = trial_arg.strip('[]').strip()
    if ',' in normalized_arg:
        parts = [part.strip() for part in normalized_arg.split(',') if part.strip()]
        if not parts:
            raise ValueError("--trial list must contain at least one index.")
        indices = [int(part) for part in parts]
        if any(idx < 0 for idx in indices):
            raise ValueError("--trial list indices must be non-negative.")
        return indices

    if trial_arg.startswith('[') and trial_arg.endswith(']'):
        if not normalized_arg:
            raise ValueError("--trial list must contain at least one index.")
        trial_index = int(normalized_arg)
        if trial_index < 0:
            raise ValueError("--trial list indices must be non-negative.")
        return [trial_index]

    trial_count = int(normalized_arg)
    if trial_count < 1:
        raise ValueError("--trial must be at least 1.")
    return list(range(trial_count))


def get_run_seed(args, trial_idx, rerun_idx):
    return args.seed + trial_idx * args.rerun + rerun_idx


def get_data_seed(args, trial_idx):
    return args.synthetic_seed + trial_idx


def get_run_name(args, trial_idx, rerun_idx):
    residual_tag = 'noresidual' if args.no_residual else 'residual'
    post_freeze_lr = args.post_freeze_lr if args.post_freeze_lr is not None else args.lr / 10.0
    return '{}-{}-r={}-layers={}-mask-structure={}-dagma_s={}-lambda_dag={}-lambda_l1={}-clip_val={}-mask_init={}-gate_floor={}-post_freeze_lr={}-plateau_factor={}-plateau_patience={}-soft_reset_max={}-{}'.format(
        args.graph,
        trial_idx,
        rerun_idx,
        args.num_layers,
        args.mask_structure,
        args.dagma_s,
        args.lambda_dag,
        args.lambda_l1,
        args.gradient_clip_val,
        args.mask_init_logit,
        args.gate_floor,
        post_freeze_lr,
        args.post_freeze_plateau_factor,
        args.post_freeze_plateau_patience,
        args.soft_reset_max_count,
        residual_tag,
    )


def merge_experiment_reports(exp_dir):
    merged_runs = []
    if not os.path.isdir(exp_dir):
        return merged_runs

    for run_name in sorted(os.listdir(exp_dir)):
        run_dir = os.path.join(exp_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        results_path = os.path.join(run_dir, 'results.json')
        hyperparams_path = os.path.join(run_dir, 'hyperparams.json')
        best_kl_path = os.path.join(run_dir, 'best_kl_summary.json')

        if not os.path.exists(results_path):
            continue

        with open(results_path, 'r') as fp:
            results = json.load(fp)

        hyperparams = None
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as fp:
                hyperparams = json.load(fp)

        best_kl_summary = None
        if os.path.exists(best_kl_path):
            with open(best_kl_path, 'r') as fp:
                best_kl_summary = json.load(fp)

        merged_runs.append({
            'run_name': run_name,
            'trial': results.get('trial'),
            'rerun': results.get('rerun'),
            'results': results,
            'hyperparams': hyperparams,
            'best_kl_summary': best_kl_summary,
        })

    with open(os.path.join(exp_dir, 'merged_metrics.json'), 'w') as fp:
        json.dump({'runs': merged_runs}, fp, indent=2)

    return merged_runs


def load_dataset(args, trial_idx):
    if args.graph != 'chain':
        raise ValueError("Only graph='chain' is supported right now.")
    if args.validation_depth != 3:
        raise ValueError("The chain PAG validation sampler requires --validation-depth 3.")
    if args.data_path is not None:
        return load_binary_table(args.data_path)
    return make_synthetic_binary_table(args.synthetic_rows, args.graph, seed=get_data_seed(args, trial_idx))


def make_trainer(output_dir, monitor, max_epochs, patience, gpu, fast_dev_run,
                 gradient_clip_val, gradient_clip_algorithm):
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='best',
        monitor=monitor,
        mode='min',
        save_top_k=1,
    )
    callbacks = [checkpoint]

    if not fast_dev_run:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='min',
            check_on_train_epoch_end=(monitor == 'train_loss'),
        ))

    trainer_kwargs = {
        'callbacks': callbacks,
        'default_root_dir': output_dir,
        'logger': pl.loggers.TensorBoardLogger(os.path.join(output_dir, 'logs')),
        'log_every_n_steps': 1,
        'max_epochs': max_epochs,
        'fast_dev_run': fast_dev_run,
    }
    if gradient_clip_val > 0:
        trainer_kwargs['gradient_clip_val'] = gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = gradient_clip_algorithm
    if gpu is not None:
        trainer_kwargs['gpus'] = [gpu]

    return pl.Trainer(**trainer_kwargs), checkpoint


def run_trial(args, trial_idx, rerun_idx, trial_position, total_trials):
    run_seed = get_run_seed(args, trial_idx, rerun_idx)
    data_seed = get_data_seed(args, trial_idx)
    pl.seed_everything(run_seed)

    data, variable_names = load_dataset(args, trial_idx)
    train_data, val_data = split_binary_table(data, val_fraction=args.val_frac, seed=run_seed)
    ground_truth_prob_table = binary_table_probability_table(data, variable_names)

    output_dir = os.path.join('out', args.name, get_run_name(args, trial_idx, rerun_idx))
    os.makedirs(output_dir, exist_ok=True)
    T.save({
        'data': data.cpu(),
        'train_data': train_data.cpu(),
        'val_data': None if val_data is None else val_data.cpu(),
        'columns': variable_names,
        'graph': args.graph,
        'trial': int(trial_idx),
        'rerun': int(rerun_idx),
        'seed': int(run_seed),
        'data_seed': int(data_seed),
    }, os.path.join(output_dir, 'data.pt'))

    hyperparams = build_hyperparams(args)
    hyperparams['trial'] = int(trial_idx)
    hyperparams['rerun_index'] = int(rerun_idx)
    hyperparams['num-trials'] = int(total_trials)
    model = PAGPipeline(
        train_data=train_data,
        val_data=val_data,
        variable_names=variable_names,
        ground_truth_prob_table=ground_truth_prob_table,
        hyperparams=hyperparams,
    )

    monitor = 'val_sample_kl' if val_data is not None else 'train_loss'
    trainer, checkpoint = make_trainer(
        output_dir=output_dir,
        monitor=monitor,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpu=args.gpu,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    print("=== Trial index {} rerun {} ({}/{}) ===".format(
        trial_idx, rerun_idx, trial_position + 1, total_trials * args.rerun))
    if args.verbose:
        print("Training rows:", train_data.shape[0])
        print("Validation rows:", 0 if val_data is None else val_data.shape[0])
        print("Variables:", variable_names)
        print("Run seed:", run_seed)
        print("Data seed:", data_seed)
    print("Ground truth probability table:\n{}".format(ground_truth_prob_table.to_string(index=False)))

    trainer.fit(model)

    best_path = checkpoint.best_model_path
    if best_path:
        state = T.load(best_path, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print("Initial adjacency matrix:\n{}".format(model.initial_adjacency))
        print("Loaded best-model adjacency matrix:\n{}".format(model.model.get_observed_adjacency().detach().cpu()))

    best_val_metrics = None
    if val_data is not None:
        saved_best_kl = model.best_val_sample_kl
        saved_best_kl_epoch = model.best_val_sample_epoch
        saved_best_sampled_table = None if model.best_sampled_table is None else model.best_sampled_table.copy()
        saved_best_adjacency = None if model.best_adjacency is None else model.best_adjacency.clone()

        validate_outputs = trainer.validate(model=model, dataloaders=model.val_dataloader(), verbose=False)
        if validate_outputs:
            best_val_metrics = validate_outputs[0]

        model.best_val_sample_kl = saved_best_kl
        model.best_val_sample_epoch = saved_best_kl_epoch
        model.best_sampled_table = saved_best_sampled_table
        model.best_adjacency = saved_best_adjacency

    train_loss = trainer.callback_metrics.get('train_loss')
    val_loss = trainer.callback_metrics.get('val_loss')
    results = {
        'train_rows': int(train_data.shape[0]),
        'val_rows': 0 if val_data is None else int(val_data.shape[0]),
        'graph': args.graph,
        'trial': int(trial_idx),
        'rerun': int(rerun_idx),
        'num_reruns': int(args.rerun),
        'num_trials': int(total_trials),
        'seed': int(run_seed),
        'data_seed': int(data_seed),
        'num_variables': int(train_data.shape[1]),
        'variable_names': variable_names,
        'ground_truth_prob_table': ground_truth_prob_table.to_dict(orient='records'),
        'initial_adjacency_matrix': model.initial_adjacency.tolist(),
        'best_checkpoint': best_path,
        'loaded_best_adjacency_matrix': model.model.get_observed_adjacency().detach().cpu().tolist(),
        'final_train_loss': None if train_loss is None else float(train_loss.detach().cpu()),
        'final_val_loss': None if val_loss is None else float(val_loss.detach().cpu()),
        'final_val_sample_kl': None if trainer.callback_metrics.get('val_sample_kl') is None else float(
            trainer.callback_metrics['val_sample_kl'].detach().cpu()),
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as fp:
        json.dump(hyperparams, fp, indent=2)

    best_kl_summary = model.get_best_kl_summary()
    if best_kl_summary is not None:
        if best_val_metrics is not None:
            best_kl_summary['best_val_loss'] = float(best_val_metrics['val_loss'])
            best_kl_summary['best_val_nll'] = float(best_val_metrics['val_nll'])
            best_kl_summary['best_val_dag_penalty'] = float(best_val_metrics['val_dag_penalty'])
        with open(os.path.join(output_dir, 'best_kl_summary.json'), 'w') as fp:
            json.dump(best_kl_summary, fp, indent=2)


def main():
    args = parse_args()
    if args.rerun < 1:
        raise ValueError("--rerun must be at least 1.")
    trial_indices = get_trial_indices(args)
    total_trials = len(trial_indices)
    for trial_position, trial_idx in enumerate(trial_indices):
        for rerun_idx in range(args.rerun):
            run_trial(args, trial_idx, rerun_idx, trial_position * args.rerun + rerun_idx, total_trials)

    merge_experiment_reports(os.path.join('out', args.name))


if __name__ == '__main__':
    main()
