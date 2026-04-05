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
    parser.add_argument('--trial', '--trail', type=int, default=0,
                        help="trial index used to derive the run seed and output directory")
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
    parser.add_argument('--mask-mode', default='additive', choices=['additive', 'multiplicative'],
                        help="how the learnable attention mask is applied")
    parser.add_argument('--latent-prior', default='normal', choices=['normal', 'uniform'],
                        help="latent prior used for Monte Carlo marginalization")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout applied inside attention blocks")
    parser.add_argument('--no-residual', action='store_true',
                        help="disable residual connections inside attention blocks")
    parser.add_argument('--print-loss-every', type=int, default=10,
                        help="print train loss and mean log-prob every N batches; use 0 to disable")

    parser.add_argument('--max-epochs', type=int, default=200, help="maximum number of training epochs")
    parser.add_argument('--patience', type=int, default=50, help="early stopping patience")
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
        'trial': args.trial,
        'lr': args.lr,
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
        'mask-mode': args.mask_mode,
        'latent-prior': args.latent_prior,
        'dropout': args.dropout,
        'residual': not args.no_residual,
        'print-loss-every': args.print_loss_every,
        'validation-sample-count': args.validation_sample_count,
        'validation-depth': args.validation_depth,
        'print-prob-table-every': args.print_prob_table_every,
        'num-workers': args.num_workers,
    }


def get_run_seed(args):
    return args.seed + args.trial


def get_data_seed(args):
    return args.synthetic_seed + args.trial


def get_run_name(args):
    residual_tag = 'noresidual' if args.no_residual else 'residual'
    return '{}-{}-layers={}-{}'.format(args.graph, args.trial, args.num_layers, residual_tag)


def load_dataset(args):
    if args.graph != 'chain':
        raise ValueError("Only graph='chain' is supported right now.")
    if args.validation_depth != 3:
        raise ValueError("The chain PAG validation sampler requires --validation-depth 3.")
    if args.data_path is not None:
        return load_binary_table(args.data_path)
    return make_synthetic_binary_table(args.synthetic_rows, args.graph, seed=get_data_seed(args))


def make_trainer(output_dir, monitor, max_epochs, patience, gpu, fast_dev_run):
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
    if gpu is not None:
        trainer_kwargs['gpus'] = [gpu]

    return pl.Trainer(**trainer_kwargs), checkpoint


def main():
    args = parse_args()
    run_seed = get_run_seed(args)
    pl.seed_everything(run_seed)

    data, variable_names = load_dataset(args)
    train_data, val_data = split_binary_table(data, val_fraction=args.val_frac, seed=run_seed)
    ground_truth_prob_table = binary_table_probability_table(data, variable_names)

    output_dir = os.path.join('out', args.name, get_run_name(args))
    os.makedirs(output_dir, exist_ok=True)
    T.save({
        'data': data.cpu(),
        'train_data': train_data.cpu(),
        'val_data': None if val_data is None else val_data.cpu(),
        'columns': variable_names,
        'graph': args.graph,
        'trial': int(args.trial),
        'seed': int(run_seed),
        'data_seed': int(get_data_seed(args)),
    }, os.path.join(output_dir, 'data.pt'))

    hyperparams = build_hyperparams(args)
    model = PAGPipeline(
        train_data=train_data,
        val_data=val_data,
        variable_names=variable_names,
        ground_truth_prob_table=ground_truth_prob_table,
        hyperparams=hyperparams,
    )

    monitor = 'val_loss' if val_data is not None else 'train_loss'
    trainer, checkpoint = make_trainer(
        output_dir=output_dir,
        monitor=monitor,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpu=args.gpu,
        fast_dev_run=args.fast_dev_run,
    )

    if args.verbose:
        print("Training rows:", train_data.shape[0])
        print("Validation rows:", 0 if val_data is None else val_data.shape[0])
        print("Variables:", variable_names)
        print("Run seed:", run_seed)
    print("Ground truth probability table:\n{}".format(ground_truth_prob_table.to_string(index=False)))

    trainer.fit(model)

    best_path = checkpoint.best_model_path
    if best_path:
        state = T.load(best_path, map_location='cpu')
        model.load_state_dict(state['state_dict'])

    train_loss = trainer.callback_metrics.get('train_loss')
    val_loss = trainer.callback_metrics.get('val_loss')
    results = {
        'train_rows': int(train_data.shape[0]),
        'val_rows': 0 if val_data is None else int(val_data.shape[0]),
        'graph': args.graph,
        'trial': int(args.trial),
        'seed': int(run_seed),
        'data_seed': int(get_data_seed(args)),
        'num_variables': int(train_data.shape[1]),
        'variable_names': variable_names,
        'ground_truth_prob_table': ground_truth_prob_table.to_dict(orient='records'),
        'best_checkpoint': best_path,
        'final_train_loss': None if train_loss is None else float(train_loss.detach().cpu()),
        'final_val_loss': None if val_loss is None else float(val_loss.detach().cpu()),
        'final_val_sample_kl': None if trainer.callback_metrics.get('val_sample_kl') is None else float(
            trainer.callback_metrics['val_sample_kl'].detach().cpu()),
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as fp:
        json.dump(hyperparams, fp, indent=2)


if __name__ == '__main__':
    main()
