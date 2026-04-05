import unittest

import pytorch_lightning as pl
import torch as T

from src.pag import PAGModel, PAGPipeline, make_synthetic_binary_table
from src.pag.data import binary_table_probability_table, probability_table_kl


class PAGSmokeTest(unittest.TestCase):
    def test_model_marginal_log_prob_shape(self):
        data, _ = make_synthetic_binary_table(num_rows=8, graph_name='chain', seed=7)
        model = PAGModel(
            num_variables=3,
            latent_dim=4,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ffn_hidden_dim=32,
            latent_mlp_hidden_dim=16,
            head_hidden_dim=16,
        )

        log_prob = model.marginal_log_prob(data[:5], mc_samples=6)
        self.assertEqual(tuple(log_prob.shape), (5,))
        self.assertTrue(T.isfinite(log_prob).all().item())

    def test_chain_sampler_returns_binary_xyz(self):
        model = PAGModel(
            num_variables=3,
            latent_dim=4,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ffn_hidden_dim=32,
            latent_mlp_hidden_dim=16,
            head_hidden_dim=16,
            graph='chain',
        )

        samples = model.sample_chain(sample_count=12, depth=3, device=T.device('cpu'))
        self.assertEqual(tuple(samples.shape), (12, 3))
        self.assertTrue(T.all((samples == 0) | (samples == 1)).item())

    def test_chain_sampler_uses_mask_order(self):
        model = PAGModel(
            num_variables=3,
            latent_dim=4,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ffn_hidden_dim=32,
            latent_mlp_hidden_dim=16,
            head_hidden_dim=16,
            graph='chain',
        )

        self.assertEqual(model._build_confidence_order(), [2, 1, 0])

    def test_probability_table_and_kl(self):
        data, columns = make_synthetic_binary_table(num_rows=16, graph_name='chain', seed=5)
        truth_table = binary_table_probability_table(data, columns)
        model_table = binary_table_probability_table(data.clone(), columns)
        kl_value = probability_table_kl(truth_table, model_table)

        self.assertEqual(list(truth_table.columns), ['X', 'Y', 'Z', 'P(V)'])
        self.assertAlmostEqual(kl_value, 0.0, places=6)

    def test_pipeline_fast_dev_run(self):
        pl.seed_everything(3)
        data, columns = make_synthetic_binary_table(num_rows=32, graph_name='chain', seed=3)
        train_data = data[:24]
        val_data = data[24:]

        pipeline = PAGPipeline(
            train_data=train_data,
            val_data=val_data,
            variable_names=columns,
            hyperparams={
                'batch-size': 8,
                'lr': 1e-3,
                'mc-sample-size': 4,
                'latent-dim': 4,
                'embed-dim': 16,
                'num-heads': 4,
                'num-layers': 1,
                'ffn-hidden-dim': 32,
                'latent-mlp-hidden-dim': 16,
                'head-hidden-dim': 16,
            },
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=False,
            checkpoint_callback=False,
        )
        trainer.fit(pipeline)

        self.assertIn('train_loss', trainer.callback_metrics)


if __name__ == '__main__':
    unittest.main()
