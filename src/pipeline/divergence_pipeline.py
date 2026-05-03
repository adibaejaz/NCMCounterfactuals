import numpy as np
import pandas as pd
import torch as T

import src.metric.divergences as dvg
from src.metric.evaluation import all_metrics
from src.ds.causal_graph import CausalGraph
from src.run.data_setup import _build_v_sizes
from src.scm.ncm.feedforward_ncm import FF_NCM
from src.scm.scm import expand_do

from .base_pipeline import BasePipeline


class DivergencePipeline(BasePipeline):
    patience = 50

    def __init__(self, generator, do_var_list, dat_sets, cg, dim, hyperparams=None, ncm_model=FF_NCM):
        if hyperparams is None:
            hyperparams = dict()

        v_size = _build_v_sizes(cg, dim, hyperparams)
        ncm = ncm_model(cg, v_size=v_size, default_u_size=hyperparams.get('u-size', 1), hyperparams=hyperparams)
        super().__init__(generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=hyperparams.get('data-bs', 1000))

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.lr = hyperparams.get('lr', 0.001)
        self.ordered_v = cg.v

        self.logged = False

        self.automatic_optimization = False

    def forward(self, n=1000, u=None, do={}):
        return self.ncm(n, u, do)

    def configure_optimizers(self):
        optim = T.optim.AdamW(self.ncm.parameters(), lr=self.lr)
        return {
            'optimizer': optim,
            'lr_scheduler': T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, 50, 1, eta_min=1e-4)
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        opt.zero_grad()
        loss = 0.0
        for i, do_set in enumerate(self.do_var_list):
            ncm_batch = self.ncm(
                self.ncm_batch_size,
                do={k: expand_do(v, n=self.ncm_batch_size).to(self.device) for (k, v) in do_set.items()},
            )
            dat_mat = T.cat([batch[i][k] for k in self.ordered_v], axis=1)
            ncm_mat = T.cat([ncm_batch[k] for k in self.ordered_v], axis=1)
            loss = loss + dvg.MMD_loss(dat_mat.float(), ncm_mat.float(), gamma=1) / len(self.do_var_list)
        loss_val = loss.item()
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss_val, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            if not self.logged:
                results = all_metrics(
                    self.generator,
                    self.ncm,
                    self.do_var_list,
                    self.dat_sets,
                    n=10000,
                    stored=self.stored_metrics,
                )
                for k, v in results.items():
                    self.log(k, v)

                if (self.current_epoch + 1) % 10 == 0:
                    print(pd.Series(results))

                self.logged = True
        else:
            self.logged = False
