"""Main module."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot

from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)
from scvi.model.base import BaseModelClass
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderTOTALVI, EncoderTOTALVI
from scvi.nn._utils import ExpActivation

torch.backends.cudnn.benchmark = True

# protein_vae_module.py

import torch
from scvi.module import TOTALVAE
from scvi.nn import ProteinEncoder, ProteinDecoder
from scvi.module.base import LossOutput
from torch.distributions import Normal, kl_divergence as kl
from scvi import REGISTRY_KEYS

class ProteinVAE(TOTALVAE):
    """
    Protein VAE Module using minimal changes to TOTALVAE.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_batch: int = 0,
        **kwargs,
    ):
        super().__init__(
            n_input_genes=0,  # Set n_input_genes to 0
            n_input_proteins=n_input_proteins,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_batch=n_batch,
            **kwargs,
        )

        # Replace the encoder and decoder with ProteinEncoder and ProteinDecoder
        self.encoder = ProteinEncoder(
            n_input_proteins=n_input_proteins,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers_encoder,
            **kwargs,
        )

        self.decoder = ProteinDecoder(
            n_latent=n_latent,
            n_output_proteins=n_input_proteins,
            n_hidden=n_hidden,
            n_layers=n_layers_decoder,
            **kwargs,
        )

        # Remove RNA-related parameters
        # Adjust other parameters accordingly

    def _get_inference_input(self, tensors):
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY, None)
        return {"y": y, "batch_index": batch_index}

    def inference(self, y, batch_index, **kwargs):
        outputs = self.encoder(y)
        qz = outputs["qz"]
        z = outputs["z"]

        # Compute back_mean_prior
        if self.n_batch > 0 and batch_index is not None:
            py_back_alpha_prior = torch.index_select(
                self.background_pro_alpha, dim=1, index=batch_index.squeeze(-1)
            )
            py_back_beta_prior = torch.exp(
                torch.index_select(
                    self.background_pro_log_beta, dim=1, index=batch_index.squeeze(-1)
                )
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)

        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        return {"qz": qz, "z": z}

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY, None)
        return {"z": z, "batch_index": batch_index}

    def generative(self, z, batch_index, **kwargs):
        py_, log_pro_back_mean = self.decoder(z)

        # Compute dispersion parameter
        if self.n_batch > 0 and batch_index is not None:
            py_r = torch.exp(
                torch.index_select(self.py_r, dim=1, index=batch_index.squeeze(-1))
            )
        else:
            py_r = torch.exp(self.py_r)
        py_["r"] = py_r

        return {"py_": py_, "log_pro_back_mean": log_pro_back_mean}

    def loss(self, tensors, inference_outputs, generative_outputs, **kwargs):
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        py_ = generative_outputs["py_"]
        log_pro_back_mean = generative_outputs["log_pro_back_mean"]

        # Reconstruction loss
        from scvi.distributions import NegativeBinomialMixture
        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        reconst_loss_protein = -py_conditional.log_prob(y).sum(dim=-1)

        # KL divergences
        qz = inference_outputs["qz"]
        kl_div_z = kl(qz, Normal(0, 1)).sum(dim=1)
        kl_div_back_pro = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        ).sum(dim=1)

        loss = torch.mean(reconst_loss_protein + kl_div_z + kl_div_back_pro)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss_protein,
            kl_local={
                "kl_divergence_z": kl_div_z,
                "kl_divergence_back_pro": kl_div_back_pro,
            },
        )
