import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers
from scvi.distributions import NegativeBinomialMixture


class ProteinVAE(BaseModuleClass):
    """
    VAE model for protein data only.
    """

    def __init__(
        self,
        n_input_proteins: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_batch: int = 0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_input_proteins = n_input_proteins
        self.n_batch = n_batch

        # Encoder
        self.encoder = FCLayers(
            n_in=n_input_proteins,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate_encoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder = torch.nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = torch.nn.Linear(n_hidden, n_latent)

        # Decoder
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers_decoder,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.py_back_mean_log_alpha = torch.nn.Linear(n_hidden, n_input_proteins)
        self.py_back_mean_log_beta = torch.nn.Linear(n_hidden, n_input_proteins)
        self.py_fore_scale_decoder = torch.nn.Linear(n_hidden, n_input_proteins)
        self.py_mixing_decoder = torch.nn.Linear(n_hidden, n_input_proteins)

        # Dispersion parameters
        self.py_r = torch.nn.Parameter(torch.randn(n_input_proteins))

    def _get_inference_input(self, tensors):
        y = tensors["protein_expression"]
        return {"y": y}

    @auto_move_data
    def inference(self, y, **kwargs):
        q = self.encoder(y)
        q_m = self.z_mean_encoder(q)
        q_v = torch.exp(self.z_var_encoder(q)) + 1e-4
        qz = Normal(q_m, q_v.sqrt())
        z = qz.rsample()
        return {"qz": qz, "z": z}

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return {"z": z}

    @auto_move_data
    def generative(self, z, **kwargs):
        decoder_output = self.decoder(z)
        py_ = {
            "back_alpha": self.py_back_mean_log_alpha(decoder_output),
            "back_beta": F.softplus(self.py_back_mean_log_beta(decoder_output)) + 1e-8,
            "fore_scale": F.softplus(self.py_fore_scale_decoder(decoder_output)) + 1e-8,
            "mixing": self.py_mixing_decoder(decoder_output),
        }
        return {"py_": py_}

    def loss(self, tensors, inference_outputs, generative_outputs, **kwargs):
        y = tensors["protein_expression"]
        qz = inference_outputs["qz"]
        py_ = generative_outputs["py_"]

        reconst_loss = -NegativeBinomialMixture(
            mu1=py_["back_alpha"],
            mu2=py_["fore_scale"],
            theta1=self.py_r,
            mixture_logits=py_["mixing"],
        ).log_prob(y).sum(dim=1)

        kl_loss = kl(qz, Normal(0, 1)).sum(dim=1)
        return LossOutput(loss=(reconst_loss + kl_loss).mean())
