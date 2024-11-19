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
from scvi.nn import _base_components

from scvi.module.base import LossOutput
from torch.distributions import Normal, kl_divergence as kl
from scvi import REGISTRY_KEYS
#from BaseModuleClass import ProteinEncoder, ProteinDecoder


class ProteinVAE(TOTALVAE, BaseModuleClass):
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
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;__init__ ")
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
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;_get_inference_input ")

        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY, None)
        return {"y": y, "batch_index": batch_index}

    def inference(self, y, batch_index, **kwargs):
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;inference ")

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
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;_get_generative_input ")

        z = inference_outputs["z"]
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY, None)
        return {"z": z, "batch_index": batch_index}

    def generative(self, z, batch_index, **kwargs):
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;generative ")

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
        print("protein_vae_moduletx.py  _totalvae, class ProteinVAE, function;loss ")

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

#####These are encoder and decoder for 2-protein to protein###
#Protein Encoder
#The encoder will take protein expression data and output the parameters of the latent distribution (mean and variance).
#These two are used by  ProteinVAE module.
import torch
from torch import nn
from torch.distributions import Normal


class ProteinEncoder(nn.Module):
    def __init__(
        self,
        n_input_proteins: int,
        n_latent: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        n_continuous_cov: int = 0,  # Include argument passed by instantiation
        n_cats_per_cov=None,
        gene_dispersion=None,
        protein_dispersion=None,
        gene_likelihood=None,
        latent_distribution=None,
        protein_batch_mask=None,
        protein_background_prior_mean=None,
        protein_background_prior_scale=None,
        use_size_factor_key=False,
        library_log_means=None,
        library_log_vars=None,
        **kwargs,  # Catch-all for future extensions
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input_proteins,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = nn.Linear(n_hidden, n_latent)

    def forward(self, y: torch.Tensor, *cat_list: int):
        q = self.encoder(y, *cat_list)
        q_m = self.z_mean_encoder(q)
        q_v = torch.exp(self.z_var_encoder(q)) + 1e-4  # Adding epsilon for numerical stability
        q_z = Normal(q_m, q_v.sqrt())
        z = q_z.rsample()
        return {"qz": q_z, "z": z}

#Protein Decoder
#The decoder will output parameters for both the background noise and the biological signal of the proteins, similar to DecoderTOTALVI in totalVI.

class ProteinDecoder(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_output_proteins: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_function_bg: Literal["exp", "softplus"] = "softplus",
        n_continuous_cov: int = 0,  # Include argument passed by instantiation
        n_cats_per_cov=None,
        gene_dispersion=None,
        protein_dispersion=None,
        gene_likelihood=None,
        latent_distribution=None,
        protein_batch_mask=None,
        protein_background_prior_mean=None,
        protein_background_prior_scale=None,
        use_size_factor_key=False,
        library_log_means=None,
        library_log_vars=None,
        **kwargs,  # Catch-all for future extensions
    ):
        super().__init__()

        self.n_output_proteins = n_output_proteins

        if activation_function_bg == "exp":
            self.activation_function_bg = ExpActivation()
        elif activation_function_bg == "softplus":
            self.activation_function_bg = nn.Softplus()

        linear_args = {
            "n_layers": 1,
            "use_activation": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0,
        }

        # Background mean decoder
        self.py_back_decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.py_back_mean_log_alpha = FCLayers(
            n_in=n_hidden + n_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        self.py_back_mean_log_beta = FCLayers(
            n_in=n_hidden + n_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # Foreground increment decoder
        self.py_fore_decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden + n_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.ReLU,
        )

        # Mixture component
        self.py_mixing_decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.py_mixing_output = FCLayers(
            n_in=n_hidden + n_latent,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        py_ = {}

        # Background parameters
        py_back = self.py_back_decoder(z, *cat_list)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z, *cat_list)
        py_["back_beta"] = self.activation_function_bg(
            self.py_back_mean_log_beta(py_back_cat_z, *cat_list)
        ) + 1e-8

        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        # Foreground parameters
        py_fore = self.py_fore_decoder(z, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z, *cat_list) + 1 + 1e-8
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        # Mixture parameters
        py_mixing = self.py_mixing_decoder(z, *cat_list)
        py_mixing_cat_z = torch.cat([py_mixing, z], dim=-1)
        py_["mixing"] = self.py_mixing_output(py_mixing_cat_z, *cat_list)

        # Compute final scale parameter
        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize(
            (1 - protein_mixing) * py_["rate_fore"], p=1, dim=-1
        )

        return py_, log_pro_back_mean

##
import collections
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.nn._utils import ExpActivation


def _identity(x):
    return x


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


 
