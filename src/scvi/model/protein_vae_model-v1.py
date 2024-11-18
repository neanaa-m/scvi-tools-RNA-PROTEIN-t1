from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data._utils import _check_nonnegative_integers
from scvi.dataloaders import DataSplitter
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_manager,
    _init_library_size,
    cite_seq_raw_counts_properties,
    get_max_epochs_heuristic,
)
from scvi.model.base._de_core import _de_core
from scvi.module import TOTALVAE
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp

from .base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scvi._types import Number
# protein_vae_model.py

import logging
from typing import Optional

from anndata import AnnData
import torch

from scvi.data import AnnDataManager, fields
from scvi.model import TOTALVI
from scvi.model.base import VAEMixin
from scvi.module import TOTALVAE
from scvi.train import TrainingPlan, TrainRunner

logger = logging.getLogger(__name__)

class ProteinVAEModel(TOTALVI):
    """
    Protein VAE Model using minimal changes to totalVI.
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(
            adata=adata,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            **model_kwargs,
        )

        # Override the module with ProteinVAE
        self.module = ProteinVAE(
            n_input_genes=0,  # Set n_input_genes to 0
            n_input_proteins=self.summary_stats.n_proteins,
            n_batch=self.summary_stats.n_batch,
            n_labels=self.summary_stats.n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_decoder=dropout_rate_decoder,
            **model_kwargs,
        )

        self._model_summary_string = f"ProteinVAEModel with n_latent: {n_latent}"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        protein_expression_obsm_key: str = "protein_expression",
        protein_names_uns_key: Optional[str] = None,
        **kwargs,
    ):
        """Sets up AnnData for use with ProteinVAEModel.

        Parameters
        ----------
        adata
            AnnData object.
        batch_key
            Key in adata.obs for batch information.
        protein_expression_obsm_key
            Key in adata.obsm for protein expression data.
        protein_names_uns_key
            Key in adata.uns for protein names. If None, uses the column names of adata.obsm[protein_expression_obsm_key] if it is a DataFrame.
        """
        # Only register protein data
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = fields.CategoricalObsField("batch", batch_key)
        protein_field = fields.ProteinObsmField(
            "protein_expression",
            protein_expression_obsm_key,
            colnames_uns_key=protein_names_uns_key,
            is_count_data=True,
            batch_field=batch_field,
        )
        adata_fields = [
            batch_field,
            protein_field,
        ]
        adata_manager = AnnDataManager(
            fields=adata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 400,
        lr: float = 1e-3,
        batch_size: int = 128,
        **kwargs,
    ):
        """Trains the model.

        Parameters
        ----------
        max_epochs
            Number of epochs to train.
        lr
            Learning rate.
        batch_size
            Batch size.
        **kwargs
            Other keyword arguments.
        """
        data_splitter = DataSplitter(
            self.adata_manager,
            batch_size=batch_size,
            **kwargs,
        )
        training_plan = TrainingPlan(self.module, lr=lr)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **kwargs,
        )
        return runner()

    # mudata handling 
    @classmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: Optional[str] = None,
        protein_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        modalities: dict[str, str] = {"rna_layer": "rna", "protein_layer": "protein"},
        **kwargs,
    ):
        """Sets up MuData for use with ProteinVAEModel."""
        setup_method_args = cls._get_setup_method_args(**locals())
    
        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities["rna_layer"],
        )
    
        mudata_fields = [
            fields.MuDataLayerField(
                REGISTRY_KEYS.X_KEY,
                rna_layer,
                mod_key=modalities["rna_layer"],
                is_count_data=True,
                mod_required=True,
            ),
            fields.MuDataProteinLayerField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_layer,
                mod_key=modalities["protein_layer"],
                use_batch_mask=True,
                batch_field=batch_field,
                is_count_data=True,
                mod_required=True,
            ),
        ]
    
        adata_manager = AnnDataManager(
            fields=mudata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)
