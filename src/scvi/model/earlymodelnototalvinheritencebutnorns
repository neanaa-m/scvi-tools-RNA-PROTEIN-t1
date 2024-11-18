import logging
from typing import Optional

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import devices_dsp, setup_anndata_dsp
from scvi.module import ProteinVAE  # Import the module we'll define later

logger = logging.getLogger(__name__)


class ProteinVAEModel(VAEMixin, BaseModelClass):
    """
    Model for training a VAE on protein data alone.
    """

    _module_cls = ProteinVAE
    _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.protein_state_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        )

        n_batch = self.summary_stats.n_batch
        n_proteins = self.summary_stats.n_proteins

        self.module = self._module_cls(
            n_input_proteins=n_proteins,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers_encoder=n_layers,
            n_layers_decoder=n_layers,
            n_batch=n_batch,
            dropout_rate_encoder=dropout_rate,
            dropout_rate_decoder=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **model_kwargs,
        )
        self._model_summary_string = f"ProteinVAEModel with n_latent: {n_latent}"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        protein_expression_obsm_key: str = "protein_expression",
        protein_names_uns_key: Optional[str] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        protein_expression_obsm_key
            Key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            Key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        protein_field = fields.ProteinObsmField(
            REGISTRY_KEYS.PROTEIN_EXP_KEY,
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

    @devices_dsp.dedent
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
            Other keyword arguments for the training plan and runner.
        """
        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            batch_size=batch_size,
            **kwargs,
        )
        training_plan = self._training_plan_cls(self.module, lr=lr)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **kwargs,
        )
        return runner()
