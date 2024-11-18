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

logger = logging.getLogger(__name__)


class confirmationthattheseareenough(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """total Variational Inference :cite:p:`GayosoSteier21`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.TOTALVI.setup_anndata`.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following:

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for
        each of 100 cells per batch and averages the distributions. Note that even with this option
        set to `True`, this only initializes a parameter that is learned during inference. If
        `False`, randomly initializes. The default (`None`), sets this to `True` if greater than 10
        proteins are used.
    override_missing_proteins
        If `True`, will not treat proteins with all 0 expression in a particular batch as missing.
    **model_kwargs
        Keyword args for :class:`~scvi.module.TOTALVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.TOTALVI.setup_anndata(
            adata, batch_key="batch", protein_expression_obsm_key="protein_expression"
        )
    >>> vae = scvi.model.TOTALVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_totalVI"] = vae.get_latent_representation()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/multimodal/totalVI`
    2. :doc:`/tutorials/notebooks/multimodal/cite_scrna_integration_w_totalVI`
    3. :doc:`/tutorials/notebooks/scrna/scarches_scvi_tools`
    """

    _module_cls = TOTALVAE
    #load your module and make it here
    _data_splitter_cls = DataSplitter
    _training_plan_cls = AdversarialTrainingPlan
    #make this just training plan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        empirical_protein_background_prior: bool | None = None,
        override_missing_proteins: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.protein_state_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        )
        if (
            fields.ProteinObsmField.PROTEIN_BATCH_MASK in self.protein_state_registry
            and not override_missing_proteins
        ):
            batch_mask = self.protein_state_registry.protein_batch_mask
            msg = (
                "Some proteins have all 0 counts in some batches. "
                + "These proteins will be treated as missing measurements; however, "
                + "this can occur due to experimental design/biology. "
                + "Reinitialize the model with `override_missing_proteins=True`,"
                + "to override this behavior."
            )
            warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
            self._use_adversarial_classifier = True
        else:
            batch_mask = None
            self._use_adversarial_classifier = False

        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (self.summary_stats.n_proteins > 10)
        )
        if emp_prior:
            prior_mean, prior_scale = self._get_totalvi_protein_priors(adata)
        else:
            prior_mean, prior_scale = None, None

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)[
                fields.CategoricalJointObsField.N_CATS_PER_KEY
            ]
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input_genes=self.summary_stats.n_vars,
            n_input_proteins=self.summary_stats.n_proteins,
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            protein_batch_mask=batch_mask,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        # how you instantiate your module? 
        self._model_summary_string = (
            f"TotalVI Model with the following params: \nn_latent: {n_latent}, "
            f"gene_dispersion: {gene_dispersion}, protein_dispersion: {protein_dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}"
        )
        self.init_params_ = self._get_init_params(locals())

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 4e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        adversarial_classifier: bool | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        external_indexing: list[np.array] = None,
        **kwargs,
    ):
        """Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping`
            is `True` or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are
            met, val is checked every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing
            proteins are detected.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed
            to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        external_indexing
            A list of data split indices in the order of training, validation, and test sets.
            Validation and test set are not required and can be left empty.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if adversarial_classifier is None:
            adversarial_classifier = self._use_adversarial_classifier
        n_steps_kl_warmup = (
            n_steps_kl_warmup if n_steps_kl_warmup is not None else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            external_indexing=external_indexing,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()

    def _validate_anndata(self, adata: AnnData | None = None, copy_if_view: bool = True):
        adata = super()._validate_anndata(adata=adata, copy_if_view=copy_if_view)
        error_msg = (
            "Number of {} in anndata different from when setup_anndata was run. Please rerun "
            "setup_anndata."
        )
        if REGISTRY_KEYS.PROTEIN_EXP_KEY in self.adata_manager.data_registry.keys():
            pro_exp = self.get_from_registry(adata, REGISTRY_KEYS.PROTEIN_EXP_KEY)
            if self.summary_stats.n_proteins != pro_exp.shape[1]:
                raise ValueError(error_msg.format("proteins"))
            is_nonneg_int = _check_nonnegative_integers(pro_exp)
            if not is_nonneg_int:
                warnings.warn(
                    "Make sure the registered protein expression in anndata contains "
                    "unnormalized count data.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
        else:
            raise ValueError("No protein data found, please setup or transfer anndata")

        return adata

    def _get_totalvi_protein_priors(self, adata, n_cells=100):
        """Compute an empirical prior for protein background."""
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            logger.info("Computing empirical prior initialization for protein background.")

            adata = self._validate_anndata(adata)
            adata_manager = self.get_anndata_manager(adata)
            pro_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
            pro_exp = pro_exp.to_numpy() if isinstance(pro_exp, pd.DataFrame) else pro_exp
            batch_mask = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).get(
                fields.ProteinObsmField.PROTEIN_BATCH_MASK
            )
            batch = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).ravel()
            cats = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)[
                fields.CategoricalObsField.CATEGORICAL_MAPPING_KEY
            ]
            codes = np.arange(len(cats))

            batch_avg_mus, batch_avg_scales = [], []
            for b in np.unique(codes):
                # can happen during online updates
                # the values of these batches will not be used
                num_in_batch = np.sum(batch == b)
                if num_in_batch == 0:
                    batch_avg_mus.append(0)
                    batch_avg_scales.append(1)
                    continue
                batch_pro_exp = pro_exp[batch == b]

                # non missing
                if batch_mask is not None:
                    batch_pro_exp = batch_pro_exp[:, batch_mask[str(b)]]
                    if batch_pro_exp.shape[1] < 5:
                        logger.debug(
                            f"Batch {b} has too few proteins to set prior, setting randomly."
                        )
                        batch_avg_mus.append(0.0)
                        batch_avg_scales.append(0.05)
                        continue

                # a batch is missing because it's in the reference but not query data
                # for scarches case, these values will be replaced by original state dict
                if batch_pro_exp.shape[0] == 0:
                    batch_avg_mus.append(0.0)
                    batch_avg_scales.append(0.05)
                    continue

                cells = np.random.choice(np.arange(batch_pro_exp.shape[0]), size=n_cells)
                batch_pro_exp = batch_pro_exp[cells]
                gmm = GaussianMixture(n_components=2)
                mus, scales = [], []
                # fit per cell GMM
                for c in batch_pro_exp:
                    try:
                        gmm.fit(np.log1p(c.reshape(-1, 1)))
                    # when cell is all 0
                    except ConvergenceWarning:
                        mus.append(0)
                        scales.append(0.05)
                        continue

                    means = gmm.means_.ravel()
                    sorted_fg_bg = np.argsort(means)
                    mu = means[sorted_fg_bg].ravel()[0]
                    covariances = gmm.covariances_[sorted_fg_bg].ravel()[0]
                    scale = np.sqrt(covariances)
                    mus.append(mu)
                    scales.append(scale)

                # average distribution over cells
                batch_avg_mu = np.mean(mus)
                batch_avg_scale = np.sqrt(np.sum(np.square(scales)) / (n_cells**2))

                batch_avg_mus.append(batch_avg_mu)
                batch_avg_scales.append(batch_avg_scale)

            # repeat prior for each protein
            batch_avg_mus = np.array(batch_avg_mus, dtype=np.float32).reshape(1, -1)
            batch_avg_scales = np.array(batch_avg_scales, dtype=np.float32).reshape(1, -1)
            batch_avg_mus = np.tile(batch_avg_mus, (pro_exp.shape[1], 1))
            batch_avg_scales = np.tile(batch_avg_scales, (pro_exp.shape[1], 1))

        return batch_avg_mus, batch_avg_scales


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        %(param_batch_key)s
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            fields.CategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY, None
            ),  # Default labels field for compatibility with TOTALVAE
            batch_field,
            fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = None,
        protein_layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        """%(summary_mdata)s.

        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            RNA layer key. If `None`, will use `.X` of specified modality key.
        protein_layer
            Protein layer key. If `None`, will use `.X` of specified modality key.
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_modalities)s

        Examples
        --------
        >>> mdata = muon.read_10x_h5("pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5")
        >>> scvi.model.TOTALVI.setup_mudata(
                mdata, modalities={"rna_layer": "rna": "protein_layer": "prot"}
            )
        >>> vae = scvi.model.TOTALVI(mdata)
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )
        mudata_fields = [
            fields.MuDataLayerField(
                REGISTRY_KEYS.X_KEY,
                rna_layer,
                mod_key=modalities.rna_layer,
                is_count_data=True,
                mod_required=True,
            ),
            fields.MuDataCategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY,
                None,
                mod_key=None,
            ),  # Default labels field for compatibility with TOTALVAE
            batch_field,
            fields.MuDataNumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                mod_key=modalities.size_factor_key,
                required=False,
            ),
            fields.MuDataCategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                categorical_covariate_keys,
                mod_key=modalities.categorical_covariate_keys,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                continuous_covariate_keys,
                mod_key=modalities.continuous_covariate_keys,
            ),
            fields.MuDataProteinLayerField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_layer,
                mod_key=modalities.protein_layer,
                use_batch_mask=True,
                batch_field=batch_field,
                is_count_data=True,
                mod_required=True,
            ),
        ]
        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)
