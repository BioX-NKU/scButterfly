.. module:: scButterfly
.. automodule:: scButterfly
   :noindex:

API
====


Import scButterfly::

   import scButterfly

butterfly
-------------------------
.. module::scButterfly.butterfly
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    butterfly.Butterfly
    butterfly.Butterfly.load_data
    butterfly.Butterfly.data_preprocessing
    butterfly.Butterfly.augmentation
    butterfly.Butterfly.construct_model
    butterfly.Butterfly.train_model
    butterfly.Butterfly.test_model
    

calculate_cluster
-------------------------
.. module::scButterfly.calculate_cluster
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    calculate_cluster.calculate_cluster_index


data_processing
-------------------------
.. module::scButterfly.data_processing
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    data_processing.inverse_TFIDF
    data_processing.RNA_data_preprocessing
    data_processing.ATAC_data_preprocessing
    data_processing.CLR_transform


draw_cluster
-------------------------
.. module::scButterfly.draw_cluster
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    draw_cluster.draw_tsne


train_model
-------------------------
.. module::scButterfly.train_model
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    train_model.Model
    train_model.Model.train
    train_model.Model.test


train_model_cite
-------------------------
.. module::scButterfly.train_model_cite
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    train_model_cite.Model
    train_model_cite.Model.train
    train_model_cite.Model.test


train_model_perturb
-------------------------
.. module::scButterfly.train_model_perturb
.. currentmodule::scButterfly
.. autosummary::
    :toctree: .

    train_model_perturb.Model
    train_model_perturb.Model.train
    train_model_perturb.Model.test