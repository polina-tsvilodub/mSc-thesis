# Language Drift of Multi-Agent Communication Systems in Reference Games

This thesis contains the supporting matrials and code of the M.Sc. thesis mentioned above and submitted to the University of Osnabrück. The thesis builds upon the work by Lazaridou et al. (2020). The code implements reference games on the MS COCO and 3Dshapes datasets.
The repository contains two main directories:

* `code`: coontains the code for all experiments and evaluations reported in the thesis
* `writing`: contains the proposal and the thesis manuscript source files

Below, the structure ad the critical functional components of the code base are documented. 

## `code`

* `scripts`: contains shell scripts for executing the reference game experiment with desired configurations, possibly to be tracked by `sacred` (see below). 
    * `ref_game_*`: these scripts contain reference game executions which were part of the initial grid search over speaker parameters prior to main experiments. 
    * `pretrain_*`: these scripts contain executions of the speaker pretraining scripts.
    * `final_ref_game_*`: these scripts contain executions of the main reference game experiments which were conducted locally.
* `src`: main directory holding all source files for the experiments
    * `agents`: 
        * `listener.py`, `speaker.py`: scripts defining the speaker and listener agents for the reference game.
        * `resnet_encoder.py`: base CNN utilized for extracting image features with the pretrained ResNet model.
    * `drift_metrics`:
        * `image_captioner.py`: vanilla speaker module replication for purposes of loading in the language drift metric computations.
        * `metrics.py`: contains implementations of all language drift measures employed in this work: semantic, structural and functional (discrete and continuous overlaps) drifts.
    * `notebooks`: directory with notebooks used for exploring, evaluating the models and plotting the results.
        * `coco_captions/`: directory containing image caption evaluation scripts adapted from https://github.com/daqingliu/coco-caption. Contact author in order to receive them (not commited due to size).
        * `compute_language_drift.ipynb`: notebook for computing test performances and language drift metrics on the test splits.
        * `dalle_mini_inference.ipynb`: notebook attempting to run the text-to-image DALL-E mini model (does not work).
        * `dataset_exploration.ipynb`: notebook for exploring the properties and examples of the MS COCO dataset.
        * `evaluate_speaker.ipynb`: notebook for computing standard image caption evaluation metrics on the speaker's productions and investigating the granularity of the speaker's messages.
        * `language_drift_stats.ipynb`: notebook for computing some statistics on the hypotheses discussed in the thesis
        * `plot_metrics.eval`: notebook for creating compound plots, especially the ones in the Appendix.
        * `text2image_exploration.ipynb`: notebook exploring similarity metrics that may be applied to raw images. 
    * `reference_game_utils`: directory with utility scripts for running reference games
        * `train.py`: main script executing the reference game training and language drift dynamics computations.
        * `update_policy.py`: script for computing speaker loss updates implementing the REINFORCE algorithm.
    * `utils`:
        * `build_dataset.py`: dataloaders for the two datasets
        * `dataset_utils.py`: custom `Dataset` classes for both datasets.
        * `download.py`: utility script for downloading the MS COCO dataset if required.
        * `early_stopping.py`: class for early stopping during training.
        * `train.py`: main script executing the training and validation steps of speaker pretraining.
        * `vocabulary.py`: custom vocabulary classes for bth datasets.
    * `imgID2annID.json`: a mapping of MS COCO image IDs onto their annotation IDs.
    * `pretrain_speaker.py`: entrypoint script for pretraining the speakers (on eother dataset).
    * `reference_game_train.py`: entrypoint for the reference game experiments on either dataset with any configurations considered in the thesis.

### Tracking experiments

Results of grid searches conducted prior to main experiments were tracked with the MLOps package `sacred` which additionally allows the visualization of results via a graphical dashboard `Omniboard`. In order to set up the experiment tracking, do:
* install `sacred` via `pip install sacred`
* add the experiment logging as it is exemplified in `src/reference_game_train.py`
* run the `docker-compose.yml` file by calling `docker-compose up` (requires docker)
* access Omniboard at (localhost:9000)[localhost:9000]
* contact author regarding information about the `.env` file required for running this stack

### Requirements

In order to run the code locally, install the requirements listed in the preliminary file `requirements_torch.yml` (to be completed).

### References 
Lazaridou, A., Potapenko, A., & Tieleman, O. (2020). Multi-agent communication meets
natural language: Synergies between functional and structural language learning.
arXiv preprint arXiv:2005.07064.