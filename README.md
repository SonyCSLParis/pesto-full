# PESTO: Pitch Estimation with Self-Supervised Transposition-equivariant Objective

**tl;dr:** Fast pitch estimation with self-supervised learning

This repository implements the full code of the [PESTO](https://arxiv.org/abs/2309.02265) paper,
which received the Best Paper Award at [ISMIR 2023](https://ismir2023.ismir.net/).

The purpose of this repository is to provide the whole pipeline for training a PESTO model.
End-users that do not need to know the specific implementation details can check instead [this repository](https://github.com/SonyCSLParis/pesto).


## Setup

```shell
git clone https://github.com/SonyCSLParis/pesto-full.git

cd pesto-full
pip install -r requirements
# or
conda env create -f environment.yml
```

**Extra dependencies:**
- [mir_eval](https://craffel.github.io/mir_eval/) for computing metrics
- [scikit-learn](https://scikit-learn.org) for cross-validation
- [wandb](https://wandb.ai) for cool logging


**Troubleshooting:** Latest version of `nnAudio` (0.3.2) uses some deprecated NumPy functions, which leads to errors.
Just overwrite the problematic files by replacing the `np.float` by `float`.


## Usage

This repository is implemented in [PyTorch](https://pytorch.org/) and relies on [Lightning](https://lightning.ai/) and [Hydra](https://hydra.cc/).
It follows the structure of the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### Basic usage

The main training script is `src/train.py`.
To train the model and log metrics in a csv file, you can run the following command:
```shell
python src/train.py data=mir-1k logger=csv
```
To use different loggers, just pick one with an existing configuration (see `configs/logger`) or create your own config, 
then replace `logger=csv` by `logger=my_logger`.

In particular, some logging features are supposed to be used with [W&B](https://wandb.ai).

### Training on a custom dataset

To deal with arbitrary dataset nested structures, datasets are specified as `csv` files.

For training on your own data, create a new YAML file `configs/data/my_dataset.yaml` and specify the path to:
- a text file containing the list of audio files in your dataset
- (optional) a text file containing the corresponding pitch annotations

Since PESTO is a fully self-supervised method, pitch annotations will never be used during training,
however if they are provided they will be used in the validation step to compute the metrics.

To generate such files, one can take advantage of the command `find`. For example:
```shell
find MIR-1K/Vocals -name "*.wav" | sort > mir-1k.csv
find MIR-1K/PitchLabel  -name "*.csv" | sort > mir-1k_annot.csv
```
will explore recursively the appropriate directories to generate the text files containing the list of audios and annotations, respectively.
Note the use of `sort` to ensure the audios and annotations to be provided in the same order.

An example config `configs/data/mir-1k.yaml` is provided as reference.


## Code organization

The code follows the structure of [this repository](https://github.com/ashleve/lightning-hydra-template).
It contains two main folders: `configs` contains the YAML config files and `src` contains the code.
In practice, configurations are built by Hydra, hence the disentangled structure of the config folder.

The `src` folder that contains the code is divided as follows:
- `train.py` is the main script. Everything is instantiated from the built config in this script using `hydra.instantiate`.
- `models` contains the main PESTO `LightningModule` as well as the transposition-equivariant architecture
- `data` contains the main `AudioDatamodule` that handles all data loading, as well as several transforms (Harmonic CQT, pitch-shift, data augmentations...)
- `losses` contains the implementation of the fancy SSL losses that we use to train our model.
- `callbacks` contains the code for computing metrics, the procedure for weighting the loss terms based on their respective gradients, as well as additional visualization callbacks.
- `utils` contains stuff.


## Miscellaneous

### Training on different devices

By default, the model is trained on one GPU.
The memory requirements being very low (~500 MB), we do not support multi-GPU.
However, training on CPU (while discouraged) is possible by setting option `trainer=cpu` in the CLI.

### Changing sampling rates

The model takes as inputs individual CQT frames so it doesn't care about the sampling rate of your audios.
In particular, CQT kernels are computed dynamically so you never have to care about sampling rate.

### Data caching

In order to simplify the implementation, all the CQT frames of the dataset are 
automatically computed from the audio files at the beginning of the first training and cached for avoiding having to always recompute them.

The cache directory `./cache` by default can be overwritten by setting the `cache_dir` option of the `AudioDatamodule`.
Moreover, CQT frames are stored with a unique hash that takes into account the path to audio files as well as the CQT options.

If you change the dataset or an option from the CQT (e.g. hop size), CQT frames will be recomputed and cached as well.
The only case where you should be careful with the caching system is if you change the content of the `audio_files`/`annot_files` text file
or the audios/annotations themselves.

### Data splitting

All the data loading logic is handled within `src/data/audio_datamodule.py`.

There are several options for splitting data into training and validation set:
- **Naive:** If you provide a `annot_files`, the model will be trained and validated on the same dataset.
Otherwise, a dummy `val_dataloader()` will be created to avoid weird Lightning bugs, but the logged metrics won't of course make any sense.
- **Manual:** You can manually provide a validation set by setting `val_audio_files` and `val_annot_files` in your YAML config.
The structure of those files should be identical to the ones of the training set.
- **Cross-validation:** If you provide an annotated training set but no validation set,
you can however perform cross-validation by setting args `fold` and `n_folds` in your YAML config.
Note that one corresponds to one single training, so in order to perform the whole cross-validation you should run the 
script `n_folds` times, either manually or by taking advantage of Hydra's multirun option for example.
Note also that the splitting strategy in that case has its own random state: for a given value of `n_folds`,
`fold=<i>` will always correspond to the same train/val split **even if you change the global seed**.

### Variable interpolation

This repository aims at taking advantage as much as possible from Hydra and OmegaConf for handling configurations.
In particular, several variables are interpolated automatically to limit the number of changes to try new stuff.
For example, changing the resolution of the CQT has a strong influence on many parameters
such as the input/output dimension of the network, the construction of the loss, etc.

However, thanks to variable interpolation, you can try to increase this CQT resolution by just typing:
```shell
python src/train.py data=my_data logger=csv data.bins_per_semitone=5
```
In practice, you shouldn't need to overwrite parameters that are defined through variable interpolation.

For more details about the configuration system management, please check [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) and [Hydra](https://hydra.cc/) docs.

## Cite

If you want to use this work, please cite:
```
@inproceedings{PESTO,
    author = {Riou, Alain and Lattner, Stefan and Hadjeres, Gaëtan and Peeters, Geoffroy},
    booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023},
    publisher = {International Society for Music Information Retrieval},
    title = {PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective},
    year = {2023}
}
```


## Credits

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) for the main structure of the code
- [nnAudio](https://github.com/KinWaiCheuk/nnAudio) for the original CQT implementation
- [multipitch-architectures](https://github.com/christofw/multipitch_architectures) for the original architecture of the model
- [mir_eval](https://craffel.github.io/mir_eval/) for computing MIR metrics (RPA, RCA...)

```
@ARTICLE{9174990,
    author={K. W. {Cheuk} and H. {Anderson} and K. {Agres} and D. {Herremans}},
    journal={IEEE Access}, 
    title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks}, 
    year={2020},
    volume={8},
    number={},
    pages={161981-162003},
    doi={10.1109/ACCESS.2020.3019084}}
@ARTICLE{9865174,
    author={Weiß, Christof and Peeters, Geoffroy},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={Comparing Deep Models and Evaluation Strategies for Multi-Pitch Estimation in Music Recordings}, 
    year={2022},
    volume={30},
    number={},
    pages={2814-2827},
    doi={10.1109/TASLP.2022.3200547}}
```

