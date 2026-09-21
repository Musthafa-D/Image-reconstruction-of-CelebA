# CelebA Image Reconstruction with Encoder–Decoder Networks

PyTorch experiments on encoding an image into an internal representation and reconstructing it through a decoder. The project combines reconstruction training with image comparisons and tools for examining how auxiliary networks respond to reconstructed outputs.

The data loader selects ten facial attributes for experiments using label information. The checked-in configuration activates the unconditional encoder/decoder with three-channel inputs and outputs.

## Method

An encoder transforms the image into a learned representation. A decoder maps it back into image space, and training minimises a reconstruction objective. The source provides unconditional and conditional classes, with configuration selecting the active variant.

Unlike unconstrained image generation, reconstruction uses the original input as the reference for the output.

## Repository guide

| File | Purpose |
|---|---|
| `Networks/encoder_decoder.py` | Encoder, decoder and conditional variants |
| `Networks/classfiers.py` | Auxiliary classifiers; original filename retained |
| `Networks/discriminators.py` | Auxiliary discriminators |
| `data_loader.py` | Dataset preparation |
| `learner.py` | Reconstruction training and evaluation |
| `utils.py` | Model loading and analysis helpers |
| `fid_custom.py` and `metric_custom.py` | Additional evaluation implementations |
| `plots.py` | Image comparisons and diagnostic plots |
| `main.py` and `optuna_hyp.py` | Experiment study |

## Preparing a run

The code uses PyTorch and the external `ccbdl` framework for configuration, data loading, experiment storage and parts of the learning workflow. A compatible installation of that framework is required; it is not bundled here. Other dependencies include torchvision, NumPy, Matplotlib, Optuna and Captum, with additional analysis libraries used by individual modules.

Use the original compatible environment, prepare the dataset at the configured location, and run from the repository root so relative paths resolve correctly. The archive does not include a complete dependency lock file. Supply datasets and optional pretrained models separately where referenced.

The learner loads an auxiliary classifier during setup. Its expected checkpoint files are not included in this snapshot and must be supplied before using that path.

Review `config.yaml`: the dataset, encoder/decoder, channels, device and auxiliary models must agree. Framework task names belong to the loader interface and do not by themselves describe the reconstruction architecture. Several model settings explicitly name CUDA.

Once the environment, data and checkpoints are prepared, the study entry point is:

```bash
python main.py
```

This launches the configured study rather than a separate pretrained-model demonstration.

## Examining reconstructions

Compare originals and reconstructions as matched pairs. Reconstruction loss and paired-image metrics describe retained detail; visual examples reveal blur and systematic failures that an average can hide.

The auxiliary classifier-based Fréchet distance is specific to its feature extractor and is not directly comparable with standard Inception FID. Classifier/discriminator attribution plots inspect those auxiliary models; they do not automatically explain every operation inside the encoder/decoder.

## Scope

The repository contains the experimental implementation and analysis routines. It does not assert a reproduced benchmark score. Present results together with the configuration, split, checkpoint identity and evaluation protocol.
