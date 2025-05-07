# CS598 DL4H Final Project
## Do We Still Need Clinical Language Models?

This repository is a replication project for the CS598 DL4H course, aimed at reproducing the study from the paper "Do We Still Need Clinical Language Models?" by Eric Lehman et al. Specifically, it focuses on reproducing the results of MedNLI and extending the model to a new task for radiology reports using the MIMIC-CXR dataset.

## Requirements

To install requirements:

```bash
conda env create -f env.yml
```

## Training
To train the model for MedNLI, use:

```bash
./scripts/finetune_mednli.sh 1.0 8 4 10 1e-4 256 false
```

To train the model for chest X-ray report generation, run:

```bash
./scripts/finetune_cxr.sh 0.25 4 8 2 1.5e-5 50 256 false
```

### Training Parameters

- **Sample Fraction**: The fraction of the dataset to use for training (e.g., `0.25` for 25%).
- **Batch Size**: The number of training examples processed in one iteration (e.g., `4`).
- **Gradient Accumulation Steps**: Number of batches to accumulate gradients before updating model weights (e.g., `8`).
- **Epochs**: The number of times the model will go through the entire dataset (e.g., `2`).
- **Learning Rate**: Controls how much to change the model in response to the estimated error (e.g., `1.5e-5`).
- **Warmup Steps**: Number of iterations to gradually increase the learning rate from zero (e.g., `50`).
- **Max Sequence Length**: The maximum length of input sequences (e.g., `256`).
- **FP16**: Indicates whether to use mixed precision training (16-bit). Set to `false` for full precision.

## Evaluation
To evaluate the models, you can check the output files generated during training, which include metrics and predictions.

## Pre-trained Model
You can download trained models [here](https://uillinoisedu-my.sharepoint.com/my?id=%2Fpersonal%2Fsyso2%5Fillinois%5Fedu%2FDocuments%2FCS598DLH%2DProject&ga=1):
- Clinical-T5-Large model was trained on MedNLI and MIMIC-CXR datasets.

## Results
The pre-trained model achieve the following performance on clinical tasks:

### Classification on MedNLI

- **Dataset Size**: 11,000+ training examples

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Clinical-T5-Large	| 87.2% | 74.5% |

### Chest X-ray Report Generation

- **Dataset Size**: 10,000 training examples

| Model | ROUGE-L | ROUGE-1 |
|-------|---------|---------|
| Clinical-T5-Large |	53.46 |	62.87 |

## Directory Structure

```mipsasm
project_root/
├── data/
│   ├── cxr/
│   │   ├── mimic-cxr.csv        # Raw CXR data
│   │   ├── reduced-cxr.csv      # Sampled version (optional)
│   │   ├── train.jsonl          # Processed training data
│   │   ├── val.jsonl            # Processed validation data
│   │   └── test.jsonl           # Processed test data
│   └── mednli/
│       ├── mli_train_v1.jsonl   # Original MedNLI training
│       ├── mli_dev_v1.jsonl     # Original MedNLI validation
│       ├── mli_test_v1.jsonl    # Original MedNLI test
│       └── processed/           # Processed versions
│           ├── train.jsonl
│           ├── val.jsonl
│           └── test.jsonl
│
├── model/
│   └── clinical-t5-large/       # Pretrained model files
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── output/
│   ├── cxr/                     # CXR training outputs
│   │   ├── best_model/          # Saved model
│   │   └── results.json         # Evaluation metrics
│   └── mednli/                  # MedNLI training outputs
│       ├── mednli-prediction.txt
│       ├── mednli-label.txt
│       ├── mednli-scores.json
│       └── checkpoint-*/        # Training checkpoints
│
├── scripts/
│   ├── finetune_cxr.sh          # CXR training script
│   └── finetune_mednli.sh       # MedNLI training script
│
└── src/
    ├── preprocess/
    │   ├── preprocess_mimic_cxr.py
    │   └── preprocess_mednli.py
    ├── utils/
    │   └── convert_deid_tags.py
    ├── finetune_cxr.py
    └── finetune_mednli.py
```

## Contributing & License

N/A
