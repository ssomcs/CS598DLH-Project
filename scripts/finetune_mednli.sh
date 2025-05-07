# Usage: ./scripts/finetune_mednli.sh 0.01 16 2 10 1e-4 256 false
# Defaults (reduced from original)
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

MEDNLI_TRAIN="data/mednli/mli_train_v1.jsonl"
MEDNLI_VAL="data/mednli/mli_dev_v1.jsonl"
MEDNLI_TEST="data/mednli/mli_test_v1.jsonl"
PROCESSED_DIR="data/mednli/processed"
OUTPUT_DIR="output/mednli/"
MODEL_PATH="model/clinical-t5-large/"

# More conservative parameters
SAMPLE_FRACTION=${1:-0.5}  # Reduced from 1.0
BATCH_SIZE=${2:-4}         # Reduced from 8
GRAD_ACC_STEPS=${3:-8}     # Increased from 4
EPOCHS=${4:-5}             # Reduced from 10
LEARNING_RATE=${5:-3e-5}   # More stable learning rate
MAX_SEQ_LENGTH=${6:-128}   # Reduced from 256

# Preprocessing
python src/preprocess/preprocess_mednli.py \
    --input_train "$MEDNLI_TRAIN" \
    --input_val "$MEDNLI_VAL" \
    --input_test "$MEDNLI_TEST" \
    --output_dir "$PROCESSED_DIR" \
    --sample_fraction "$SAMPLE_FRACTION"

# Training with memory optimizations
python src/finetune_mednli.py \
    --model-path "$MODEL_PATH" \
    --mednli-train-path "$PROCESSED_DIR/train.jsonl" \
    --mednli-val-path "$PROCESSED_DIR/val.jsonl" \
    --mednli-test-path "$PROCESSED_DIR/test.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACC_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_predict \
	--is-instruction-finetuned