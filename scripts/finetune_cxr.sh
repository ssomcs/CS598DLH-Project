# Defaults
# MIMIC_CXR_CSV="data/cxr/mimic-cxr.csv"
MIMIC_CXR_CSV="data/cxr/reduced-cxr.csv"
DATA_DIR="data/cxr"
OUTPUT_DIR="output/cxr"
MODEL_PATH="model/clinical-t5-large/"

# Configurable parameters
SAMPLE_FRACTION=${1:-0.01}
BATCH_SIZE=${2:-8}
GRAD_ACC_STEPS=${3:-4}
EPOCHS=${4:-3}
LEARNING_RATE=${5:-1e-5}
WARMUP_STEPS=${6:-100}
MAX_SEQ_LENGTH=${7:-256}
FP16=${8:-true}

# Preprocess
echo "Preprocessing ${SAMPLE_FRACTION} of data..."
python src/preprocess/preprocess_mimic_cxr.py \
    --input_csv "$MIMIC_CXR_CSV" \
    --output_dir "$DATA_DIR" \
    --sample_fraction "$SAMPLE_FRACTION"

# Train
echo "Starting training with:"
echo "- Batch: $BATCH_SIZE"
echo "- Grad steps: $GRAD_ACC_STEPS"
echo "- Epochs: $EPOCHS"
echo "- Seq length: $MAX_SEQ_LENGTH"
echo "- LR: $LEARNING_RATE"
echo "- Warmup: $WARMUP_STEPS"

python src/finetune_cxr.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    $( [ "$FP16" = "true" ] && echo "--fp16" )