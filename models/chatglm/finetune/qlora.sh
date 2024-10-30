#! /bin/bash

MODEL="THUDM/chatglm-6b"
DATA="data/test_a.jsonl"
OUTPUT_DIR="output/test"

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -o | --output )
            shift
            OUTPUT_DIR=$1
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

python tinetune/train_qlora.py \
    --model_name_or_path "$MODEL" \
    --train_data_path "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_input_length 1024 \
    --max_output_length 1300
