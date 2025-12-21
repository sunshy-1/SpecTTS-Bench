#!/bin/bash

################################################################################
# The following script is used to run the qwen3-4b/8b/14b on different models
# under the framework Multi-Round-Thinking Or Best-of-N with question_cot.jsonl.
################################################################################

# Qwen3-series (Recommended Environment)
# conda activate spb_q3 (torch==2.1.1+cu121, transformers==4.53.1, accelerate==1.8.1)

export PYTHONWARNINGS="ignore"
MODEL_PATH=./local_model
GPU_DEVICES=6
MODEL_SIZE=8
Eagle3_PATH=$MODEL_PATH/EAGLE3-Qwen3-${MODEL_SIZE}B
Qwen3_PATH=$MODEL_PATH/Qwen3-${MODEL_SIZE}B
Sps_Drafter_PATH=$MODEL_PATH/Qwen3-0.6B
MODEL_NAME=Qwen3-${MODEL_SIZE}B
datastore_PATH=./model/rest/datastore/datastore_chat_small.idx
GPU_DEVICES=${GPU_DEVICES}
bench_NAME="SpecTTS_Bench"
torch_dtype="float16"

## Reasoning Under Multi-Round Thinking
TEMP=0.0
MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Qwen3_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Qwen3_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Qwen3_PATH --drafter-path $Sps_Drafter_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice

# TEMP=0.6
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Qwen3_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Qwen3_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Qwen3_PATH --drafter-path $Sps_Drafter_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice

## Reasoning Under BoN
# TEMP=0.6
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Qwen3_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Qwen3_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Qwen3_PATH --drafter-path $Sps_Drafter_PATH --model-id ${MODEL_NAME}-sps-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="QWEN3" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Qwen3_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON

## Evaluation
# python ./evaluation/speed_cot.py --file-path "./data/SpecTTS_Bench/model_answer/SPEC_METHOD.jsonl" --base-path ""./data/SpecTTS_Bench/model_answer_FILES/NAIVE_GENERATE.jsonl"" --tokenizer-path "./local_model/MODEL_NAME"
# Example: python ./evaluation/speed_cot.py --file-path "./data/SpecTTS_Bench/model_answer_BON/Qwen3-8B-samd-eagle3-float16-temp-0.6.jsonl" --base-path "./data/SpecTTS_Bench/model_answer_BON/Qwen3-8B-vanilla-float16-temp-0.6.jsonl" --tokenizer-path "./local_model/Qwen3-8B"
