#!/bin/bash

################################################################################
# The following script is used to run the deepseek-llama-8b on different models
# under the framework Multi-Round-Thinking Or Best-of-N with question_cot.jsonl.
################################################################################

# DeepSeek-R1-Distill-LLaMA-8B (Recommended Environment)
# conda activate spb (torch==2.1.1+cu121, transformers==4.43.1, accelerate==0.21.0)

export PYTHONWARNINGS="ignore"
MODEL_PATH=./local_model
GPU_DEVICES=6
MODEL_SIZE=8
Eagle3_PATH=$MODEL_PATH/EAGLE3-DeepSeek-R1-Distill-LLaMA-${MODEL_SIZE}B
DeepSeek_PATH=$MODEL_PATH/DeepSeek-R1-Distill-Llama-${MODEL_SIZE}B
MODEL_NAME=DeepSeek-R1-Distill-Llama-${MODEL_SIZE}B
datastore_PATH=./model/rest/datastore/datastore_chat_small_ds.idx
GPU_DEVICES=${GPU_DEVICES}
bench_NAME="SpecTTS_Bench"
torch_dtype="float16"

## Reasoning Under Multi-Round Thinking
# TEMP=0.0
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $DeepSeek_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice

# TEMP=0.6
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $DeepSeek_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --think-twice

## Reasoning Under BoN
# TEMP=0.6
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $DeepSeek_PATH --ea-model-path $Eagle3_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-eagle3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle3 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-samd-only-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --tree_model_path $Eagle3_PATH --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-recycling-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pia --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-pia-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON
# MODEL_PATCH="DEEPSEEK" CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $DeepSeek_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 15000 --use-cot-data --BON

## Evaluation
# python ./evaluation/speed_cot.py --file-path "./data/SpecTTS_Bench/model_answer/SPEC_METHOD.jsonl" --base-path ""./data/SpecTTS_Bench/model_answer_FILES/NAIVE_GENERATE.jsonl"" --tokenizer-path "./local_model/MODEL_NAME"
# Example: python ./evaluation/speed_cot.py --file-path "./data/SpecTTS_Bench/model_answer/DeepSeek-R1-Distill-Llama-8B-eagle3-float16-temperature-0.0.jsonl" --base-path "./data/SpecTTS_Bench/model_answer/DeepSeek-R1-Distill-Llama-8B-vanilla-float16-temp-0.0.jsonl" --tokenizer-path "./local_model/DeepSeek-R1-Distill-Llama-8B"
