import argparse
import transformers
from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, GenerationConfig
from model.sps.decoding import assisted_decoding


def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    input_ids = inputs.input_ids
    
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.temperature = args.temperature
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.generation_config.do_sample = True if args.temperature > 0 else False
    drafter.generation_config.temperature = model.generation_config.temperature
    drafter.generation_config.top_p = model.generation_config.top_p
    drafter.generation_config.top_k = model.generation_config.top_k
    drafter.generation_config.do_sample = model.generation_config.do_sample

    if temperature:
        output_ids, idx, accept_length_list = model.generate(
            **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, temperature=temperature, top_k=0, top_p=1.0)
        new_token = len(output_ids[0][len(input_ids[0]):])
        return output_ids, new_token, idx+1, accept_length_list
    
    else:
        output_ids, idx, accept_length_list = model.generate(
            **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, top_k=0, top_p=1.0)
        new_token = len(output_ids[0][len(input_ids[0]):])
        return output_ids, new_token, idx+1, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--drafter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench", help="The name of the benchmark question set.")
    parser.add_argument("--question-begin", type=int, help="A debug option. The begin index of questions.")
    parser.add_argument("--question-end", type=int, help="A debug option. The end index of questions.")
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="The maximum number of new generated tokens.")
    parser.add_argument("--num-choices", type=int, default=1, help="How many completion choices to generate.")
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="The number of GPUs per model.")
    parser.add_argument("--num-gpus-total", type=int, default=1, help="The total number of GPUs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="The temperature for medusa sampling.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float64", "float16", "bfloat16"], help="Override the default dtype. If not set, it will use float16 on GPU.")
    parser.add_argument("--use-cot-data", action="store_true", help="Use cot data. If not set, will use False by default.")
    parser.add_argument("--think-twice", action="store_true", help="Use original deepseek forward(). If not set, will use False by default.")
    parser.add_argument("--BON", action="store_true", help="Use original deepseek forward(). If not set, will use False by default.")
    args = parser.parse_args()

    if transformers.__version__ >= '4.53.0':
        GenerationMixin._assisted_decoding = assisted_decoding
    else:
        GenerationMixin.assisted_decoding = assisted_decoding

    if args.use_cot_data:
        question_file = f"data/{args.bench_name}/question_cot.jsonl"
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.BON:
            answer_file = f"data/{args.bench_name}/model_answer_BON/{args.model_id}.jsonl"
        else:
            answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    _support_cot_models = ["DeepSeek", "Qwen3"]
    cot_model_flag = any(model in args.model_path for model in _support_cot_models)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature == 0:
        model_config_path = "./model/sps/tmp_config/Qwen3_8B_tmp_config"
        drafter_config_path = "./model/sps/tmp_config/Qwen3_0.6B_tmp_config"
        model.generation_config = GenerationConfig.from_pretrained(model_config_path)
        drafter.generation_config = GenerationConfig.from_pretrained(drafter_config_path)

    model.eval()
    drafter.eval()

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=sps_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        cot_model_flag=cot_model_flag,
        think_twice=args.think_twice,
        BON=args.BON,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)