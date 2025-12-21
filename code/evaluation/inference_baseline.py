import argparse
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.eagle3.ea_model import EaModel

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    input_ids = inputs.input_ids

    if hasattr(model, 'base_model_name_or_path') and "DeepSeek" in model.base_model_name_or_path:
        output_ids, new_token, step = model.naivegenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            repetition_penalty=0.0,
            max_new_tokens=max_new_tokens,
            log=True,
            is_llama3=True,
        )

        accept_length_list = [1] * step
        return output_ids, new_token, step, accept_length_list

    elif hasattr(model, 'base_model_name_or_path') and "Qwen3" in model.base_model_name_or_path:
        output_ids, new_token, step = model.naivegenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            repetition_penalty=0.0,
            max_new_tokens=max_new_tokens,
            log=True,
        )

        accept_length_list = [1] * step
        return output_ids, new_token, step, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
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
    parser.add_argument("--ea-model-path", type=str, default="down_checkpoints/LC70B", help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    args = parser.parse_args()

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

    if cot_model_flag:
        model = EaModel.from_pretrained(
            base_model_path=args.model_path,
            ea_model_path=args.ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        tokenizer = model.get_tokenizer()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False
    
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        cot_model_flag=cot_model_flag,
        think_twice=args.think_twice,
        BON=args.BON,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)
