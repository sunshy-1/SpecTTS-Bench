import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):

    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)

def extract_answer(output):
    pass

@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]
    cot_model_flag = kwargs.get('cot_model_flag')
    think_twice = kwargs.get('think_twice')
    BON= kwargs.get('BON')
    del kwargs['cot_model_flag']
    del kwargs['think_twice']
    del kwargs['BON']

    patch_select = os.environ.get('MODEL_PATCH', '').lower()

    ## ThinkTwice
    if think_twice and cot_model_flag:
        print(f"Reasoning under the \033[1m\033[94mThink Twice\033[0m framework (\033[1m\033[31mT={t}\033[0m) ..." if (t := kwargs.get('temperature')) is not None else None)
        for _ in range(3):
            torch.manual_seed(0)
            messages = []
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(2):
                
                qs = "WARMUPFLAG " + question["turns"][0]
                if j == 1:
                    messages = []
                    if patch_select == 'qwen3':
                        pre_round_answer = output.rsplit('</think>', 1)[-1] if '</think>' in output else ""
                    else:
                        pre_round_answer = output.rsplit('\n</think>\n', 1)[-1] if '\n</think>\n' in output else ""
                    think_twice_prompt = f"\nPreviously I tried to solve the problem. Now I need to double check and verify my previous answer and re-answer the question. My previous answer is: <answer> {pre_round_answer} </answer>, and please re-answer." if pre_round_answer else ""
                    messages.append({
                        "role": "user",
                        "content": qs + think_twice_prompt
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": qs
                    })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                inputs = tokenizer([prompt],add_special_tokens=False,return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]
                    # be consistent with the template's stop_token_ids
                    stop_token_ids = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    if stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                
        print('Warmup done!')

        accept_lengths_tree = []
        for question in tqdm(questions):     
            choices = []
            for i in range(num_choices):
                cur_accept_lengths_tree = []
                torch.manual_seed(i)
                messages = []
                turns = []
                steps = []
                new_tokens = []
                wall_time = []
                
                for j in range(2):
                    qs = question["turns"][0]
                    if j == 1:
                        messages = []
                        if patch_select == 'qwen3':
                            pre_round_answer = output.rsplit('</think>', 1)[-1] if '</think>' in output else ""
                        else:
                            pre_round_answer = output.rsplit('\n</think>\n', 1)[-1] if '\n</think>\n' in output else ""
                        think_twice_prompt = f"\nPreviously I tried to solve the problem. Now I need to double check and verify my previous answer and re-answer the question. My previous answer is: <answer> {pre_round_answer} </answer>, and please re-answer." if pre_round_answer else ""
                        messages.append({
                            "role": "user",
                            "content": qs + think_twice_prompt
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": qs
                        })

                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer([prompt], add_special_tokens=False,return_tensors="pt").to("cuda")
                    input_ids = inputs.input_ids
                    
                    try:
                        torch.cuda.synchronize()
                        start_time = time.time()
                        output_ids, new_token, step, accept_length_tree = forward_func(
                            inputs,
                            model,
                            tokenizer,
                            max_new_tokens,
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                        total_time = time.time() - start_time
                        accept_lengths_tree.extend(accept_length_tree)
                        output_ids = output_ids[0][len(input_ids[0]):]
                        # be consistent with the template's stop_token_ids
                        stop_token_ids = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]

                        if stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]
                                
                        output = tokenizer.decode(
                            output_ids,
                            spaces_between_special_tokens=False,
                        )
                        
                        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()

                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"

                    turns.append(output)
                    steps.append(int(step))
                    new_tokens.append(int(new_token))
                    wall_time.append(total_time)
                    cur_accept_lengths_tree.extend(accept_length_tree)

                # torch.cuda.empty_cache()
                choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))

    ## BoN (N=4)
    elif BON and cot_model_flag:
        print(f"Reasoning under the \033[1m\033[94mBoN (N=4)\033[0m framework (\033[1m\033[31mT={t}\033[0m) ..." if (t := kwargs.get('temperature')) is not None else None)
        for _ in range(3):
            torch.manual_seed(0)
            messages = []
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(4):
                qs = "WARMUPFLAG " + question["turns"][0]
                
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                inputs = tokenizer([prompt],add_special_tokens=False,return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]
                    # be consistent with the template's stop_token_ids
                    stop_token_ids = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    if stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                
        print('Warmup done!')

        accept_lengths_tree = []
        for question in tqdm(questions):         
            choices = []
            for i in range(num_choices):
                cur_accept_lengths_tree = []
                torch.manual_seed(i)
                messages = []
                turns = []
                steps = []
                new_tokens = []
                wall_time = []
                
                for j in range(4):
                    qs = question["turns"][0]
                    
                    messages.append({
                        "role": "user",
                        "content": qs
                    })

                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer([prompt], add_special_tokens=False,return_tensors="pt").to("cuda")
                    input_ids = inputs.input_ids
                    
                    try:
                        torch.cuda.synchronize()
                        start_time = time.time()
                        output_ids, new_token, step, accept_length_tree = forward_func(
                            inputs,
                            model,
                            tokenizer,
                            max_new_tokens,
                            **kwargs,
                        )
                        torch.cuda.synchronize()
                        total_time = time.time() - start_time
                        accept_lengths_tree.extend(accept_length_tree)
                        output_ids = output_ids[0][len(input_ids[0]):]
                        # be consistent with the template's stop_token_ids
                        stop_token_ids = [
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]

                        if stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]
                                
                        output = tokenizer.decode(
                            output_ids,
                            spaces_between_special_tokens=False,
                        )
                        
                        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()

                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"

                    turns.append(output)
                    steps.append(int(step))
                    new_tokens.append(int(new_token))
                    wall_time.append(total_time)
                    cur_accept_lengths_tree.extend(accept_length_tree)

                # torch.cuda.empty_cache()
                choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                                "accept_lengths": cur_accept_lengths_tree})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
