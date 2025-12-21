import argparse
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.pia.kv_cache import initialize_past_key_values
from model.pia.modeling_llama_kv import LlamaForCausalLM
from model.pia.lookahead_cache import LookaheadCache
from model.pia.utils import *
import sys
import torch

@torch.no_grad()
def pia_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, branch_length=12, decoding_length=64, top_p=0.0, top_k=0.0):

    """
    # initialize lookahead cache and set eos
    lookahead_cache = LookaheadCache(debug=False)
    lookahead_cache.eos_ids = [tokenizer.eos_token_id]
    """
    warmup_tokens_dict = {"deepseek":[54, 18394, 3202], "qwen3":[54, 17911, 3124, 22542]}
    offset_cot_model_dict = {"deepseek":2, "qwen3":3}
    cot_model_name = "deepseek" if "deepseek" in model.name_or_path.lower() else "qwen3"
    warmup_tokens = torch.tensor(warmup_tokens_dict[cot_model_name]).to(inputs.input_ids.device) 

    # 1. first time  2. first round 3. warmup round 
    if model.pre_input_tokens is None or \
        not torch.equal(model.pre_input_tokens, inputs.input_ids[0, :10]) or \
        torch.equal(model.pre_input_tokens[offset_cot_model_dict[cot_model_name]:offset_cot_model_dict[cot_model_name]+len(warmup_tokens)], warmup_tokens):
        lookahead_cache.reset()
        
    model.pre_input_tokens = inputs.input_ids[0, :10]

    # *** Step 1: put prompt into lookahead cache
    input_id_list = inputs.input_ids[0].tolist()
    # print("write {} into lookahead cache".format(input_id_list[:-1]))
    lookahead_cache.put(input_id_list[:-1], branch_length=branch_length + 1, mode='input', idx=0)


    input_ids = inputs.input_ids.cuda()
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)    
    verify_input_ids = input_ids[:, :-1]
    next_input_id = input_ids[:, -1:]
    model.max_new_tokens = max_new_tokens

    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model)
    
    # prefilling
    model(input_ids=verify_input_ids, past_key_values=past_key_values)
    device = model.device
    eos_token_id = torch.tensor(tokenizer.eos_token_id, dtype=torch.long, device=device)
    
    def pad_to_same_length(search_path):
        if len(search_path) == 1:
            return search_path
        max_len = max(len(path) for path in search_path)
        padded_paths = [path + [-1] * (max_len - len(path)) for path in search_path]
        return padded_paths

    def lookahead_prepare_inputs(decoding_qids):
        min_input_size = 0
        min_output_size = max(decoding_length // 2, 1)

        # decoding ids: []
        # decoding masks: [n, n]
        # sizes: tuple
        # depths: []
        # search path: [[]]

        # *** Step 2: get speculative tokens
        decoding_ids, decoding_masks, sizes, depths, search_path = lookahead_cache.hier_get(decoding_qids,
                                                                            decoding_length=decoding_length,
                                                                            branch_length=branch_length,
                                                                            min_input_size=min_input_size,
                                                                            min_output_size=min_output_size,
                                                                            mode="mix",
                                                                            idx=0)
        
        search_path = pad_to_same_length(search_path)
        return decoding_ids, decoding_masks, depths, search_path
    
    _thinking_budget = int(max_new_tokens*0.8)
    
    ## Deepseek \n\n<think>\n\n
    ## Qwen3 Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n 
    if "Qwen3" in model.name_or_path:
        _end_thinking_tokens = [82796, 279, 7199, 882, 553, 279, 1196, 11, 358, 614, 311, 2968, 279, 6291, 3118, 389, 279, 7274, 5961, 1431, 624, 151668, 382][::-1]
    elif "DeepSeek" in model.name_or_path:
        _end_thinking_tokens = [271, 128014, 271][::-1]
    else:
        raise NotImplementedError("Other models are not supported yet!!!!")
    
    _thinking_add_cnt = len(_end_thinking_tokens)

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        logits_processor = None

    for i in range(max_new_tokens):
        if verify_input_ids.size(1) - start_len > _thinking_budget and _thinking_add_cnt:
            decoding_qids = [verify_input_ids[0, -1].item(), next_input_id[0].item()]
            input_ids = decoding_qids[-1:]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
  
            current_seq_len = verify_input_ids.size(1) + 1
            attention_mask = torch.ones([1, 1, 1, current_seq_len], dtype=torch.long, device=device)
            position_ids = torch.tensor([verify_input_ids.size(1)], dtype=torch.long, device=device).unsqueeze(0)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)

            accept_len = 1
            accept_length_list.append(accept_len)
            
            # update kv length
            current_length_data.fill_(current_seq_len)
            
            # concat previous token ids and accepted new token ids
            verify_input_ids = torch.cat([verify_input_ids, input_ids], dim=-1)
            
            # check eos and sequence len
            if (input_ids == eos_token_id).any() or current_seq_len - start_len > max_new_tokens:
                break
            
            if "DeepSeek" in model.name_or_path:
                stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if (stop_token_id == input_ids).any():
                    break
            
            # sample next input id, shape [1, 1]
            next_input_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            next_input_id[0][0] = _end_thinking_tokens[_thinking_add_cnt-1]

            if (next_input_id == eos_token_id).any():
                start_len -= 1
                break
            
            next_token_list = next_input_id[0].cpu().tolist()

            step += 1

            # Step 3: write new token list into lookahead cache
            lookahead_cache.stream_put(next_token_list, branch_length=branch_length + 1, final=False,
                                                    mode='output', idx=0)
            
            _thinking_add_cnt -= 1    

        else:
            decoding_qids = [verify_input_ids[0, -1].item(), next_input_id[0].item()]
            # print("decoding_qids {}".format(decoding_qids))

            input_ids, decoding_masks, depths, search_path = lookahead_prepare_inputs(decoding_qids)

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            decoding_size = len(decoding_masks)


            # print("decoding_qids {}, input_ids {}, decoding_masks {}, depths {}, search_path {}".format(decoding_qids, input_ids, decoding_masks, depths, search_path))
            # no matched n-gram
            if decoding_size == 1:
                current_seq_len = verify_input_ids.size(1) + 1
                attention_mask = torch.ones([1, 1, 1, current_seq_len], dtype=torch.long, device=device)
                position_ids = torch.tensor([verify_input_ids.size(1)], dtype=torch.long, device=device).unsqueeze(0)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)

                accept_len = 1
                accept_length_list.append(accept_len)
                
                # update kv length
                current_length_data.fill_(current_seq_len)
                
                # concat previous token ids and accepted new token ids
                verify_input_ids = torch.cat([verify_input_ids, input_ids], dim=-1)
                
                # check eos and sequence len
                if (input_ids == eos_token_id).any() or current_seq_len - start_len > max_new_tokens:
                    break
                
                if "DeepSeek" in model.name_or_path:
                    stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if (stop_token_id == input_ids).any():
                        break

                # sample next input id, shape [1, 1]
                if temperature > 1e-5:
                    logits = logits_processor(None, outputs.logits[:, -1, :])
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    next_input_id = torch.multinomial(probabilities, 1)
                else:
                    next_input_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
                if (next_input_id == eos_token_id).any():
                    start_len -= 1
                    break

                
                next_token_list = next_input_id[0].cpu().tolist()
                # print("next_token_list: {}".format(next_token_list))

            else:
                # [q, q] -> [1, q, q]
                attention_mask = torch.tensor(decoding_masks, dtype=torch.long, device=device).unsqueeze(0)
                # [q] -> [1, q]
                position_ids = torch.tensor(depths, dtype=torch.long, device=device).unsqueeze(0)
                # [1, q, k] + [1, q, q] => [1, q, k+q] => [1, 1, q, k+q]
                merge_attention_mask = torch.cat([torch.ones([1, decoding_size, verify_input_ids.size(1)], 
                                                    dtype=torch.long, device=device), attention_mask], dim=-1).unsqueeze(1)
                # [1, q] + kv_len
                merge_positon_ids = position_ids + verify_input_ids.size(1)
                # [1, n_path]
                search_path = torch.tensor(search_path, dtype=torch.long, device=device)

                # print("input_ids {}, attention_mask {}, position_ids {}".format(input_ids.shape, merge_attention_mask.shape, merge_positon_ids.shape))

                outputs = model(input_ids=input_ids, attention_mask=merge_attention_mask, position_ids=merge_positon_ids, past_key_values=past_key_values)

                if temperature > 1e-5:
                    all_input_path = input_ids[0][search_path]
                    all_input_path[search_path==-1] = -100

                    best_path_index, accept_len, sample_p = evaluate_posterior(logits=outputs.logits[0, search_path], candidates=all_input_path, logits_processor=logits_processor)

                    best_path_index = best_path_index.to(sample_p.device)
                    accept_len = torch.tensor(accept_len).to(sample_p.device) + 1
                    accept_length_list.append(accept_len.item())

                else:
                    # output_logits [b, q, d] -> [b, q]
                    model_res = torch.argmax(outputs.logits, dim=-1)

                    # get speculation path and model generation
                    all_input_path = input_ids[0][search_path]
                    all_input_path[search_path==-1] = -100
                    all_output_path = model_res[0][search_path]
                    
                    # print("search path {}, type {}".format(search_path, type(search_path)))
                    # print("all input path {}, all output path {}".format(all_input_path, all_output_path))

                    # verification step
                    reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
                    best_reward = reward.max()
                    
                    accept_len = 1 + best_reward
                    accept_length_list.append(accept_len.item())

                    # select the best speculation path
                    best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
                
                index_path = search_path[best_path_index][:accept_len]
                best_path_input = torch.index_select(input_ids, index=index_path, dim=1)

                # update kvcache
                tgt = past_key_values_data[..., verify_input_ids.size(1) + index_path, :]
                dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
                
                # update kv length
                current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])
                        
                # concat previous token ids and accepted new token ids
                verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)

                # check eos and sequence len
                if (best_path_input == eos_token_id).any() or verify_input_ids.size(1) - start_len > max_new_tokens:
                    break

                if "DeepSeek" in model.name_or_path:
                    stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if (stop_token_id == input_ids).any():
                        break
                    if (stop_token_id == best_path_input).any():
                        break

                # sample next input id, shape [1, 1]
                if temperature > 1e-5:
                    next_input_id = torch.multinomial(sample_p, 1)
                    next_input_id = next_input_id[None].to(sample_p.device)
                else:
                    next_input_id = model_res[:, search_path[best_path_index][accept_len - 1]].unsqueeze(-1).cuda()
                
                if (next_input_id == eos_token_id).any():
                    start_len -= 1
                    break
                
                next_token_list = best_path_input[0].cpu().tolist()

            # print("next_token_list {}".format(next_token_list))


            step += 1
            # Step 3: write new token list into lookahead cache
            lookahead_cache.stream_put(next_token_list, branch_length=branch_length + 1, final=False,
                                                    mode='output', idx=0)

    """
    # Step 4: reset lookahead cache
    lookahead_cache.stream_put([], branch_length=branch_length + 1, final=True,
                                    mode='output', idx=0)
    """
    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list


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
    parser.add_argument("--branch-length", type=int, default=12, help="The maximum length of one branch.")
    parser.add_argument("--decoding-length", type=int, default=64, help="The maximum decoding length.")
    parser.add_argument("--use-cot-data", action="store_true", help="Use cot data. If not set, will use False by default.")
    parser.add_argument("--think-twice", action="store_true", help="Use original deepseek forward(). If not set, will use False by default.")
    parser.add_argument("--BON", action="store_true", help="Use original deepseek forward(). If not set, will use False by default.")
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
    
    if "DeepSeek" in args.model_path:
        from model.recycling.modeling_llama_kv_ds import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    elif "Qwen3" in args.model_path:
        from model.recycling.modeling_qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    model.pre_input_tokens = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # initialize lookahead cache and set eos
    lookahead_cache = LookaheadCache(debug=False)
    lookahead_cache.eos_ids = [tokenizer.eos_token_id]

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=pia_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        cot_model_flag=cot_model_flag,
        think_twice=args.think_twice,
        BON=args.BON,
        branch_length=args.branch_length,
        decoding_length=args.decoding_length
    )

    reorg_answer_file(answer_file)