import os 
from transformers import GenerationMixin
from transformers.models.llama import modeling_llama 

from .decoding import greedy_search_proxy, FUNC_MAP, CONFIG_MAP

if os.environ.get('MODEL_PATCH', '').lower() == "qwen3":
    from transformers.models.qwen3 import modeling_qwen3 
    from .models import qwen3, generation_utils_qw
else:
    from .models import llama, generation_utils_ds

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch 

def config_lade(WINDOW_SIZE=None, LEVEL=None, DEBUG=None, GUESS_SET_SIZE=None, ALWAYS_FWD_ONE=None, SPLIT_FLAG=None, DIST_WORKERS=None, backend = 'nccl', USE_FLASH=None):
    if WINDOW_SIZE is not None:
        CONFIG_MAP["WINDOW_SIZE"] = WINDOW_SIZE
    if LEVEL is not None:
        CONFIG_MAP["LEVEL"] = LEVEL
    if GUESS_SET_SIZE is not None:
        CONFIG_MAP["GUESS_SET_SIZE"] = GUESS_SET_SIZE
    if ALWAYS_FWD_ONE is not None:
        CONFIG_MAP["ALWAYS_FWD_ONE"] = ALWAYS_FWD_ONE
    if DEBUG is not None:
        CONFIG_MAP["DEBUG"] = DEBUG
    if SPLIT_FLAG is not None:
        CONFIG_MAP["SPLIT_FLAG"] = SPLIT_FLAG
    if DIST_WORKERS is not None and DIST_WORKERS > 1:
        CONFIG_MAP["DIST_WORKERS"] = DIST_WORKERS
        CONFIG_MAP["LOCAL_RANK"] = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend, rank=CONFIG_MAP["LOCAL_RANK"])
        torch.cuda.set_device(CONFIG_MAP["LOCAL_RANK"])
        assert dist.get_world_size() == DIST_WORKERS, "DIST_WORKERS config should be equal to work size"
    if USE_FLASH is not None:
        CONFIG_MAP["USE_FLASH"] = USE_FLASH
    
    CONFIG_MAP["log"] = []


def augment_llama(model_choice=[]):
    if "Qwen3" in model_choice:
        modeling_qwen3.Qwen3ForCausalLM.jforward_multilevel = qwen3.jforward_multilevel
        modeling_qwen3.Qwen3Model.LlamaModeljforward = qwen3.Qwen3Modeljforward
        modeling_qwen3.Qwen3Model.j_prepare_decoder_attention_mask = qwen3.j_prepare_decoder_attention_mask  

    else:
        modeling_llama.LlamaForCausalLM.jforward_multilevel = llama.jforward_multilevel
        modeling_llama.LlamaModel.LlamaModeljforward = llama.LlamaModeljforward
        modeling_llama.LlamaModel.j_prepare_decoder_attention_mask = llama.j_prepare_decoder_attention_mask    

def augment_generate(model_choice=[]):

    if "DeepSeek" in model_choice:
        GenerationMixin.greedy_search = generation_utils_ds.greedy_search
        GenerationMixin.generate = generation_utils_ds.generate
        GenerationMixin._update_model_kwargs_for_generation = generation_utils_ds._update_model_kwargs_for_generation
        GenerationMixin._prepare_attention_mask_for_generation = generation_utils_ds._prepare_attention_mask_for_generation
        GenerationMixin._get_logits_processor = generation_utils_ds._get_logits_processor
        GenerationMixin._get_stopping_criteria = generation_utils_ds._get_stopping_criteria
        GenerationMixin._extract_past_from_model_output = generation_utils_ds._extract_past_from_model_output
    elif "Qwen3" in model_choice:
        GenerationMixin.greedy_search = generation_utils_qw.greedy_search
        GenerationMixin.generate = generation_utils_qw.generate
        GenerationMixin._update_model_kwargs_for_generation = generation_utils_qw._update_model_kwargs_for_generation
        GenerationMixin._prepare_attention_mask_for_generation = generation_utils_qw._prepare_attention_mask_for_generation
        GenerationMixin._get_logits_processor = generation_utils_qw._get_logits_processor
        GenerationMixin._get_stopping_criteria = generation_utils_qw._get_stopping_criteria
        GenerationMixin.prepare_inputs_for_generation = generation_utils_qw.prepare_inputs_for_generation
        
    FUNC_MAP["greedy_search"] = GenerationMixin.greedy_search
    GenerationMixin.greedy_search = greedy_search_proxy

    #FUNC_MAP["sample"] = GenerationMixin.sample
    #GenerationMixin.sample = sample_proxy
    
def augment_all(model_choice=[]):
    augment_llama(model_choice)
    augment_generate(model_choice)

def log_history(clear=False):
    gen = 0
    step = 0    
    if "log" in CONFIG_MAP:
        for log in CONFIG_MAP["log"]:
            gen += log[0]
            step += log[1]
    if clear:
        CONFIG_MAP["log"] = []
    print("LADE LOG - OVERALL GEN: ", gen, " STEPS: ", step, " AVG COMPRESS RATIO: ", (gen / step) if step > 0 else 0)

def save_log(log_dir):
    if "log" in CONFIG_MAP:
        torch.save(CONFIG_MAP["log"], log_dir)

def get_hf_model(model_path, quant, dtype, device, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
    model_config = AutoConfig.from_pretrained(model_path)
    assert quant is None or len(quant) == 0

    model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device, cache_dir=cache_dir if len(cache_dir) > 0 else None)
    model = model.eval()
    model.tokenizer = tokenizer
    
    return model, tokenizer

def get_model(model_path, quant, dtype, device, cache_dir, use_ds, native_offload = False):
    return get_hf_model(model_path, quant, dtype, device, cache_dir)