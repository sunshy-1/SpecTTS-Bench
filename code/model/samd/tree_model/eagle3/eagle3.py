import torch
import copy
from transformers import LlamaConfig, LlamaForCausalLM
from typing import List, Tuple, Dict

from ...samd_config import SamdConfig
from ..tree import TreeModel
from .eagle3_config import Eagle3Config
from .eagle3_model import Eagle3Model


class Eagle3(TreeModel):
    
    def __init__(self,
        config: SamdConfig,
        lm: LlamaForCausalLM,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.head: torch.nn.Linear = lm.lm_head
        self.model: Eagle3Model = Eagle3Model(
            config=Eagle3Config(**config.tree_config),
            bias=config.tree_config.get("bias", True),
            load_emb=True,
            path=config.base_model_path,
            total_tokens=config.total_token, 
            depth=config.depth, 
            top_k=config.top_k
        ).to(device=device, dtype=dtype)
        
        self.SamdConfig = config
        
        if Eagle3Config(**config.tree_config).vocab_size==Eagle3Config(**config.tree_config).draft_vocab_size:
            del self.model.d2t,self.model.t2d

        self.model.load_weight(config.tree_model_path)
        self.model.init_tree()
        
        self.accpet_tokens: torch.Tensor = None
        self.accept_hidden_states: torch.Tensor = None
    
    def reset(self):
        self.model.stable_kv = None
    
    def update(self, 
        tokens: torch.Tensor,
        last_hidden_states: torch.Tensor,
        **kwargs,
    ):
        tokens = tokens.to(self.device)
        if self.accpet_tokens is None:
            self.accpet_tokens = tokens
        else:
            self.accpet_tokens = torch.cat([self.accpet_tokens, tokens], dim=-1)
            
        if self.accept_hidden_states is None:
            self.accept_hidden_states = last_hidden_states # prefill [66, 5120 * 3]
        else:
            self.accept_hidden_states = torch.cat([self.accept_hidden_states, last_hidden_states], dim=-2)
    
    def gen_draft(self, start_token: int) -> List[int]:
        
        # only keep_length for SAM, not Eagle!
        start_token_for_cot_models = {"deepseek": 128000, "qwen3": 151644}
        cot_model_name = "qwen3" if "qwen3" in self.SamdConfig.tree_model_path.lower() else "deepseek"
        keep_length = self.SamdConfig.n_predicts
        if self.accpet_tokens[0].item() != start_token_for_cot_models[cot_model_name] and self.accpet_tokens.shape[0] > keep_length*2:
            self.accpet_tokens = self.accpet_tokens[-keep_length:]
            self.accept_hidden_states = self.accept_hidden_states[-keep_length:,:]

        start_token = torch.tensor([start_token], dtype=torch.long, device=self.device)
        accpet_tokens = torch.cat((self.accpet_tokens, start_token), dim=-1)
        accept_hidden_states = self.accept_hidden_states

        self.accpet_tokens = self.accept_hidden_states = None

        pred_ids, buffers_kwargs = self.model.topk_genrate(
            accept_hidden_states,
            accpet_tokens,
            self.head,
        )
        pred_ids = pred_ids.view(-1).tolist()
        return pred_ids, buffers_kwargs
    
    def gen_buffers(self):
        return {
            "tree_attn_mask": None,
            "tree_position_ids": None,
            "tree_retrieve_indices": None,
        }
