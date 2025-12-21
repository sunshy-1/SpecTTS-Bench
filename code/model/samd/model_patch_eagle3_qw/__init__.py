from .qwen3 import qwen3_patch_dict, qwen3_attn_patch_dict

patch_dict = {}
attn_patch_dict = {}

patch_dict.update(qwen3_patch_dict)
attn_patch_dict.update(qwen3_attn_patch_dict)

