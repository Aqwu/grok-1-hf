import numpy as np
import torch
import jax
from tqdm import tqdm
from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

from modeling_grok import GrokForCausalLM
from configuration_grok import GrokConfig

print(f"GrokConfig ...")
config = GrokConfig()

print(f"GrokForCausalLM ...")
model = GrokForCausalLM(config)

config = model.config

for name, param in model.named_parameters():
    print(f"Parameter name: {name}, shape: {param.size()}")
    
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

print(f"state_dict ...")
own_state = model.state_dict()

CKPT_PATH = "./checkpoints"

print(f"LanguageModelConfig ...")
grok_1_model = LanguageModelConfig(
    vocab_size=128 * 1024,
    pad_token=0,
    eos_token=2,
    sequence_len=8192,
    embedding_init_scale=1.0,
    output_multiplier_scale=0.5773502691896257,
    embedding_multiplier_scale=78.38367176906169,
    model=TransformerConfig(
        emb_size=48 * 128,
        widening_factor=8,
        key_size=128,
        num_q_heads=48,
        num_kv_heads=8,
        num_layers=64,
        attn_output_multiplier=0.08838834764831845,
        shard_activations=True,
        # MoE.
        num_experts=8,
        num_selected_experts=2,
        # Activation sharding.
        data_axis="data",
        model_axis="model",
    ),
)

runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path=CKPT_PATH,
)

dummy_data = dict(
    inputs=np.zeros((1, 256), dtype=np.int32),
    targets=np.zeros((1, 256), dtype=np.int32),
)

print(f"load_or_init ...")
runner.transform_forward = True
runner.initialize(dummy_data, (1, 1), (1, 1))
params = runner.load_or_init(dummy_data)

print(f"load_or_init ...")
keys = list(params.keys())
for key in tqdm(keys):
    new_key = key.replace('/', '.').replace('decoder_layer_', 'decoder_layer.').replace('language_model', 'transformer')
    new_key += '.weight'
    print(f"{key} -> {new_key}.")
    v = list(params[key].values())[0]
    if hasattr(v , 'scales'):
        dtype = torch.float32 if v.scales.dtype == np.float32 else torch.bfloat16
        weight = torch.from_numpy(np.asarray(v.weight).astype(np.float32)).to(dtype)
        scale =torch.from_numpy(np.asarray(v.scales).astype(np.float32)).to(dtype)
        if len(scale.shape) >= 2 and scale.shape[-2] != 1:
            scale = scale[..., None, :]
            weight = weight.view(*weight.shape[:-2], 8, -1, weight.shape[-1])
            weight = (weight * scale).view(*weight.shape[:-3], -1, weight.shape[-1])
        else:
            weight = weight * scale
    else:
        dtype = torch.float32 if v.dtype == np.float32 else torch.bfloat16
        weight = torch.from_numpy(np.asarray(v).astype(np.float32)).to(dtype)

    if len(weight.shape) >= 2 and 'in_out_embed' not in new_key:
        weight = weight.transpose(-1, -2).contiguous()

    if 'moe' not in new_key:
        if new_key in own_state:
            own_state[new_key].copy_(weight)
        else:
            print(f"Skipped updating parameter1 {new_key} as it's not in the model.")
    else:
        for i in range(8):
            new_key_i = new_key.replace('moe', f'moe.{i}')
            if new_key_i in own_state:
                own_state[new_key_i].copy_(weight[i])
            else:
                print(f"Skipped updating parameter2 {new_key_i} as it's not in the model.")

    del params[key]

print(f"load_state_dict ...")
model.load_state_dict(own_state)

print(f"model.half ...")
model.half()

print(f"config.save_pretrained")
config.save_pretrained('mygrok')
print(f"model.save_pretrained")
model.save_pretrained('mygrok')
print(f"model.save_pretrained: ok")
