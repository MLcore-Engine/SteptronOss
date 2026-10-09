"""Shard the HF model once per decoder block; only rank zero loads base weights."""

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from steptronoss.optimizer.fsdp2_gradient_manager import require_fsdp2_version


def reshard_for_checkpoint(model):
    """Restore DTensor parameters after a forward that did not run backward.

    Call only between training steps, when no pending graph needs the gathered
    adapter parameters. FSDPModule.reshard() is not recursive.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def build_fsdp2_qwen(cfg, device=None):
    require_fsdp2_version()
    if cfg.tuner != "lora":
        raise ValueError("The FSDP2 Qwen experiment currently supports LoRA only")
    if not torch.distributed.is_initialized():
        raise RuntimeError("Initialize the process group before building FSDP2")
    device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    model = cfg.build_model(meta_init=rank != 0)
    hf_model = model.hf_model
    # Retain CPU references before fully_shard replaces Parameter storage. This
    # avoids loading a full 27B CPU replica in every worker on an 8-GPU host.
    full_state = hf_model.state_dict() if rank == 0 else {}
    buffers = {name: b.detach().cpu().clone() for name, b in hf_model.named_buffers()} if rank == 0 else {}
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("fsdp",))
    base = hf_model.get_base_model()
    for block in base.model.language_model.layers:
        # Older FSDP2 versions require a uniform original dtype per group.
        # Group the FP32 LoRA leaf modules separately from the BF16 frozen
        # decoder. Keep the small adapter group unsharded through forward to
        # avoid an all-gather on every tiny A/B matrix invocation.
        adapters = []
        for module in block.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                adapters.extend(module.lora_A.values())
                adapters.extend(module.lora_B.values())
        if adapters:
            fully_shard(adapters, mesh=mesh, reshard_after_forward=False)
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    # The root includes the embedding, head and unused frozen vision tower.
    # Keep the vision tower frozen and never execute it for text-only SFT.
    fully_shard(hf_model, mesh=mesh, reshard_after_forward=True)
    if rank != 0:
        hf_model.to_empty(device=device)
    set_model_state_dict(
        hf_model,
        full_state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )
    del full_state
    # Non-persistent RoPE buffers are absent from state_dict and must not be
    # left uninitialized after meta -> to_empty. Preserve rank-zero dtypes too.
    for name, _target in list(hf_model.named_buffers()):
        metadata = [(tuple(buffers[name].shape), buffers[name].dtype) if rank == 0 else None]
        torch.distributed.broadcast_object_list(metadata, src=0)
        shape, dtype = metadata[0]
        value = buffers[name].to(device) if rank == 0 else torch.empty(shape, dtype=dtype, device=device)
        torch.distributed.broadcast(value, src=0)
        owner, _, key = name.rpartition(".")
        hf_model.get_submodule(owner)._buffers[key] = value
    model.fsdp_mesh = mesh
    model.checkpoint_identity = dict(model.checkpoint_identity, format="steptron-qwen38-fsdp2-v1", world_size=world)
    return model
