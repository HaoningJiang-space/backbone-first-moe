# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from .modeling_qwen import SyncQwen2MoeSparseMoeBlock, Qwen2MoeMLP
from .modeling_olmoe.modeling_olmoe import (
    SyncOlmoeSparseMoeBlock,
    OlmoeMLP,
    OlmoeConfig,
)
from .model_utils import apply_rotary_pos_emb

try:
    from .modeling_mixtral.modeling_mixtral import SyncMixtralSparseMoeBlock
except ImportError:
    SyncMixtralSparseMoeBlock = None

try:
    from .modeling_deepseek_v2.modeling_deepseek_v2 import SyncDeepseekV2Moe
except ImportError:
    SyncDeepseekV2Moe = None

try:
    from .modeling_deepseek_v3.modeling_deepseek_v3 import SyncDeepseekV3MoE
except ImportError:
    SyncDeepseekV3MoE = None
