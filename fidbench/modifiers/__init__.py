def assert_transformer_version(version):
    import transformers
    this_version = transformers.__version__
    assert this_version == version, f"Expect transformer=={version}, but got `{this_version}`."


def get_modifier(method: str):
    if method == "origin":
        from .origin import Origin
        return Origin

    elif method == 'sage-attention':
        assert_transformer_version("4.45.0")
        from .sage_attention import SageAttention
        return SageAttention
    
    elif method == 'flash-attention-fp8':
        assert_transformer_version("4.45.0")
        from .flash_attention_fp8 import FlashAttnentionFP8
        return FlashAttnentionFP8
    
    elif method == 'sparge-attention':
        assert_transformer_version("4.45.0")
        from .sparge_attention import SpargeAttention
        return SpargeAttention
    
    elif method == 'medusa':
        assert_transformer_version("4.37.0")
        from .medusa import Medusa
        return Medusa
    
    elif method == 'h2o':
        assert_transformer_version('4.37.2')
        from .h2o import H2O
        return H2O
    
    elif method == 'topk':
        assert_transformer_version('4.45.0')
        from .topk import Topk
        return Topk
    
    elif method == 'gear':
        assert_transformer_version('4.37.2')
        from .gear import Gear
        return Gear
    
    elif method == 'snapkv':
        assert_transformer_version('4.37.2')
        from .snapkv import SnapKV
        return SnapKV

    elif method == 'gptq':
        assert_transformer_version('4.45.0')
        from .gptq import GPTQMethod
        return GPTQMethod

    elif method == 'awq':
        assert_transformer_version('4.52.4')
        from .awq import AWQMethod
        return AWQMethod
    
    elif method == 'sparsegpt':
        assert_transformer_version('4.37.2')
        from .sparsegpt import SparseGPT
        return SparseGPT
    
    elif method == 'wanda':
        assert_transformer_version('4.37.2')
        from .wanda import Wanda
        return Wanda
    
    else:
        raise NotImplementedError(method)