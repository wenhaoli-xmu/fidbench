def get_modifier(method: str):

    if method == "origin":
        from .origin import Origin
        return Origin
    
    elif method == 'topk-llama':
        from .topk import Topk
        return Topk

    elif method == 'topk-qwen':
        from .topk import Topk
        return Topk
    
    elif method == 'qInt4bit':
        from.quantization import QuantizedInference
        return QuantizedInference
    
    elif method == 'speculativeDecoding':
        from.speculative_decoding import SpeculativeDecoding
        return SpeculativeDecoding

    else:
        raise NotImplementedError(method)