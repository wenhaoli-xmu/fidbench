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
    
    else:
        raise NotImplementedError(method)