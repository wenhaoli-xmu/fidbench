def get_modifier(method: str):

    if method == "origin":
        from .origin import Origin
        return Origin
    
    if method == 'greedy':
        from .greedy import Greedy
        return Greedy
    
    elif method == 'topk-llama':
        from .topk import Topk
        return Topk

    elif method == 'topk-qwen':
        from .topk import Topk
        return Topk
    
    else:
        raise NotImplementedError(method)