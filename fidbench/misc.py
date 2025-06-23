import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .modifiers import get_modifier


from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


def get_torch_dtype(dtype: str):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype '{dtype}'")


def get_env_conf(env_conf: str):
    import json
    with open(env_conf, 'r') as f:
        env_conf = json.load(f)
    return env_conf


def get_tokenizer(
        model_name, 
        **kwargs
):
    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get('tokenizer_name'), 
            use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True)

    return tokenizer


def get_model_and_tokenizer(
        model_name, 
        model_dtype, 
        model_method, 
        save_ckp, 
        load_ckp, 
        config, 
        device_map, 
        **kwargs
    ):

    from accelerate import dispatch_model

    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(kwargs.get('tokenizer_name'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    student_dtype = get_torch_dtype(model_dtype)
    student = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=student_dtype, 
        device_map="auto" if device_map is None else None,
        trust_remote_code=True)
    student_modifier = get_modifier(model_method)

    if student_modifier is not None:
        student = student_modifier(
            student,
            save_ckp=save_ckp,
            load_ckp=load_ckp,
            config=config)

    student.eval()

    if device_map is not None:
        student.model = dispatch_model(student.model, device_map=device_map)

    return tokenizer, student
