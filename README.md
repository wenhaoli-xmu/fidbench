# Fidlity Bench


## 🚀 Quick Start
```
cd fidbench
pip install -e .
pip install -r requirement.txt
```

## 添加新压缩方法

1. 第一步，在`fidbench/modifiers/`下添加对应方法的文件

   需要继承`Modifier`类，且必须在init方法中拥有4个输入参数
   * model, 模型本身
   * save_ckp, load_ckp，这两个不用管
   * config, 配置文件，dict字典，自动给值，对应config文件夹中的配置文件

   例如：
   ```python
   from ..modifier import Modifier
   import torch

   class GreedyGeneration(Modifier):

      def __init__(self, model, save_ckp=None, load_ckp=None, config=None):

         # 如果不需要配置文件，则不需要下面几行
         self.get_conf(config)
         a = self.conf['a']
         b = self.conf['b']
         ...

         super().__init__(model, save_ckp, load_ckp)


      # 不用管，return [] 就行
      def ft_params(self):
         return []

      # 不用管，pass就行
      def reset(self):
         pass

      # 这个必须有，输入参数必须保证至少有input_ids, max_new_tokens以及eos_token_id三个
      @torch.no_grad()
      def generate(self, input_ids, max_new_tokens=128, eos_token_id=2):

         if isinstance(input_ids, list):
               input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]

         if input_ids.ndim == 3:
               input_ids = input_ids.flatten(0,1)

         # put the tensor on to the model's device
         device = next(iter(self.model.parameters())).device
         input_ids = input_ids.to(device)

         # fake prefilling
         output = self.model(input_ids=input_ids[:, :128])
         logits, past_key_values = output.logits, output.past_key_values

         new_tok = logits.argmax(dim=-1)
         new_ids = [new_tok]

         while len(new_ids) < max_new_tokens:
               output = self.model.model(input_ids=new_tok, past_key_values=past_key_values)
               logits, past_key_values = output.logits, output.past_key_values

               new_tok = logits.argmax(dim=-1)
               if new_tok.ravel().item() in eos_token_id:
                  break

               new_ids.append(new_tok.to(input_ids.device))

         return torch.cat([input_ids, *new_ids], dim=-1)
   ```


2. 修改`fidbench/modifiers/__init__.py`, 注册 GreedyGeneration类
   ```python
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

    # 添加在了这里!!!
    elif method == 'greedy-gen':
        from .greedy_gen import GreedyGeneration
        return GreedyGeneration
    
    else:
        raise NotImplementedError(method)
   ```

3. 在runs文件夹中添加对应的运行配置文件 `llama3.1-8b-instruct-greedy_gen.json`

   注意名字必须是 `{model_name}-{method}.json`格式的，方便批处理
   ```python
   {
      "model": {
         "model_name": "/mnt/petrelfs/share_data/ai4good_shared/models/meta-llama/Llama-3.1-8B-Instruct",
         "model_dtype": "bf16",
         "model_method": "greedy-gen", # 对应上一步的注册
         "save_ckp": null,
         "load_ckp": null,
         "config": "config/greedy-gen.json", # 这个是配置文件，如果没有配置，则设置为null
         "device_map": null,
         "max_length": 131072
      },

      # 这个可以根据需要来设置，会自动被送进generation()方法中作为额外的参数
      "generation_kwargs": {}
   }
   ```

4. 运行

   在`pred.sh`中添加greedy_gen方法
   ```bash
      bases=(
      "llama3.1-8b-instruct"
   )

   methods=(
      "topk"
      "greedy_gen" # 新增加
   )
   ```