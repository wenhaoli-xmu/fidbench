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

   class XXXMethod(Modifier):

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

      # 这个必须有，输入为p_ids (prefilling tokens) 以及 g_ids (generation tokens)，分别对应上下文，以及base model生成的那部分
      @torch.no_grad()
      def compute_accuracy(self, p_ids, g_ids):

         assert p_ids.shape[0] == 1, 'only support batch size 1'
         assert p_ids.ndim == 2 and g_ids.ndim == 2

         device = next(iter(self.model.parameters())).device
         p_ids, g_ids = p_ids.to(device), g_ids.to(device)

         # pre-fill, 可能要变动
         output = self.model(input_ids=p_ids)
         kv_cache = output.past_key_values

         acc1, acc5 = 0, 0
         turns = g_ids.shape[-1] - 1

         for tok, label in zip(
                  torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                  torch.chunk(g_ids[:, 1:], turns, dim=-1)):

               # generation, 可能要变动
               output = self.model(input_ids=tok, kv_cache=kv_cache)
               logits, kv_cache = output.logits, output.past_key_values

               label = label.ravel().item()
               next_1 = logits.argmax(dim=-1).ravel().item()
               next_5 = logits.topk(k=5, dim=-1).indices.ravel().tolist()

               acc1 += next_1 == label
               acc5 += label in next_5

         acc1 /= turns
         acc5 /= turns

         return acc1, acc5
   ```


2. 修改`fidbench/modifiers/__init__.py`, 注册XXXMethod类
   ```python
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

    # 添加在了这里!!!
    elif method == 'xxxmethod':
        from .xxxmethod import XXXMethod
        return XXXMethod
    
    else:
        raise NotImplementedError(method)
   ```

3. 在runs文件夹中添加对应的运行配置文件 `llama3.1-8b-instruct-xxxmethod.json`

   注意名字必须是 `{model_name}-{method}.json`格式的，方便批处理
   ```json
   {
      "model": {
         "model_name": "/mnt/petrelfs/share_data/ai4good_shared/models/meta-llama/Llama-3.1-8B-Instruct",
         "model_dtype": "bf16",
         "model_method": "xxxmethod", # 对应上一步的注册
         "save_ckp": null,
         "load_ckp": null,
         "config": "config/xxxmethod.json" | null, # 这个是配置文件，如果没有配置，则设置为null
         "device_map": null,
         "max_length": 131072
      }
   }
   ```

4. 运行

   在`pred.sh`中添加xxxmethod方法
   ```bash
      bases=(
      "llama3.1-8b-instruct"
   )

   methods=(
      "topk"
      "xxxmethod" # 新增加
   )
   ```