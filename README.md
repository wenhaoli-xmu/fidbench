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

   # 这个也必须有，输入为input_ids
   @torch.no_grad()
   def compute_ppl(self, input_ids):
      assert input_ids.shape[0] == 1, 'only support batch size 1'
      assert input_ids.ndim == 2

      device = next(iter(self.model.parameters())).device
      input_ids = input_ids.to(device)

      output = self.model(input_ids=input_ids)
      logits = output.logits

      shift_logits = logits[:, :-1, :].contiguous()
      shift_labels = input_ids[:, 1:].contiguous()

      loss = F.cross_entropy(
         shift_logits.view(-1, shift_logits.size(-1)),
         shift_labels.view(-1),
         reduction='mean'
      )

      ppl = torch.exp(loss).item()

      return ppl
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

4. 测试accuracy

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

5. 测试perplexity

   在`perplexity.sh`中添加xxxmethod方法
   ```bash
   bases=(
      "llama3.1-8b-instruct"
   )

   methods=(
      "topk"
      "xxxmethod" # 新增加
   )
   
   ```


## 已有方法

1. TransMLA
   ```bash
   conda create -n transmla python=3.12.8
   conda activate transmla
   ```

   ```bash
   cd thirdparty/TransMLA
   pip install -r requirements.txt
   ```

   转换模型参数
   ```
   bash scripts/convert/qwen2.5-7B-Instruct.sh
   ```

2. SageAttention2
   ```bash
   conda create -n transmla
   conda activate sageattn
   ```
   ```bash
   cd thirdparty/SageAttention
   python setup.py install
   ```

3. FlashAttention FP8
   Enviromental requiremnt: `CUDA==12.8`
   ```bash
   conda create -n transmla
   conda activate fp8
   ```
   ```
   cd thirdparty/flash-attention
   git submodule update --init csrc/cutlass
   cd hopper
   python setup.py install
   ```

4. SpargeAttention
   Environment requirements:
   `python>=3.9`, `torch>=2.3.0`, `CUDA>=12.8` for Blackwell, `>=12.4` for fp8 support on Ada, `>=12.3` for fp8 support on Hopper, `>=12.0` for Ampere
   ```bash
   conda create -n spargeattn --clone transmla
   conda activate spargeattn
   ```

   ```bash
   cd thirdparty/SpargeAttn
   pip install ninja
   python setup.py install
   ```

5. Medusa
   Data Prepare
   ```bash
   cd thirdparty/Medusa
   mkdir data
   huggingface-cli download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset --include ShareGPT_V4.3_unfiltered_cleaned_sp
lit.json
   ```

   训练
   ```
   pip install transformers==4.37.2
   mv /mnt/petrelfs/liwenhao/fidbench/thirdparty/Medusa/medusa/train/train_legacy.py /mnt/petrelfs/liwenhao/fidbench/thirdparty/Medusa/medusa/train/train.py
   ```

   medusa_model.py里面要改动 LlamaAttention里的
   ```bash
   self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
   self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
   self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
   self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
   ```

   在medusa_model.py的最末尾写死使用LLaMA结构进行读取
   ```
   class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        return MedusaModelLlama.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )
   ```

   medusa_model.py里面去掉所有的self.pretrain_tp相关的
   

6. Dejavu
   创建新的环境并安装依赖
   ```bash
   pip install cupy-cuda12x
   python -m cupyx.tools.install_library --cuda 12.x --library nccl
   ```