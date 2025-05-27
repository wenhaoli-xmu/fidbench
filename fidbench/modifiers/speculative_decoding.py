from ..modifier import Modifier
import torch
import torch.nn.functional as F

class SpeculativeDecoding(Modifier):

    def __init__(self, model, save_ckp=None, load_ckp=None, config=None):
        super().__init__(model, save_ckp, load_ckp)
        
        # 从配置文件读取参数
        if config is not None:
            self.get_conf(config)
            self.draft_model_path = self.conf.get('draft_model_path', None)
            self.gamma = self.conf.get('gamma', 4)  # 每次推测的token数量
            self.temperature = self.conf.get('temperature', 1.0)
        else:
            # 默认参数
            self.draft_model_path = None
            self.gamma = 4
            self.temperature = 1.0
        
        # 加载draft model（小模型）
        self.draft_model = self._load_draft_model()
        
    def _load_draft_model(self):
        """加载draft模型（小模型用于生成候选token）"""
        if self.draft_model_path is None:
            # 如果没有指定draft模型，使用主模型本身（这不是最优的，但可以作为演示）
            return self.model
        
        from transformers import AutoModelForCausalLM
        draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        draft_model.eval()
        return draft_model

    def ft_params(self):
        """不需要微调参数"""
        return []

    def reset(self):
        """重置方法"""
        pass

    @torch.no_grad()
    def speculative_sampling(self, input_ids, kv_cache=None, draft_kv_cache=None):
        """
        执行一轮speculative decoding
        返回: 接受的token数量, 新的token, 更新后的kv_cache
        """
        device = input_ids.device
        
        # Step 1: 使用draft模型生成gamma个候选token
        draft_tokens = []
        draft_probs = []
        current_input = input_ids
        temp_draft_kv = draft_kv_cache
        
        for i in range(self.gamma):
            # Draft模型前向传播
            draft_output = self.draft_model(
                input_ids=current_input[:, -1:] if i > 0 else current_input,
                past_key_values=temp_draft_kv,
                use_cache=True
            )
            
            draft_logits = draft_output.logits[:, -1, :] / self.temperature
            draft_prob = F.softmax(draft_logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(draft_prob, num_samples=1)  # shape: [batch_size, 1]
            draft_tokens.append(next_token)
            draft_probs.append(draft_prob)
            
            current_input = torch.cat([current_input, next_token], dim=1)
            temp_draft_kv = draft_output.past_key_values
        
        # Step 2: 使用target模型（主模型）验证候选序列
        # 构建验证序列：原始input + draft tokens
        verify_input = torch.cat([input_ids] + draft_tokens, dim=1)
        
        # Target模型前向传播
        target_output = self.model(
            input_ids=verify_input,
            past_key_values=kv_cache,
            use_cache=True
        )
        
        target_logits = target_output.logits[:, -(self.gamma + 1):, :] / self.temperature
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Step 3: 接受/拒绝采样
        accepted_tokens = []
        accepted_count = 0
        
        for i in range(self.gamma):
            draft_token = draft_tokens[i].item()
            draft_prob_i = draft_probs[i][0, draft_token]
            target_prob_i = target_probs[0, i, draft_token]
            
            # 计算接受概率
            accept_prob = min(1.0, target_prob_i / draft_prob_i)
            
            if torch.rand(1).item() < accept_prob:
                # 接受这个token
                accepted_tokens.append(draft_tokens[i])
                accepted_count += 1
            else:
                # 拒绝，需要重新采样
                # 计算修正后的概率分布
                adjusted_prob = torch.clamp(target_probs[0, i] - draft_probs[i], min=0)
                adjusted_prob = adjusted_prob / adjusted_prob.sum()
                
                # 从修正分布中采样
                new_token = torch.multinomial(adjusted_prob, num_samples=1)  # 确保是[1, 1]
                accepted_tokens.append(new_token)
                accepted_count += 1
                break
        
        # 如果所有draft tokens都被接受，从target模型额外采样一个token
        if accepted_count == self.gamma:
            extra_prob = target_probs[0, -1]
            extra_token = torch.multinomial(extra_prob.unsqueeze(0), num_samples=1)  # 确保是[1, 1]
            accepted_tokens.append(extra_token)
            accepted_count += 1
        
        # 构建最终序列和更新kv_cache
        if accepted_tokens:
            # 确保所有tokens都是2D张量 [batch_size, 1]
            normalized_tokens = []
            for token in accepted_tokens:
                if token.dim() == 1:
                    # 如果是1D，添加batch维度
                    token = token.unsqueeze(0)
                if token.shape[1] != 1:
                    # 如果第二维不是1，reshape
                    token = token.view(1, -1)
                normalized_tokens.append(token)
            
            final_tokens = torch.cat(normalized_tokens, dim=1)
            # 重新计算kv_cache（简化版本，实际应该更精确地处理）
            final_input = torch.cat([input_ids, final_tokens], dim=1)
            final_output = self.model(input_ids=final_input, use_cache=True)
            new_kv_cache = final_output.past_key_values
        else:
            final_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
            new_kv_cache = kv_cache
        
        return accepted_count, final_tokens, new_kv_cache

    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):
        """
        使用speculative decoding计算准确率
        """
        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        # pre-fill阶段
        output = self.model(input_ids=p_ids, use_cache=True)
        kv_cache = output.past_key_values
        
        # 为draft模型初始化kv_cache
        if self.draft_model != self.model:
            draft_output = self.draft_model(input_ids=p_ids, use_cache=True)
            draft_kv_cache = draft_output.past_key_values
        else:
            draft_kv_cache = kv_cache

        acc1, acc5 = 0, 0
        total_generated = 0
        total_accepted = 0
        current_pos = 0
        
        # 使用speculative decoding逐步生成
        while current_pos < g_ids.shape[-1] - 1:
            # 获取当前需要匹配的标签序列
            remaining_labels = g_ids[:, current_pos + 1:]
            
            if remaining_labels.shape[-1] == 0:
                break
                
            # 执行speculative decoding
            accepted_count, generated_tokens, kv_cache = self.speculative_sampling(
                input_ids=p_ids if current_pos == 0 else torch.cat([p_ids, g_ids[:, 1:current_pos+1]], dim=1),
                kv_cache=kv_cache,
                draft_kv_cache=draft_kv_cache
            )
            
            if generated_tokens.shape[-1] == 0:
                break
                
            # 计算准确率
            for i, generated_token in enumerate(generated_tokens[0]):
                if current_pos + i < g_ids.shape[-1] - 1:
                    label = g_ids[0, current_pos + i + 1].item()
                    
                    # Top-1准确率
                    acc1 += (generated_token.item() == label)
                    
                    # Top-5准确率（简化版本，这里只能检查生成的token）
                    acc5 += (generated_token.item() == label)
                    
                    total_generated += 1
            
            total_accepted += accepted_count
            current_pos += generated_tokens.shape[-1]
        
        if total_generated > 0:
            acc1 /= total_generated
            acc5 /= total_generated
        
        # 打印一些统计信息
        acceptance_rate = total_accepted / max(total_generated * self.gamma, 1)
        print(f"Acceptance rate: {acceptance_rate:.3f}")
        
        return acc1, acc5