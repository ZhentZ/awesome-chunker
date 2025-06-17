import torch


# 定义一个名为Chunking的类，用于计算文本的困惑度（Perplexity, PPL）
class Chunking:
    # 类的初始化方法，接收一个预训练的语言模型和对应的分词器作为参数
    def __init__(self, model, tokenizer) -> None:
        self.model = model  # 将传入的模型赋值给类的实例属性self.model
        self.tokenizer = tokenizer  # 将传入的分词器赋值给类的实例属性self.tokenizer

    # 定义一个方法，用于计算批量文本的困惑度
    def get_ppl_batch(
            self,
            input_ids=None,  # 输入文本的token ID张量，形状为 (batch_size, sequence_length)
            attention_mask=None,  # 输入文本的注意力掩码张量，用于指示哪些位置是真实的token，哪些是填充的token
            past_key_values=None,  # 模型的历史键值对，用于缓存计算结果，提高计算效率
            return_kv=False,  # 布尔值，指示是否返回模型的历史键值对
            end=None  # 可选参数，指定计算困惑度的结束位置
    ):

        past_length = 0  # 可选参数，指定计算困惑度的结束位置
        # 如果没有指定结束位置，则将结束位置设置为输入token ID的序列长度
        if end is None:
            end = input_ids.shape[1]
        # 在不计算梯度的上下文中进行操作，以节省内存和计算资源
        with torch.no_grad():
            # 调用模型进行推理，传入输入token ID、注意力掩码和历史键值对，并开启缓存机制
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values  # 更新历史键值对

        shift_logits = response.logits[..., :-1, :].contiguous()  # 将模型输出的logits张量进行调整，去掉最后一个时间步，使其与标签长度匹配
        shift_labels = input_ids[..., past_length + 1: end].contiguous()  # 将输入的token ID张量进行调整，去掉第一个时间步，使其与调整后的logits长度匹配
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)  # 过滤出注意力掩码中有效的token位置，并将其展平为一维张量
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]  # 根据有效的token位置，从调整后的logits中提取对应的logits，并展平为二维张量
        active_labels = shift_labels.view(-1)[active]  # 根据有效的token位置，从调整后的标签中提取对应的标签，并展平为一维张量
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # 定义交叉熵损失函数，设置reduction为"none"，表示不进行损失的聚合
        loss = loss_fct(active_logits, active_labels)  # 计算有效的logits和标签之间的交叉熵损失
        res = loss  # 将损失赋值给res变量

        return (res, past_key_values) if return_kv else res  # 如果return_kv为True，则返回损失和历史键值对；否则，仅返回损失
