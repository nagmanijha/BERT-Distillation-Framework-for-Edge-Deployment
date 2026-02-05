import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class TinyBERTStudent(nn.Module):
    """
    Configurable Student Model architecture designed for Knowledge Distillation.
    Standard config: 4 layers, 384 hidden size.
    """
    def __init__(self, num_layers=4, hidden_size=384, num_labels=2):
        super().__init__()
        # Configure a custom smaller BERT
        self.config = BertConfig(
            vocab_size=30522,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
            num_labels=num_labels
        )
        self.bert = BertModel(self.config)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits, outputs.last_hidden_state

def get_student_metrics(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2) # Assume FP32 (4 bytes per param)
    return {
        "parameters": f"{total_params / 1e6:.1f}M",
        "size_mb": f"{size_mb:.1f} MB"
    }

if __name__ == "__main__":
    student = TinyBERTStudent(num_layers=4, hidden_size=384)
    print("Student Architecture Metrics:")
    print(get_student_metrics(student))
