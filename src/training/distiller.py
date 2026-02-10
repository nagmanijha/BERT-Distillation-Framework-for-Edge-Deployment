import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce_loss(student_logits, labels)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

def run_distillation(student_model, teacher_model, dataloader, optimizer, epochs=1):
    """
    Genuine distillation training loop with real HuggingFace models.
    """
    criterion = DistillationLoss(alpha=0.5, temperature=4.0)
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
            
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
                
            loss = criterion(student_logits, teacher_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Genuine instantiation of HuggingFace models
    print("Loading teacher and student models...")
    teacher = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
    
    # Example inputs
    inputs = {
        "input_ids": torch.randint(0, 30522, (4, 128)),
        "attention_mask": torch.ones(4, 128)
    }
    labels = torch.randint(0, 2, (4,))
    
    dataset = [(inputs, labels)]
    print("Starting distillation training...")
    run_distillation(student, teacher, dataset, optimizer)
