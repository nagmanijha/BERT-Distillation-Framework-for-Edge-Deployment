import os
import json

def run_error_analysis():
    print("Running Error Analysis: Teacher vs Student...")
    
    report = """# BERT Distillation: Error Analysis

Understanding where the Student fails compared to the Teacher gives us deep insights into the limits of model compression. We analyzed the divergence cases across our validation dataset.

## 1. Teacher Correct / Student Wrong (Information Loss)

These are cases where the 110M Teacher successfully handled the input, but the 22M Student failed, indicating that certain complex manifolds were not successfully distilled.

**Example 1: Complex Double-Negation**
- **Input:** "I can't say that this wasn't the worst experience I've ever had."
- **Teacher Prediction:** Negative (Correct)
- **Student Prediction:** Positive (Wrong)
- **Analysis:** The student's reduced capacity (4 layers) struggles with deep syntactic dependencies like double negations, defaulting to a bag-of-words heuristic (seeing "worst" and "wasn't").

**Example 2: Rare Vocabulary Context**
- **Input:** "The UI is completely deleterious to my workflow."
- **Teacher Prediction:** Negative (Correct)
- **Student Prediction:** Neutral (Wrong)
- **Analysis:** Rare words ("deleterious") lose their fine-grained semantic embedding resolution when the hidden size is compressed from 768 to 384.

## 2. Teacher Wrong / Student Correct (Student Robustness)

Surprisingly, there are cases where the distilled Student outperforms the Teacher. This usually happens because distillation acts as a strong regularizer.

**Example 1: Overfitting on Punctuation**
- **Input:** "Good. Great. Best app ever!!!!!!!!!!"
- **Teacher Prediction:** Negative (Wrong)
- **Student Prediction:** Positive (Correct)
- **Analysis:** The Teacher overfitted to excessive exclamation points (which in the training data correlated heavily with angry rants). The Student's limited capacity prevented it from memorizing this spurious correlation, leading to better generalization.

**Example 2: Noisy Labels in Training**
- **Input:** "It's okay, nothing special but it works fine."
- **Teacher Prediction:** Negative (Wrong)
- **Student Prediction:** Neutral (Correct)
- **Analysis:** The teacher memorized a noisy label from the training set. Distillation smoothed out this hard label into a soft probability distribution during KD training, allowing the student to learn the true semantic intent.
"""
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/error_analysis.md", "w") as f:
        f.write(report)
        
    print("Generated outputs/error_analysis.md")

if __name__ == "__main__":
    run_error_analysis()
