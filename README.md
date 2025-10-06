# Auto Tagging Support Tickets using LLMs

##  Objective
The objective of this project is to **automatically assign relevant category tags** to incoming customer support tickets written in free text form.  
It compares three approaches:
1. **Zero-Shot Classification** using a pre-trained LLM (BART-MNLI)
2. **Few-Shot Prompt Engineering** for better contextual tagging
3. **Fine-Tuned Transformer Model** (Roberta-base setup for later extension)

---

## Methodology / Approach

### 1. Dataset
- Source: `customer_support_tickets.csv`
- Columns used:
  - `Ticket Description` → converted to `text`
  - `Ticket Type` → converted to `label`
- Preprocessing: cleaning, lowercasing, removing noise and URLs

### 2. Zero-Shot Classification
Used **`facebook/bart-large-mnli`** model via the Hugging Face `pipeline` to predict probable ticket tags without any training.

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
out = classifier(text, candidate_labels)
## 3. Key Results / Observations
Aspect:	
Zero-Shot Model	
Few-Shot Prompting	
Fine-Tuning (Future Work)	
Description:
Successfully predicted relevant tags with good accuracy for major categories
Improved precision on ambiguous tickets
Expected to increase overall F1-score through domain adaptation



