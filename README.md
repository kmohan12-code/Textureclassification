
#  Texture Classification using Vision Transformer (ViT)

This project uses a **pre-trained Vision Transformer (ViT)** model (`google/vit-base-patch16-224`) to classify images from the **Describable Texture Dataset (DTD)** into 47 texture categories.  
The model achieves high accuracy through **transfer learning** and **data augmentation** techniques.

---

##  Dataset

**Name:** [Describable Texture Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)  
**Classes:** 47 texture categories (e.g., `banded`, `blotchy`, `bubbly`, etc.)

**Structure:**
```

dtd/
└── images/
├── banded/
├── blotchy/
├── bubbly/
└── ...

```

**Setup:**
1. Download and extract the dataset from the official source.
2. Place it in your Google Drive:
```

/content/drive/MyDrive/dtd/images

````

---

##  Environment Setup

This code is designed to run on **Google Colab** with **GPU** enabled.

**Required Libraries:**
```bash
pip install torch torchvision transformers tqdm
````

---

## Model Architecture

* **Base Model:** ViT Base (Patch16-224)
* **Modification:** Final classification head modified for **47 classes** instead of 1000.
* **Pretrained:** Yes, on ImageNet, then fine-tuned on DTD.

---

## Training Pipeline

### **Data Augmentation & Transforms**

**Training:**

* `RandomResizedCrop`
* `RandomHorizontalFlip`
* `ColorJitter`
* `RandomRotation`
* `Normalize` (using ViT's mean/std)

**Validation & Testing:**

* `Resize` → `CenterCrop` → `Normalize`

---

##  Data Splits

* **Train:** 70%
* **Validation:** 15%
* **Test:** 15%

---


---

##  Performance

| Metric              | Value      |
| ------------------- | ---------- |
| Best Validation Acc | **78.25%** |
| Total Classes       | 47         |

---

##  Evaluation

```python
def evaluate(model, loader):
    # Evaluation logic
    ...
```

---

## Saving & Loading Model

```python
# Save
torch.save(model.state_dict(), "vit_texture_model.pth")

# Load
model.load_state_dict(torch.load("vit_texture_model.pth"))
```

---

##  References

* [Describable Texture Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [PyTorch](https://pytorch.org/)
