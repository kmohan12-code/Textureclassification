This project uses a pre-trained Vision Transformer (ViT) model (google/vit-base-patch16-224) to perform texture classification on the Describable Texture Dataset (DTD), achieving high accuracy through transfer learning and data augmentation techniques.

 Dataset
Dataset: Describable Texture Dataset (DTD)

Classes: 47 texture categories (e.g., banded, blotchy, bubbly, etc.)

Structure:

markdown
Copy
Edit
dtd/
  images/
    banded/
    blotchy/
    ...
Download and extract the dataset, then place it in your Google Drive (e.g., /content/drive/MyDrive/dtd/images).

 Setup
Environment
This code is designed to run on Google Colab using a GPU.

Required Libraries
Install the required libraries:

bash
Copy
Edit
pip install torch torchvision transformers tqdm
 Model Architecture
Base Model: ViT Base (Patch16-224)

Modification: Final classification head modified to output 47 classes instead of 1000.

Pretrained: Yes, fine-tuned on DTD.

Training Pipeline
Data Augmentation & Transforms
Training:

RandomResizedCrop

RandomHorizontalFlip

ColorJitter

RandomRotation

Normalize using ViT's mean/std

Validation & Testing:

Resize → CenterCrop → Normalize

Splits
Train: 70%

Validation: 15%

Test: 15%

Hyperparameters
Epochs: 10

Batch Size: 32

Optimizer: Adam

Loss: CrossEntropyLoss

Performance
Metric	Value
Best Validation Acc	78.25%
Total Classes	47
 Evaluation
python
Copy
Edit
def evaluate(model, loader):
    ...
 Saving & Loading Model
python
Copy
Edit
# Save best model
torch.save(model.state_dict(), "best_model.pth")


