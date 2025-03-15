# Pet-MultiClassification

## 🐶🐱 Multi-Class Pet Image Classification
This project is a deep learning pipeline for classifying pet images into multiple categories. The model is trained using PyTorch and follows best practices in dataset preprocessing, model training, and evaluation.

## 📌 Features
- **Multi-class classification** of pet images
- **Custom PyTorch model** with CNN architecture
- **Automated testing** using CI/CD (GitHub Actions)
- **Dataset handling** with augmentation


## 🏗️ Project Structure
```
Pet-MultiClassification/
│-- data/                     # Dataset storage
│   ├── images/
│   ├── annotations/
│-- models/                   # Trained models and checkpoints
│-- src/
│   ├── preprocess.py         # Dataset preprocessing
│   ├── train.py              # Training pipeline
│   ├── test.py               # Model testing
│-- tests/                    # Unit tests for model & data pipeline
│   ├── test_pipeline.py      # tests the whole pipeline 
│-- requirements.txt          # Dependencies
│-- README.md                 # Project documentation
│-- .github/workflows/        # CI/CD configuration
```

## 🚀 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AmrRagab0/Pet-MultiClassification.git
cd Pet-MultiClassification
```
### 2️⃣ Create a Conda Environment
```bash
conda create --name pet-classifier python=3.9
conda activate pet-classifier
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Training the Model
Run the training script with:
```bash
python src/train.py --epochs 10 
```
Options:
- `--epochs`: Number of training epochs (default: 30)

## 🧪 Running Tests
Run unit tests with:
```bash
cd tests/
python test_pipeline.py
```

## 🔄 CI/CD Pipeline
This project uses **GitHub Actions** for automated testing. On each commit, the pipeline:
1. **Runs unit tests** (`Unittest`)
2. **Ensures model can make predictions**

## 👥 Contributing
Feel free to fork the repo and submit pull requests!

1. **Fork the project**
2. **Create a feature branch** (`git checkout -b new-feature`)
3. **Commit changes** (`git commit -m 'Added feature'`)
4. **Push to GitHub** (`git push origin new-feature`)
5. **Submit a Pull Request** 🚀

## 📜 License
This project is licensed under the MIT License.

---
⚡ _Happy Coding!_

