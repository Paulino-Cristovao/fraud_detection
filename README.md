# Project Title

## Overview
This project is designed to implement a PyTorch-based Convolutional Neural Network (CNN) for training on a specified dataset. It includes scripts for data handling, model training, and evaluation, as well as testing and containerization support.


# AI Credit Card Fraud Detection with PyTorch CNN

This repository implements a credit card fraud detection model using a PyTorch-based 1D Convolutional Neural Network (CNN). The project includes:

- **Data Download**: Uses the Kaggle API to download the Credit Card Fraud dataset.
- **Data Preprocessing**: Normalizes and prepares tabular data for training.
- **Model Training**: Trains a CNN model and displays training loss and accuracy curves.
- **Evaluation**: Calculates ROC AUC and prints a classification report.
- **Testing**: Includes tests with pytest.
- **CI/CD**: GitHub Actions workflow for linting (flake8, mypy, pylint) and testing.
- **Containerization**: Dockerfile provided to run the project in a container.
- **Dependency Management**: Uses `pyproject.toml` with Poetry.

## Setup

### 1. Kaggle API Key
Place your Kaggle API credentials in `~/.kaggle/kaggle.json`:
```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}



## Directory Structure
```
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                   # Directory for downloaded dataset
├── main.py                 # Optional helper script (can simply call train.py)
├── model.py                # Contains the PyTorch CNN model
├── train.py                # Script to download data, train model, and plot metrics
├── test_main.py            # Pytest file with tests
├── Dockerfile              # Dockerfile for containerization
├── pyproject.toml          # Dependency and configuration file (Poetry)
└── README.md               # Documentation for the project
```

## Installation
1. Clone the repository:
   ```
   git clone [repository-url]
   cd [project-name]
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Usage
To train the model, you can run the `train.py` script. If you want to use the optional helper script, you can execute `main.py`:
```
python main.py
```

## Testing
To run the tests, use pytest:
```
pytest test_main.py
```

## Docker
To build the Docker image, run:
```
docker build -t [image-name] .
```

To run the container:
```
docker run [image-name]
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.