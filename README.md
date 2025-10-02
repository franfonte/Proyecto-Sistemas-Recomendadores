# Proyecto-Sistemas-Recomendadores

This project measures the carbon footprint of Recommender Systems using the CodeCarbon library.

## Project Structure

```
.
├── README.md
├── requirements.txt         # Python dependencies
├── run_experiment.py        # Main script to run experiments with carbon tracking
├── data/                    # Directory for datasets
└── models/                  # Directory for model scripts
    └── svd_model.py        # Sample SVD recommender model
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### List available models

```bash
python run_experiment.py --list
```

### Run a model with carbon tracking

```bash
python run_experiment.py --model svd_model
```

### Custom models directory

```bash
python run_experiment.py --model my_model --models-dir custom_models/
```

## Features

- **Carbon Footprint Tracking**: Uses CodeCarbon to measure CO2 emissions during model execution
- **Flexible Model System**: Easy to add new models by placing Python scripts in the `models/` directory
- **Detailed Reports**: Prints comprehensive emission measurements after each experiment
- **Subprocess Execution**: Models run in isolated subprocesses for accurate measurement

## Adding New Models

To add a new recommender system model:

1. Create a new Python script in the `models/` directory (e.g., `my_model.py`)
2. Implement your model logic in the script
3. Make sure the script can be executed standalone
4. Run it with: `python run_experiment.py --model my_model`

## Dependencies

- **pandas**: Data manipulation and analysis
- **codecarbon**: Carbon emissions tracking
- **psutil**: System and process utilities