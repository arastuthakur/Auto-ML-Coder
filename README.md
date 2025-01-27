# Auto ML Coder

An automated machine learning project generator that creates unique, well-structured ML projects and deploys them to GitHub.

## Features

- **Diverse Project Generation**: Creates unique ML projects daily with different:
  - Tasks (Classification, Regression, Clustering)
  - Datasets (Breast Cancer, Iris, Wine Quality, Diabetes, etc.)
  - Algorithms (Random Forest, XGBoost, Neural Networks, etc.)
  - Advanced Techniques (SMOTE, Feature Selection, Model Calibration)

- **Complete Project Structure**:
  - Main Python implementation file
  - Requirements.txt with dependencies
  - Comprehensive README
  - Generated visualizations

- **Automated GitHub Integration**:
  - Creates/updates repository
  - Organizes projects in directories
  - Maintains main README with project index
  - Proper documentation and setup instructions

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Auto\ Coder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
```

## Usage

Run the project generator:
```bash
python auto_ml_coder.py
```

This will:
1. Generate a unique ML project
2. Create necessary files and documentation
3. Deploy to GitHub under ml-projects-collection

## Project Structure

- `auto_ml_coder.py`: Main implementation
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys)

## Requirements

- Python 3.8+
- Google Gemini API access
- GitHub account with API token
- Required Python packages (see requirements.txt)

## License

MIT License 