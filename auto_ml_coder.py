import os
import google.generativeai as genai
from dotenv import load_dotenv
from github import Github
import time
import sys
import json
import random
import logging

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoMLCoder:
    def __init__(self):
        try:
            self.github = Github(os.getenv('GITHUB_TOKEN'))
            # Test GitHub authentication
            self.github.get_user().login
            logger.info("GitHub authentication successful")
        except Exception as e:
            logger.error(f"Error authenticating with GitHub: {e}")
            sys.exit(1)
        
    def generate_ml_code(self):
        # Core components for project generation
        datasets = {
            "classification": [
                {"name": "breast_cancer_classification", "loader": "load_breast_cancer()", 
                 "description": "Breast Cancer Classification using Wisconsin Diagnostic Dataset"},
                {"name": "iris_flower_classification", "loader": "load_iris()", 
                 "description": "Iris Flower Species Classification"},
                {"name": "wine_quality_classification", "loader": "load_wine()", 
                 "description": "Wine Quality Classification"},
                {"name": "digit_recognition", "loader": "load_digits()", 
                 "description": "Handwritten Digit Recognition"},
                {"name": "diabetes_prediction", "loader": "fetch_openml('diabetes', version=1)", 
                 "description": "Diabetes Prediction using Clinical Features"},
                {"name": "credit_risk_assessment", "loader": "fetch_openml('credit-g', version=1)", 
                 "description": "Credit Risk Assessment"}
            ],
            "regression": [
                {"name": "diabetes_progression", "loader": "load_diabetes()", 
                 "description": "Diabetes Disease Progression Prediction"},
                {"name": "house_price_prediction", "loader": "fetch_openml('house_prices', version=1)", 
                 "description": "House Price Prediction"},
                {"name": "medical_cost_prediction", "loader": "fetch_openml('medical_charges', version=1)", 
                 "description": "Medical Cost Prediction"},
                {"name": "bike_sharing_demand", "loader": "fetch_openml('bike_sharing', version=1)", 
                 "description": "Bike Sharing Demand Prediction"}
            ],
            "clustering": [
                {"name": "customer_segmentation", "loader": "fetch_openml('wholesale-customers', version=1)", 
                 "description": "Customer Segmentation Analysis"},
                {"name": "mall_customer_clustering", "loader": "make_blobs(n_samples=1000, centers=3)", 
                 "description": "Mall Customer Clustering"},
                {"name": "market_segmentation", "loader": "make_blobs(n_samples=1000, centers=4)", 
                 "description": "Market Segmentation Analysis"}
            ]
        }

        algorithms = {
            "classification": [
                ["RandomForestClassifier", "GradientBoostingClassifier", "SVC"],
                ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"],
                ["LogisticRegression", "RidgeClassifier", "SGDClassifier"],
                ["KNeighborsClassifier", "DecisionTreeClassifier", "AdaBoostClassifier"],
                ["ExtraTreesClassifier", "BaggingClassifier", "HistGradientBoostingClassifier"]
            ],
            "regression": [
                ["RandomForestRegressor", "GradientBoostingRegressor", "SVR"],
                ["XGBRegressor", "LGBMRegressor", "CatBoostRegressor"],
                ["ElasticNet", "Lasso", "Ridge"],
                ["KNeighborsRegressor", "DecisionTreeRegressor", "AdaBoostRegressor"],
                ["ExtraTreesRegressor", "BaggingRegressor", "HistGradientBoostingRegressor"]
            ],
            "clustering": [
                ["KMeans", "AgglomerativeClustering", "DBSCAN"],
                ["SpectralClustering", "Birch", "MiniBatchKMeans"],
                ["MeanShift", "AffinityPropagation", "OPTICS"]
            ]
        }

        techniques = {
            "preprocessing": [
                "with feature selection",
                "with dimensionality reduction",
                "with feature engineering",
                "with data cleaning",
                "with outlier detection",
                "with missing value imputation"
            ],
            "enhancement": [
                "with cross-validation",
                "with hyperparameter optimization",
                "with ensemble methods",
                "with pipeline optimization",
                "with model stacking",
                "with model calibration"
            ],
            "analysis": [
                "with feature importance analysis",
                "with model interpretation",
                "with learning curves",
                "with error analysis",
                "with model comparison",
                "with performance profiling"
            ],
            "special": [
                "with SMOTE balancing",
                "with time series cross-validation",
                "with custom scoring metrics",
                "with threshold optimization",
                "with feature selection stability",
                "with model explainability"
            ]
        }

        metrics = {
            "classification": [
                ["accuracy", "precision", "recall", "f1"],
                ["roc_auc", "precision_recall_auc", "log_loss"],
                ["cohen_kappa", "matthews_corrcoef", "balanced_accuracy"]
            ],
            "regression": [
                ["mse", "mae", "r2"],
                ["explained_variance", "max_error", "median_absolute_error"],
                ["mean_squared_log_error", "mean_poisson_deviance", "mean_gamma_deviance"]
            ],
            "clustering": [
                ["silhouette", "calinski_harabasz", "davies_bouldin"],
                ["adjusted_rand", "adjusted_mutual_info", "homogeneity"],
                ["completeness", "v_measure", "fowlkes_mallows"]
            ]
        }

        # Randomly select components
        task = random.choice(["classification", "regression", "clustering"])
        dataset_info = random.choice(datasets[task])
        algorithm_set = random.choice(algorithms[task])
        metric_set = random.choice(metrics[task])
        
        # Select multiple techniques
        selected_techniques = []
        for technique_type in techniques:
            selected_techniques.append(random.choice(techniques[technique_type]))
        
        # Create unique project configuration
        project_type = {
            "name": dataset_info["name"],
            "description": dataset_info["description"],
            "dataset": dataset_info["loader"],
            "task": task,
            "algorithms": algorithm_set,
            "metrics": metric_set,
            "techniques": selected_techniques
        }
        
        # Create unique task name combining multiple techniques
        unique_task = f"{project_type['description']} with {' and '.join(selected_techniques)}"
        
        # Base imports based on project type
        base_imports = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"Machine Learning Project - {unique_task.title()}

This module implements a complete machine learning pipeline for {unique_task}
using various algorithms and comprehensive evaluation metrics.
\"\"\"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""

        # Add dataset import
        if "make_" in project_type['dataset']:
            base_imports += f"from sklearn.datasets import {project_type['dataset'].split('(')[0]}\n"
        else:
            base_imports += f"from sklearn.datasets import {project_type['dataset'].split('(')[0]}\n"

        # Add common imports
        base_imports += """from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
"""

        # Add task-specific imports
        if "feature selection" in selected_techniques:
            base_imports += "from sklearn.feature_selection import RFE, RFECV\n"
        if "dimensionality reduction" in selected_techniques:
            base_imports += "from sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\n"
        if "SMOTE" in selected_techniques:
            base_imports += "from imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline as ImbPipeline\n"
        if "ensemble" in selected_techniques:
            base_imports += "from sklearn.ensemble import VotingClassifier, StackingClassifier\n"
        if "calibration" in selected_techniques:
            base_imports += "from sklearn.calibration import CalibratedClassifierCV\n"

        # Add metric imports based on task
        if "classification" in project_type['task']:
            base_imports += """from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)"""
        else:
            base_imports += """from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)"""

        # Add algorithm imports
        if task == "classification":
            base_imports += """from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    logger.warning("Some boosting libraries not available. Using default classifiers.")
"""
        elif task == "regression":
            base_imports += """from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
except ImportError:
    logger.warning("Some boosting libraries not available. Using default regressors.")
"""
        else:  # clustering
            base_imports += """from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering, Birch, MiniBatchKMeans,
    MeanShift, AffinityPropagation, OPTICS
)
"""

        base_code = base_imports + """
import logging
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
"""
        
        try:
            # Generate class implementation
            class_prompt = f"""
            Create a Python class called MLProject that implements a machine learning pipeline for {unique_task} with these methods:
            1. __init__: Initialize project variables
            2. load_data: Load {project_type['dataset']} dataset
            3. explore_data: Perform EDA with visualizations specific to {', '.join(selected_techniques)}
            4. preprocess_data: Implement preprocessing specific to {', '.join(selected_techniques)}
            5. train_models: Train and tune multiple models ({', '.join(project_type['algorithms'])})
            6. evaluate_models: Calculate metrics ({', '.join(project_type['metrics'])}) and create plots
            
            Include proper error handling, logging, and type hints.
            Store multiple models and their results for comparison.
            Implement specific techniques for {', '.join(selected_techniques)}.
            """
            
            class_response = model.generate_content(class_prompt)
            if not class_response:
                return None, None
            
            # Generate main function
            main_prompt = f"""
            Create a main function that:
            1. Initializes the MLProject class
            2. Runs the complete pipeline
            3. Prints evaluation results for all models
            4. Compares model performances
            5. Includes proper error handling
            
            The function should handle {unique_task} specific metrics and visualizations.
            Include specific analysis for {', '.join(selected_techniques)}.
            """
            
            main_response = model.generate_content(main_prompt)
            if not main_response:
                return None, None
            
            # Combine all parts
            full_code = (
                base_code + "\n" +
                class_response.text.replace("```python", "").replace("```", "").strip() + "\n\n" +
                main_response.text.replace("```python", "").replace("```", "").strip()
            )
            
            return full_code, unique_task
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return None, None
    
    def convert_to_notebook(self, code_text):
        notebook = {
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
            "cells": []
        }

        # Split the code into cells based on markdown and code sections
        cells = []
        current_cell = ""
        cell_type = "code"
        in_markdown_section = False

        lines = code_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for markdown headers
            if line.startswith('# '):
                if current_cell.strip():
                    cells.append({"type": cell_type, "content": current_cell.strip()})
                current_cell = line
                cell_type = "markdown"
                in_markdown_section = True
            
            # Check for code blocks
            elif line.startswith('```python'):
                if current_cell.strip():
                    cells.append({"type": cell_type, "content": current_cell.strip()})
                current_cell = ""
                cell_type = "code"
                i += 1  # Skip the ```python line
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    current_cell += lines[i] + "\n"
                    i += 1
                in_markdown_section = False
            
            # Regular line
            else:
                if in_markdown_section:
                    if line.startswith('##'):  # Subsection
                        if current_cell.strip():
                            cells.append({"type": "markdown", "content": current_cell.strip()})
                        current_cell = line
                    else:
                        current_cell += "\n" + line
                else:
                    current_cell += "\n" + line
            
            i += 1

        # Add the last cell
        if current_cell.strip():
            cells.append({"type": cell_type, "content": current_cell.strip()})

        # Convert cells to notebook format
        for cell in cells:
            if cell["type"] == "code":
                notebook["cells"].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": cell["content"].split('\n')
                })
            else:
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": cell["content"].split('\n')
                })

        return json.dumps(notebook, indent=2)
    
    def review_code(self, code):
        if not code:
            return "Code generation failed, cannot proceed with review."
            
        review_prompt = f"""
        Analyze this machine learning project and create a professional README that includes:
        
        1. Project Overview:
           - Main objectives
           - Key features
           - Technical highlights
        
        2. Technical Details:
           - Model architecture
           - Data processing pipeline
           - Key algorithms
        
        3. Performance Metrics:
           - Evaluation results
           - Benchmark comparisons
           - Model strengths
        
        4. Implementation Details:
           - Dependencies
           - System requirements
           - Setup instructions
        
        Make it professional and concise. Do not mention anything about automated generation.
        Focus on the technical merits and features of the project.
        
        Code to analyze:
        {code}
        """
        
        try:
            response = model.generate_content(review_prompt)
            return response.text
        except Exception as e:
            print(f"Error reviewing code: {e}")
            return None
    
    def create_github_repo(self, code, review):
        if not code or not review:
            logger.error("Missing code or review, cannot create repository.")
            return None
            
        try:
            # Use a single repository for all ML projects
            repo_name = "ml-projects-collection"
            user = self.github.get_user()
            
            try:
                # Try to get existing repository
                repo = user.get_repo(repo_name)
                logger.info(f"Using existing repository: {repo_name}")
            except:
                # Create new repository if it doesn't exist
                repo = user.create_repo(repo_name, description="Collection of Machine Learning Projects")
                logger.info(f"Created new repository: {repo_name}")
            
            # Create a new directory for this project using the project name from review
            project_name = review.split('\n')[0].strip('#').strip().lower().replace(' ', '-')
            project_dir = project_name
            
            logger.info(f"Creating project directory: {project_dir}")
            
            # Create main Python file in project directory with descriptive name
            main_file_name = f"{project_name.replace('-', '_')}.py"
            repo.create_file(
                f"{project_dir}/{main_file_name}",
                f"Add {project_name} implementation",
                code
            )
            
            # Create requirements.txt in project directory
            requirements = """numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
imbalanced-learn>=0.10.0  # for SMOTE
xgboost>=1.7.0  # for XGBoost
lightgbm>=4.0.0  # for LightGBM
optuna>=3.0.0  # for hyperparameter optimization
shap>=0.41.0  # for model interpretation"""
            
            repo.create_file(
                f"{project_dir}/requirements.txt",
                f"Add {project_name} dependencies",
                requirements
            )
            
            # Create README.md in project directory with proper formatting
            readme_content = f"""# {project_name.replace('-', ' ').title()}

{review}

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/{user.login}/ml-projects-collection.git
cd ml-projects-collection/{project_dir}
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
python {main_file_name}
```

## Project Structure

- `{main_file_name}`: Main implementation file
- `requirements.txt`: Project dependencies
- Generated visualizations:
  - Feature distributions
  - Correlation matrix
  - ROC curve
  - Feature importance plot

## License

MIT License
"""
            
            repo.create_file(
                f"{project_dir}/README.md",
                f"Add {project_name} documentation",
                readme_content
            )
            
            # Update main README to include this project
            try:
                main_readme = repo.get_contents("README.md")
                current_content = main_readme.decoded_content.decode()
                new_project_entry = f"\n## [{project_name.replace('-', ' ').title()}]({project_dir})\n{review.split('\n')[1].strip()}\n"
                updated_content = current_content + new_project_entry
                repo.update_file(
                    "README.md",
                    f"Add {project_name} to project list",
                    updated_content,
                    main_readme.sha
                )
                logger.info("Updated main README with new project")
            except:
                # Create main README if it doesn't exist
                initial_content = """# Machine Learning Projects Collection

This repository contains various machine learning projects implementing different algorithms and techniques.

"""
                repo.create_file(
                    "README.md",
                    "Initial commit",
                    initial_content
                )
                logger.info("Created main README")
            
            logger.info(f"Project successfully created at {project_dir}")
            return repo.html_url + f"/tree/main/{project_dir}"
        except Exception as e:
            logger.error(f"Error creating GitHub repository: {e}")
            return None

def main():
    auto_coder = AutoMLCoder()
    
    print("🔧 Initializing project generation...")
    
    # Generate ML code
    print("\n📝 Generating project code...")
    code, task = auto_coder.generate_ml_code()
    if not code:
        print("Failed to generate code. Exiting.")
        return
    
    # Create documentation
    print(f"\n📚 Creating documentation for {task} project...")
    review = auto_coder.review_code(code)
    if not review:
        print("Failed to create documentation. Exiting.")
        return
    
    # Create GitHub repository
    print("\n🚀 Deploying project...")
    repo_url = auto_coder.create_github_repo(code, review)
    
    if repo_url:
        print(f"\n✨ Project successfully deployed to: {repo_url}")
        print(f"Task: {task}")
    else:
        print("\n❌ Project deployment failed.")

if __name__ == "__main__":
    main() 