# Ad Sales Predicting App

## Project Overview
This project aims to predict the probability of an advertisement impression leading to an app installation. By leveraging data sampled from a production environment, I developed a Deep Learning Model to estimate the likelihood of app installations from ad impressions. The predictions are crucial for estimating the optimal bid values for ad impressions.

## Environment Setup
- **Tools Used**: Python, Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, Scikit-learn, Imbalanced-learn
- **Data**: Utilized `train_data.csv` for model training and `assessment_data.csv` for model assessment.

## Data Analysis and Visualization

### Data description

- ```id```: impression id
- ```timestamp```: time of the event in UTC
- ```campaignId```: id of the advertising campaign (the game being advertised)
- ```platform```: device platform
- ```softwareVersion```: OS version of the device
- ```sourceGameId```: id of the publishing game (the game being played)
- ```country```: country of user
- ```startCount```: how many times the user has started (any) campaigns
- ```viewCount```: how many times the user has viewed (any) campaigns
- ```clickCount```: how many times the user has clicked (any) campaigns
- ```installCount```: how many times the user has installed games from this ad network
- ```lastStart```: last time user started any campaign
- ```startCount1d```: how many times user has started (any) campaigns within the last 24 hours
- ```startCount7d```: how many times user has started (any) campaigns within the last 7 days
- ```connectionType```: internet connection type
- ```deviceType```: device model
- ```install```: binary indicator if an install was observed (install=1) or not (install=0) after impression
- 
### Exploratory Data Analysis (EDA)
- Examined feature distributions, target variable (install), and relationships between features.
- Identified patterns and anomalies such as the imbalance in the target variable.
- Conducted bivariate analysis to understand the interaction between features and the target variable.

### Visualization
- Generated plots to visualize relationships between different features and the install rate.
- Utilized pair plots, count plots, and scatter plots to uncover insights from the data.

## Feature Engineering and Data Preprocessing
- Handled missing values and engineered new features to improve model performance.
- Performed data cleaning and transformation, including normalization and conversion of categorical variables.

## Model Development
- Explored various models including Logistic Regression, Decision Trees, and Random Forest before finalizing a Neural Network model.
- Implemented techniques to handle imbalanced datasets, such as SMOTE and undersampling.

## Model Evaluation
- Evaluated the model based on ROC AUC, log loss, prediction bias, and other metrics.
- Conducted extensive experiments to optimize model performance, including hyperparameter tuning and regularization.

## Final Model and Predictions
- The final Neural Network model showcased my ability to predict app installation probabilities with a moderate ROC AUC score.
- Utilized the model to predict install probabilities on the unseen `assessment_data.csv`, focusing on accuracy and bias.

## Key Findings and Observations
- The imbalance in the target variable posed a significant challenge, addressed through resampling techniques.
- Feature engineering played a crucial role in enhancing model performance.
- Despite the moderate success of the final model, further improvements are suggested for future work.

## Tools and Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, Scikit-learn, Imbalanced-learn
- **Deep Learning Framework**: TensorFlow
- **Model Evaluation Metrics**: ROC AUC, Log Loss, Prediction Bias

## Future Work
- Explore additional feature engineering and selection techniques to improve model performance.
- Experiment with more advanced neural network architectures and optimization algorithms.
- Investigate alternative strategies for handling imbalanced data.

## Conclusion
This project demonstrates the process of predicting app installation probabilities from ad impressions using deep learning. Through meticulous data analysis, visualization, feature engineering, and model evaluation, I developed a model that provides insights into the factors influencing app installations. Future improvements could further enhance the model's predictive capabilities, providing valuable guidance for ad bidding strategies.
