# Meta-Learning System: Podcast Listening Time Prediction

This repository contains a few-shot meta-learning system to predict podcast listening time.
It includes:
- Improved Prototypical Network with per-task adaptation
- Guardrails & production monitoring (error, coverage, drift-ish check, business metrics)
- Final monitoring dashboard
PODCAST LISTENING TIME PREDICTION USING META-LEARNING

PROJECT OVERVIEW

This project implements a meta-learning system to predict podcast listening times using few-shot learning techniques. The core challenge addressed is predicting user engagement for new podcasts with minimal historical data, a common problem in podcast platforms where new shows need quick performance insights.

The system compares traditional machine learning approaches that require extensive data with meta-learning methods that can make accurate predictions using only 1-5 episodes of a new podcast. This represents a significant advancement in cold-start prediction scenarios common in content recommendation systems.


BUSINESS PROBLEM

Podcast platforms face a critical challenge when new shows launch. Traditional prediction models require 50 or more episodes to achieve reasonable accuracy, meaning platforms must wait months before understanding show performance. This delay impacts several business decisions:

Content acquisition teams cannot quickly evaluate new show potential
Advertising placement requires historical performance data
Recommendation algorithms struggle with new content
Revenue forecasting remains uncertain during crucial early months

This project addresses these challenges by enabling accurate predictions from minimal data, typically 5 episodes or less. The business impact is substantial, allowing platforms to make informed decisions within weeks rather than months of a podcast launch.


TECHNICAL APPROACH

The project implements three distinct modeling approaches to solve this problem:

BASELINE MODEL - RANDOM FOREST

The baseline uses a traditional Random Forest regressor trained on 601,797 episodes from 38 podcasts. This model serves as the performance benchmark, representing conventional machine learning approaches that require substantial training data. The baseline achieved 13.03 RMSE and 0.772 R-squared on unseen test podcasts.

The baseline model requires access to all available episodes for each podcast during testing, typically 50 or more episodes per show. While accurate, this approach is impractical for new podcast launches where historical data is limited.

PROTOTYPICAL NETWORK - BASIC META-LEARNING

The prototypical network implements classic few-shot learning by creating prototype representations from support sets. For each new podcast, the model uses the first K episodes as a support set to create a prototype, then predicts listening times for remaining episodes.

This approach uses simple mean-based prediction where all query episodes receive the average listening time from the support set. While computationally efficient and requiring minimal data, this basic approach achieved 29.90 RMSE with 5-shot learning, significantly worse than the baseline but using 10x less data.

IMPROVED PROTOTYPICAL NETWORK - RIDGE REGRESSION ADAPTATION

The improved approach addresses the limitations of mean-based prediction by training a global Ridge regression model on all training data, then adapting predictions for new tasks using support set error correction. This hybrid approach combines the generalization power of traditional models with the data efficiency of meta-learning.

The adaptation mechanism calculates the difference between support set predictions and actual values, then adjusts query predictions accordingly. This task-specific calibration allows the model to quickly adapt to each podcast's unique characteristics using only a handful of episodes.

The improved prototypical network achieved 13.96 RMSE with 5-shot learning, approaching baseline performance while using only 5 episodes per podcast compared to the baseline's 50+ episodes requirement.


GUARDRAIL IMPLEMENTATION

A critical production consideration is prediction validity. Podcast listening times have natural physical constraints - users cannot listen for negative time, and the dataset shows episodes range from 0 to 120 minutes maximum. Without constraints, models occasionally produce invalid predictions outside these bounds.

The implementation includes a guardrail that clips all predictions to the valid range of 0-120 minutes. This simple but essential constraint ensures predictions remain interpretable and actionable for business decisions. The clipping operation is applied post-prediction:

predictions = np.clip(predictions, 0, 120)

This guardrail prevents edge cases where model predictions might suggest impossible listening behaviors, ensuring system outputs remain reliable for downstream applications like recommendation engines and performance dashboards.


FEW-SHOT LEARNING STRATEGY

The project implements and evaluates multiple few-shot learning configurations:

1-SHOT LEARNING

Using only the first episode to predict all subsequent episodes represents the most extreme data scarcity scenario. Performance at this level provides a lower bound on model capabilities. The improved prototypical network achieved 15.41 RMSE with 1-shot learning.

While accuracy is limited, 1-shot prediction enables same-day performance forecasting for brand new podcasts. This capability has immediate business value for content acquisition teams evaluating pilot episodes.

3-SHOT LEARNING

Three episodes provide enough signal to identify basic trends while maintaining rapid deployment timelines. The improved model achieved 14.52 RMSE, representing 5.7% improvement over 1-shot learning.

This configuration enables weekly performance reviews for new shows, substantially faster than traditional approaches requiring months of data collection.

5-SHOT LEARNING

Five episodes emerged as the optimal balance between data efficiency and prediction accuracy. The improved prototypical network achieved 13.96 RMSE, approaching baseline performance while using 90% less data.

This represents the recommended production configuration, enabling accurate predictions within 5-6 weeks of podcast launch while maintaining near-baseline accuracy. The model explains 73.9% of variance in listening times at this configuration.

10-SHOT AND 20-SHOT LEARNING

Higher shot counts showed diminishing returns. Ten-shot learning achieved 13.79 RMSE and 20-shot achieved 13.62 RMSE, demonstrating only marginal improvements beyond 5 shots.

These results validate that 5 episodes capture sufficient podcast-specific characteristics for accurate prediction, with additional episodes providing minimal accuracy gains while substantially increasing deployment delays.


DESIGN DECISIONS

FEATURE ENGINEERING APPROACH

The system engineers 61 features across 7 categories: host characteristics, guest popularity, content attributes, publication timing, advertising load, temporal patterns, and interaction effects. Feature importance analysis revealed that episode length alone accounts for 88.97% of predictive power.

Based on this analysis, meta-learning models use only the top 10 features, capturing 90% of total feature importance while reducing computational complexity. This dimensionality reduction proved essential for few-shot learning where limited samples make high-dimensional spaces problematic.

MODEL ARCHITECTURE CHOICE

The decision to use Ridge regression in the improved prototypical network rather than deep neural networks was deliberate. The dataset size and feature count do not justify neural network complexity, and Ridge regression provides interpretable coefficients while maintaining computational efficiency.

The linear model also enables faster training and prediction, critical for production deployment where new podcasts require immediate performance forecasts.

EVALUATION METHODOLOGY

The project implements podcast-level train-test splitting rather than random episode splitting. This ensures test podcasts were completely unseen during training, providing realistic performance estimates for the true cold-start scenario.

Test sets contain 10 completely new podcasts totaling 148,203 episodes, representing 20% of available podcasts. This methodology prevents data leakage and validates that models truly generalize to new content rather than memorizing training podcast patterns.


BUSINESS IMPACT

ACCELERATED DECISION MAKING

The 5-shot learning system enables accurate performance forecasting within 5-6 weeks of podcast launch compared to 12-16 weeks required by traditional approaches. This 60-70% reduction in evaluation time allows content teams to make acquisition and cancellation decisions 2-3 months earlier.

For a platform acquiring 100 new podcasts annually, this acceleration enables 8-12 additional decision cycles per year. The resulting improvement in portfolio quality compounds over time as poor performers are identified and replaced faster.

REDUCED DATA REQUIREMENTS

Meta-learning reduces data collection requirements by 90% while maintaining 93% of baseline accuracy. This efficiency has direct cost implications. Each episode requires storage, processing, and maintenance infrastructure. Reducing data requirements from 50 to 5 episodes per prediction decreases operational costs proportionally.

The system also enables prediction for short-run experimental content where 50 episodes may never accumulate, unlocking a previously inaccessible market segment.

IMPROVED RECOMMENDATION QUALITY

Cold-start content typically receives poor recommendations due to insufficient historical data. By enabling accurate predictions from minimal samples, the system allows new podcasts to enter recommendation engines weeks earlier than traditional approaches.

This improvement directly impacts new show discoverability and audience growth rates. Early inclusion in recommendations during the critical launch window substantially affects long-term show success.

RISK MITIGATION

Early performance visibility reduces financial risk in content acquisition. Rather than committing to multi-year contracts based on pilot episodes alone, platforms can validate performance trends within weeks, enabling more informed negotiation and contract structuring.

For high-value acquisitions, the ability to validate performance quickly justifies premium pricing for proven concepts while avoiding overpayment for underperforming content.


PERFORMANCE COMPARISON

BASELINE MODEL PERFORMANCE

RMSE: 13.03 minutes
MAE: 9.41 minutes
R-squared: 0.772
Data requirement: 50+ episodes per podcast

The baseline establishes the performance ceiling achievable with unlimited data. It explains 77.2% of variance in listening times, representing strong predictive power but requiring substantial historical data.

IMPROVED PROTOTYPICAL NETWORK PERFORMANCE

RMSE: 13.96 minutes (5-shot)
MAE: 10.33 minutes (5-shot)
R-squared: 0.739 (5-shot)
Data requirement: 5 episodes per podcast

The meta-learning approach achieves 93% of baseline accuracy using 10% of the data. The 0.93 RMSE difference (7.1% accuracy gap) is negligible for most business applications, especially considering the 10x reduction in data requirements.

The R-squared difference of 0.033 indicates the meta-learning model explains only 3.3% less variance than the baseline, a marginal trade-off for the substantial efficiency gains.

PERFORMANCE ACROSS K-SHOT CONFIGURATIONS

1-shot: 15.41 RMSE, enables same-day forecasting
3-shot: 14.52 RMSE, enables weekly evaluation
5-shot: 13.96 RMSE, optimal accuracy-efficiency balance
10-shot: 13.79 RMSE, diminishing returns begin
20-shot: 13.62 RMSE, minimal improvement over 5-shot

The performance curve shows rapid improvement from 1 to 5 shots followed by minimal gains, validating 5-shot as the production recommendation.


INSTALLATION AND USAGE

INSTALLATION

pip install -r requirements.txt

BASIC USAGE

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import engineer_features
from src.utils.helpers import create_meta_tasks, split_tasks
from src.models.improved_prototypical import train_improved_prototypical, evaluate_improved_prototypical
from src.evaluation.metrics import get_feature_importance, get_top_n_features

preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.load_data('train.csv', 'test.csv')
train_df, test_df = preprocessor.preprocess(train_df, test_df)

train_df = engineer_features(train_df)

tasks = create_meta_tasks(train_df, X, y)
train_tasks, test_tasks = split_tasks(tasks)

baseline_model = train_baseline_model(train_tasks)
importance_df = get_feature_importance(baseline_model, feature_cols)
top_features = get_top_n_features(importance_df, n=10)

improved_proto = train_improved_prototypical(train_tasks, top_features, feature_cols)
results = evaluate_improved_prototypical(improved_proto, test_tasks, feature_cols)

RUNNING TESTS

pytest tests/ -v

or run individual test files:

python tests/test_preprocessing.py
python tests/test_evaluation.py
python tests/test_models.py


PROJECT STRUCTURE

src/data/
  preprocessing.py - Data loading and cleaning
  feature_engineering.py - 61 feature creation

src/models/
  baseline.py - Random Forest baseline
  meta_learning.py - Basic prototypical network
  improved_prototypical.py - Ridge regression adaptation

src/evaluation/
  metrics.py - Performance metrics and feature importance

src/utils/
  helpers.py - Task creation and data utilities

tests/
  test_preprocessing.py - Data processing tests
  test_evaluation.py - Metrics tests
  test_models.py - Model tests


FUTURE ENHANCEMENTS

The current implementation provides a strong foundation for production deployment. Several enhancements could further improve performance:

Incorporate temporal dynamics such as episode recency and publication frequency patterns.

Implement ensemble methods combining multiple meta-learning approaches.

Add genre-specific adaptation allowing models to specialize by content category.

Integrate user-level features to personalize predictions beyond podcast characteristics.

Develop online learning capabilities enabling continuous model updates as new data arrives.

These enhancements represent natural next steps as the system matures in production environments.


CONCLUSION

This meta-learning system successfully addresses the cold-start prediction problem in podcast analytics. By achieving near-baseline accuracy with 10% of the data, the system enables faster business decisions, reduces operational costs, and improves recommendation quality for new content.

The 5-shot learning configuration represents the optimal production deployment, providing accurate predictions within 5-6 weeks of podcast launch. The guardrail implementation ensures predictions remain valid and actionable, while the comprehensive test suite validates system reliability.

The business impact is substantial, accelerating decision timelines by 60-70% while maintaining prediction accuracy sufficient for content acquisition, advertising placement, and recommendation systems. This combination of technical capability and business value positions the system as a practical solution for production podcast platforms.
## Dashboard

![Monitoring Dashboard](meta_learning_dashboard.png)

## Learning Curve

![Learning Curve](meta_learning_results.png)

## Quick Start

\\\ash
# create venv
python -m venv metafinal.venv
./metafinal.venv/Scripts/Activate.ps1

pip install -r requirements.txt
# open notebook
code finalnotebook.ipynb
\\\

## Structure
- \src/\ code (models, preprocessing, utils)
- \
esults/\ sample outputs
- \	ests/\ quick checks

