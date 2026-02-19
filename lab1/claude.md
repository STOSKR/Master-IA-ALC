# EXIST2025 - Sexism Detection Project

## Project Overview
This project focuses on analyzing and detecting sexism in Spanish tweets using the EXIST2025 dataset. The goal is to build models for multi-level classification of sexist content in social media.

## Dataset Structure

### Location
```
lab1_materials/
├── dataset_task1_exist2025/
│   ├── training.json    # Training data with annotations
│   └── test.json        # Test/development data
└── golds_task1_exist2025/
    └── training.json    # Gold standard labels
```

### Data Format

#### Training/Test Dataset (dataset_task1_exist2025/)
Each entry is indexed by a unique ID (e.g., "100001", "300002") and contains:

**Tweet Information:**
- `id_EXIST`: Unique identifier
- `lang`: Language code ("es" for Spanish)
- `tweet`: The actual tweet text content
- `split`: Dataset split ("TRAIN_ES" or "DEV_ES")

**Annotator Information:**
- `number_annotators`: Number of annotators (typically 6)
- `annotators`: List of annotator IDs
- `gender_annotators`: Gender of each annotator ["F", "M"]
- `age_annotators`: Age ranges ["18-22", "23-45", "46+"]
- `ethnicities_annotators`: Ethnicity of annotators
- `study_levels_annotators`: Education levels
- `countries_annotators`: Country of origin

**Annotation Labels:**
- `labels_task1`: Array of YES/NO labels (sexism detection)
- `labels_task2`: Type of sexism classification
  - "REPORTED": Reported/indirect sexism
  - "JUDGEMENTAL": Judgemental sexism
  - "DIRECT": Direct sexism
  - "-": Not applicable (when task1 is NO)
- `labels_task3`: Categories of sexism (can be multiple)
  - "OBJECTIFICATION"
  - "SEXUAL-VIOLENCE"
  - "STEREOTYPING-DOMINANCE"
  - "IDEOLOGICAL-INEQUALITY"
  - "MISOGYNY-NON-SEXUAL-VIOLENCE"
  - "-": Not applicable

#### Gold Standard Labels (golds_task1_exist2025/)
Simplified format with consensus labels:
```json
{
  "test_case": "EXIST2025",
  "id": "100001",
  "value": "YES" or "NO"
}
```

## Task Hierarchy

### Task 1: Binary Sexism Detection
- **Goal**: Determine if a tweet contains sexist content
- **Labels**: YES (sexist) / NO (not sexist)
- **Note**: This is the primary task; Tasks 2 and 3 only apply when Task 1 = YES

### Task 2: Sexism Type Classification
- **Goal**: Classify the type of sexism present
- **Categories**:
  - REPORTED: Indirect or reported sexism
  - JUDGEMENTAL: Judgemental statements
  - DIRECT: Direct sexist attacks or comments
- **Applicability**: Only when Task 1 = YES

### Task 3: Sexism Category Classification
- **Goal**: Multi-label classification of sexism categories
- **Categories**:
  - OBJECTIFICATION: Treating people as objects
  - SEXUAL-VIOLENCE: References to sexual violence
  - STEREOTYPING-DOMINANCE: Gender stereotypes and dominance
  - IDEOLOGICAL-INEQUALITY: Ideological inequality
  - MISOGYNY-NON-SEXUAL-VIOLENCE: Misogyny and non-sexual violence
- **Note**: Multiple categories can apply to a single tweet
- **Applicability**: Only when Task 1 = YES

## Data Characteristics

### Language
- Primary language: Spanish (es)
- Mix of standard Spanish and colloquial expressions
- May include code-switching and internet slang

### Annotation Diversity
- Each tweet annotated by 6 different annotators
- Diverse annotator backgrounds:
  - Gender: Female and Male
  - Age groups: 18-22, 23-45, 46+
  - Various ethnicities and countries
  - Different education levels

### Data Splits
- **TRAIN_ES**: Training data (IDs starting with 100xxx)
- **DEV_ES**: Development/test data (IDs starting with 300xxx)

## Exploratory Data Analysis Findings

### Dataset Statistics
- **Training samples**: 6,064 tweets
- **Test samples**: 934 tweets
- **Class distribution (Task 1)**:
  - NO (non-sexist): 3,367 tweets (55.5%)
  - YES (sexist): 2,697 tweets (44.5%)
  - Balance ratio: 0.80 (relatively balanced)

### Inter-Annotator Agreement
- **Mean agreement**: 84.6%
- **Median agreement**: 83.3%
- **Perfect agreement (100%)**: 75% of cases
- **Minimum agreement**: 66.7% (4 of 6 annotators)
- **Conclusion**: Excellent consensus, labels are reliable

### Text Characteristics
- **Average tweet length**: 178 characters
- **Average word count**: 28 words
- **Average word length**: 5.5 characters
- **Special patterns**:
  - Mentions (@): 53.5% of tweets
  - Hashtags (#): 14.2%
  - URLs: 32.1%
  - Emojis: 7.6%

### Feature Correlation Analysis

#### Most Informative Features (Mutual Information > 0.01):
1. **task1_agreement** (MI: 0.0204) - Agreement between annotators is the STRONGEST predictor
2. **avg_word_length** (MI: 0.0173) - Average word length
3. **has_hashtag** (MI: 0.0103) - Presence of hashtags

#### Point-Biserial Correlations with Sexism:
- **avg_word_length**: r = -0.147, p < 0.001 - Sexist tweets use SHORTER words
- **task1_agreement**: r = -0.126, p < 0.001 - Sexist tweets have LOWER annotator agreement
- **tweet_length**: r = -0.038, p < 0.01 - Sexist tweets are slightly SHORTER

#### Chi-Square Test for Categorical Features:
- **has_url**: χ² = 93.39, p < 0.001 - HIGHLY significant
- **has_hashtag**: χ² = 67.33, p < 0.001 - HIGHLY significant
- **has_emoji**: χ² = 3.96, p = 0.047 - Significant
- **has_mention**: χ² = 3.42, p = 0.064 - NOT significant

#### Feature Redundancy:
- **tweet_length and word_count**: correlation = 0.929 - HIGHLY correlated
- **Recommendation**: Remove tweet_length, keep word_count

### Annotator Demographic Bias Analysis

#### Total Annotations Analyzed: 36,384 (6,064 tweets × 6 annotators)

#### Gender: NOT SIGNIFICANT
- **Female annotators**: 44.8% label as YES
- **Male annotators**: 44.2% label as YES
- **Difference**: 0.6 percentage points
- **Chi-square**: p = 0.229 (not significant)
- **Conclusion**: Annotator gender does NOT affect labeling

#### Age: SIGNIFICANT (small effect)
- **23-45 years**: 45.8% YES (more strict)
- **46+ years**: 45.3% YES
- **18-22 years**: 42.3% YES (more permissive)
- **Chi-square**: p < 0.001, Cramér's V = 0.031
- **Difference**: 3.5 percentage points between oldest and youngest

#### Ethnicity: SIGNIFICANT (small effect)
- **Multiracial**: 52.1% YES (most strict)
- **Hispanic/Latino**: 46.5% YES
- **White/Caucasian**: 43.8% YES
- **Black/African American**: 40.4% YES (most permissive)
- **Chi-square**: p < 0.001, Cramér's V = 0.036
- **Difference**: 11.7 percentage points between extremes

#### Education: SIGNIFICANT (small effect)
- **Doctorate**: 48.8% YES
- **High school**: 41.6% YES
- **Chi-square**: p < 0.001, Cramér's V = 0.041
- **Conclusion**: Higher education correlates with higher sexism detection rate

#### Country: HIGHLY SIGNIFICANT (strongest demographic effect)
- **Cuba**: 75.5% YES (very strict)
- **Brazil**: 65.3% YES
- **Latin America**: Generally >50% YES
- **Eastern Europe**: Generally <40% YES
- **Macedonia**: 14.5% YES (very permissive)
- **Chi-square**: p < 0.001, Cramér's V = 0.090
- **Difference**: 61 percentage points between extremes
- **Conclusion**: Country is the STRONGEST demographic predictor

#### Key Insights from Demographic Analysis:
1. **Cultural context matters**: Latin American annotators are more sensitive to sexism
2. **Gender is NOT a bias factor**: Equal labeling behavior between genders
3. **Diversity mitigates bias**: 6 annotators from 45 countries balance perspectives
4. **Consensus is robust**: Majority voting from diverse annotators reduces individual biases

## Key Considerations

### Inter-Annotator Agreement
- Multiple annotators may disagree on labels
- Gold standard represents consensus or majority vote
- Consider handling label disagreement in model development

### Class Imbalance
- Expect imbalance between YES/NO classes
- Some sexism categories may be rarer than others
- Consider appropriate evaluation metrics (F1, precision, recall)

### Multi-Label Nature
- Task 3 is multi-label (multiple categories can apply)
- Need to handle hierarchical dependencies (Task 2 and 3 depend on Task 1)

### Text Processing Challenges
- Social media language (informal, abbreviations)
- Mentions (@username) and hashtags
- Mixed language content
- URLs and special characters

## Evaluation Metrics
Consider using:
- **Task 1**: Accuracy, F1-score, Precision, Recall
- **Task 2**: Macro/Micro F1-score
- **Task 3**: Multi-label metrics (Hamming loss, subset accuracy, label-based F1)

## Model Development Strategy

### Potential Approaches
1. **Traditional ML**: SVM, Logistic Regression with TF-IDF features
2. **Deep Learning**: LSTM, GRU, CNN for text
3. **Transformers**: BETO, RoBERTa-es, mBERT for Spanish
4. **Hierarchical Models**: Separate models for each task level
5. **Multi-task Learning**: Joint learning of all three tasks

### Preprocessing Recommendations
- Text normalization (lowercase, handle mentions/URLs)
- Tokenization appropriate for Spanish
- Handle special characters and emojis
- Consider preserving some features (e.g., ALL CAPS for emphasis)

### Feature Engineering
- Bag of words / TF-IDF
- Word embeddings (Word2Vec, FastText for Spanish)
- Contextual embeddings (BETO, BERT multilingual)
- Metadata features from annotators (optional)

### Techniques for Model Robustness

#### Data Augmentation for Text
Text augmentation techniques to increase training data diversity:

1. **Synonym Replacement**:
   - Replace words with synonyms using Spanish WordNet or lexical databases
   - Libraries: `nlpaug`, `textaugment`
   - Preserve tweet context and meaning

2. **Back Translation**:
   - Translate Spanish → English → Spanish
   - Creates paraphrases with similar meaning
   - Use Google Translate API or MarianMT models

3. **Contextual Word Embeddings**:
   - Use BERT-based models to replace words with contextually similar alternatives
   - More sophisticated than synonym replacement
   - Library: `nlpaug` with transformer models

4. **Random Operations**:
   - Random insertion of synonyms
   - Random swap of word positions
   - Random deletion of non-critical words
   - **Caution**: May alter sexist meaning, use carefully

5. **Character-level Augmentation**:
   - Simulate typos common in social media
   - Add/remove accents (á → a)
   - Character swaps for adjacent keys

#### Handling Class Imbalance

1. **Resampling Techniques**:
   - **SMOTE** (Synthetic Minority Over-sampling): Generate synthetic samples for minority class
   - **ADASYN**: Adaptive synthetic sampling
   - **Random Under-sampling**: Reduce majority class samples
   - **Random Over-sampling**: Duplicate minority class samples
   - Library: `imbalanced-learn` (imblearn)

2. **Class Weights**:
   - Assign higher weights to minority class in loss function
   - `class_weight='balanced'` in scikit-learn
   - Custom weights: inversely proportional to class frequency
   - Apply in model training (sklearn, PyTorch, TensorFlow)

3. **Focal Loss**:
   - Alternative to cross-entropy for imbalanced data
   - Focuses on hard-to-classify examples
   - Reduces weight of well-classified examples

4. **Threshold Moving**:
   - Adjust classification threshold (default 0.5)
   - Optimize for F1-score instead of accuracy
   - Use validation set to find optimal threshold

#### Regularization Techniques

1. **L1/L2 Regularization**:
   - Prevent overfitting by penalizing large weights
   - L1 (Lasso): Encourages sparsity
   - L2 (Ridge): Prevents large weights
   - ElasticNet: Combination of L1 and L2

2. **Dropout**:
   - Randomly drop neurons during training
   - Typical rates: 0.1-0.5
   - Apply to dense layers in neural networks

3. **Early Stopping**:
   - Monitor validation loss/metric
   - Stop training when validation performance plateaus
   - Prevent overfitting to training data
   - Patience parameter: typically 3-10 epochs

4. **Learning Rate Scheduling**:
   - Reduce learning rate when validation metric stops improving
   - ReduceLROnPlateau callback
   - Helps fine-tune model convergence

#### Ensemble Methods

1. **Voting Classifiers**:
   - Hard voting: Majority vote from multiple models
   - Soft voting: Average predicted probabilities
   - Combine different model types (SVM, RF, BERT)

2. **Stacking**:
   - Train meta-learner on predictions of base models
   - Base models: Different architectures or feature sets
   - Meta-model: Logistic Regression, XGBoost

3. **Bagging**:
   - Train multiple models on bootstrap samples
   - Random Forest is a bagging ensemble
   - Reduces variance

4. **Boosting**:
   - Sequential training focusing on misclassified examples
   - XGBoost, LightGBM, AdaBoost
   - Good for tabular features (TF-IDF)

#### Transfer Learning & Fine-tuning

1. **Pre-trained Language Models**:
   - Start with BETO, RoBERTa-es, or mBERT
   - Fine-tune on EXIST2025 data
   - Much better than training from scratch

2. **Layer Freezing**:
   - Freeze early layers, train only top layers
   - Faster training, less overfitting
   - Gradually unfreeze layers if needed

3. **Domain Adaptation**:
   - Pre-train on similar Spanish social media data
   - Fine-tune on sexism detection task
   - Use Spanish Twitter corpora

4. **Multi-task Learning**:
   - Train all three tasks simultaneously
   - Shared representations benefit all tasks
   - Task-specific heads for each output

#### Noise Reduction & Robustness

1. **Label Smoothing**:
   - Convert hard labels (0, 1) to soft labels (0.1, 0.9)
   - Prevents overconfident predictions
   - Improves generalization

2. **Mixup**:
   - Create virtual training examples
   - Linear interpolation of embeddings and labels
   - Works with hidden representations

3. **Adversarial Training**:
   - Add small perturbations to input embeddings
   - Train model to be robust to these perturbations
   - Improves generalization

4. **Consensus-based Training**:
   - Use annotator agreement as confidence scores
   - Weight samples by inter-annotator agreement
   - 6/6 agreement → higher weight than 4/6

### Validation Strategies

#### Recommended Approach: Held-out Validation
- **Training set**: Use TRAIN_ES split (IDs 100xxx) for model training
- **Validation set**: Use DEV_ES split (IDs 300xxx) for validation and hyperparameter tuning
- This follows the natural split already provided in the dataset
- Prevents data leakage and provides realistic performance estimates

#### Alternative: K-Fold Cross-Validation
- **Standard K-Fold**: 5-fold or 10-fold cross-validation on TRAIN_ES
- **Stratified K-Fold**: Recommended to maintain class distribution in each fold
  - Stratify by Task 1 labels (YES/NO)
  - Important due to class imbalance
- Use for model selection and hyperparameter optimization
- Final evaluation should still use the held-out DEV_ES set

#### Considerations for Multi-label Tasks
- **Task 3 Stratification**: More complex due to multi-label nature
  - Use `IterativeStratification` from scikit-multilearn
  - Ensures balanced distribution of all label combinations
- **Hierarchical Validation**: Ensure validation accounts for task dependencies
  - Task 2 and 3 only evaluated on samples where Task 1 = YES

#### Validation Workflow
1. **Development Phase**:
   - Use TRAIN_ES for training with cross-validation
   - Tune hyperparameters using CV results
   - Monitor for overfitting

2. **Testing Phase**:
   - Train final model on full TRAIN_ES
   - Evaluate on DEV_ES (held-out set)
   - Report final metrics from DEV_ES

3. **Production Phase** (if applicable):
   - Retrain on TRAIN_ES + DEV_ES combined
   - Deploy for real predictions

#### Metrics to Track During Validation
- Per-fold metrics for stability assessment
- Mean and standard deviation across folds
- Confusion matrices for each task
- Class-wise performance (especially for minority classes)

## Expected Outputs

### For Training Phase
- Models capable of predicting all three task levels
- Evaluation reports with metrics per task
- Error analysis and confusion matrices

### For Test Phase
- Predictions in the format matching gold standard:
```json
{
  "test_case": "EXIST2025",
  "id": "300002",
  "value": "YES" or "NO"
}
```

## Tools and Libraries

### Recommended Python Libraries
- **Data Processing**: pandas, numpy, json
- **NLP**: spaCy (es_core_news_sm), NLTK
- **ML**: scikit-learn
- **Deep Learning**: PyTorch, TensorFlow/Keras
- **Transformers**: Hugging Face transformers (BETO, mBERT)
- **Evaluation**: scikit-learn metrics, sklearn-multilearn
- **Data Augmentation**: nlpaug, textaugment
- **Class Imbalance**: imbalanced-learn (imblearn)
- **Visualization**: matplotlib, seaborn, plotly
- **Progress Tracking**: tqdm
- **Experiment Tracking**: wandb, mlflow (optional)

### Spanish Language Models
- **BETO**: Spanish BERT model
- **RoBERTa-es**: Spanish RoBERTa
- **mBERT**: Multilingual BERT
- **XLM-RoBERTa**: Cross-lingual RoBERTa

## Project Goals
1. Understand and explore the EXIST2025 dataset
2. Develop baseline models for Task 1 (binary classification)
3. Extend to Task 2 and Task 3 (hierarchical classification)
4. Evaluate model performance on development set
5. Generate predictions for test data
6. Analyze model errors and biases

## Implementation Approach

### Jupyter Notebook Workflow
This project is designed to be implemented in Jupyter Notebooks for better experimentation and visualization:

#### Recommended Notebook Structure

1. **01_data_exploration.ipynb**:
   - Load and explore dataset structure
   - Analyze label distributions
   - Visualize annotator agreement
   - Identify class imbalance
   - Text statistics (length, vocabulary, common words)
   - Dataset split verification

2. **02_preprocessing.ipynb**:
   - Text cleaning and normalization
   - Tokenization experiments
   - Handle URLs, mentions, hashtags
   - Create vocabulary
   - Save preprocessed data

3. **03_baseline_models.ipynb**:
   - Traditional ML baselines (Logistic Regression, SVM)
   - TF-IDF feature extraction
   - Baseline metrics for comparison
   - Error analysis

4. **04_deep_learning_models.ipynb**:
   - LSTM/GRU implementations
   - Word embeddings (Word2Vec, FastText)
   - Training and evaluation
   - Hyperparameter tuning

5. **05_transformer_models.ipynb**:
   - Fine-tune BETO/RoBERTa-es
   - Transfer learning experiments
   - Compare pre-trained models
   - Best model selection

6. **06_robustness_techniques.ipynb**:
   - Data augmentation experiments
   - Class balancing techniques
   - Ensemble methods
   - Final model optimization

7. **07_hierarchical_tasks.ipynb**:
   - Implement Task 2 and Task 3
   - Hierarchical model architecture
   - Multi-task learning experiments
   - Combined evaluation

8. **08_final_predictions.ipynb**:
   - Generate predictions on test set
   - Format outputs correctly
   - Create submission files
   - Final evaluation and analysis

#### Benefits of Notebook Implementation
- **Interactive exploration**: Visualize data and results inline
- **Incremental development**: Test ideas cell-by-cell
- **Documentation**: Mix code, results, and explanations
- **Reproducibility**: Save outputs and intermediate results
- **Collaboration**: Easy to share and review

#### Best Practices for Notebooks
- Use markdown cells to explain each section
- Keep cells focused and modular
- Save intermediate results (models, preprocessed data)
- Use version control (git) for notebook files
- Clear outputs before committing to reduce file size
- Add requirements.txt or environment.yml for dependencies

## Notes for AI Assistant
- This is a Master's level AI/NLP course assignment
- Focus is on understanding sexism detection in Spanish social media
- Consider ethical implications of the task
- Pay attention to the hierarchical nature of the tasks
- Training data has ~139,475 lines (many tweets)
- Test data has ~12,145 lines
- Label disagreement among annotators is normal and expected
