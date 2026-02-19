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

## Notes for AI Assistant
- This is a Master's level AI/NLP course assignment
- Focus is on understanding sexism detection in Spanish social media
- Consider ethical implications of the task
- Pay attention to the hierarchical nature of the tasks
- Training data has ~139,475 lines (many tweets)
- Test data has ~12,145 lines
- Label disagreement among annotators is normal and expected
