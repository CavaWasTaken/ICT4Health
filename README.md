# ICT for Health - Laboratory Activities

This repository contains laboratory activities focused on applying Information and Communication Technologies to healthcare applications, using machine learning, image processing, and signal analysis techniques on real medical datasets.

---

## Lab 1: Linear Regression on Parkinson's Disease Data

### Medical Background

**Parkinson's Disease** is a neurodegenerative disorder affecting motor control:
- **Symptoms**: Tremor, difficulty walking, problems initiating movement, impaired speech control
- **Treatment**: Levodopa medication (absorption challenges as disease progresses)
- **Assessment**: UPDRS (Unified Parkinson's Disease Rating Scale)
  - Total UPDRS: Overall disease severity score
  - Motor UPDRS: Motor function specific score
  - Evaluated by neurologists through various movement tests and quality of life assessments

### Lab Objective

Predict **total UPDRS** from voice parameters and other features using linear regression, enabling:
- Objective, automatic scoring without neurologist visit
- Multiple daily measurements via smartphone voice recordings
- Treatment optimization assistance
- **Note**: Method applicable only to patients whose voice is affected by Parkinson's

### Dataset

**Source**: UCI Machine Learning Repository - Parkinsons Telemonitoring Dataset
- **File**: `parkinsons_updrs_av.csv`
- **Structure**: Multiple rows per patient (measurements over 6 months)
- **Features**:
  - Subject information: `subject#`, `age`, `sex`
  - Time: `test_time` (days since enrollment)
  - Target variables: `total_UPDRS`, `motor_UPDRS`
  - Voice parameters (22 features):
    - **Jitter**: `Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`
    - **Shimmer**: `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA`
    - **Other**: `NHR`, `HNR`, `RPDE`, `DFA`, `PPE`

### Data Analysis Workflow

#### 1. Data Exploration with Pandas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("parkinsons_updrs_av.csv")
```

**Essential Pandas Operations**:
- `X.head()`: View first rows
- `X.info()`: Check data types and missing values
- `X.describe().T`: Statistical summary (min, max, mean, std)
- `X.plot.hist(bins=50)`: Feature distributions
- `X.plot.scatter('feature_a', 'feature_b')`: Pairwise relationships
- `X.cov()`: Covariance matrix
- `X.values`: Convert to NumPy array

#### 2. Data Preparation
- **Check for issues**: Missing values, out-of-scale values, data quality
- **Feature selection**: Remove unwanted features (e.g., `subject#` as identifier)
- **Data shuffling**: Randomize order to avoid bias
- **Train-test split**: Typically 70-80% training, 20-30% testing

#### 3. Data Normalization
**Why normalize?**
- Features have different scales (age in years, voice parameters in different units)
- Ensures equal contribution of all features
- Improves numerical stability

**Method**: Z-score normalization
```
x_normalized = (x - mean) / std
```

**Important**: Calculate mean and std from **training set only**, apply to all datasets

#### 4. Linear Regression Methods

##### Method 1: Linear Least Squares (LLS)
**Analytical solution**:
```
w = (A^T A)^(-1) A^T y
```
Where:
- A: Design matrix (training data with features)
- y: Target vector (total UPDRS values)
- w: Weight vector (coefficients)

**Prediction**:
```
ŷ = x^T w
```

##### Method 2: Ridge Regression (Regularized LLS)
**With regularization term**:
```
w = (A^T A + λI)^(-1) A^T y
```
- λ (lambda): Regularization parameter (e.g., 10^-8)
- I: Identity matrix
- **Purpose**: Prevent overfitting, improve numerical stability

#### 5. Performance Evaluation

**Error Metrics**:
- **Mean error**: Average prediction error
- **Standard deviation of error**: Error variability
- **Mean Square Error (MSE)**: Average of squared errors
- **Root Mean Square Error (RMSE)**: √MSE

**Statistical Measures**:
- **Correlation coefficient (r)**: Linear relationship strength
- **Coefficient of determination (R²)**: Proportion of variance explained
  - R² = 1: Perfect prediction
  - R² = 0: No better than mean prediction
  - R² < 0: Worse than mean prediction

**Visualizations**:
- **Scatter plot**: True vs predicted UPDRS
- **Regression line**: Best fit line through predictions
- **Error histogram**: Distribution of prediction errors
- **Residual plots**: Error patterns over time or vs predictions

### Implementation Tasks

1. **Data loading and exploration**: Use Pandas to understand dataset structure
2. **Data cleaning**: Handle missing values, remove outliers
3. **Feature engineering**: Select relevant features for regression
4. **Normalization**: Standardize features using training set statistics
5. **Model training**: Implement LLS and Ridge regression
6. **Prediction**: Generate UPDRS estimates for test set
7. **Evaluation**: Compute metrics and create visualization plots
8. **Analysis**: Interpret results, identify strengths and limitations

---

## Lab 2: K-Nearest Neighbors Regression on Parkinson's Dataset

### Motivation

**Concept**: "Each function is linear if you zoom enough"

Based on Taylor series approximation:
```
f(x) ≈ f(x₀) + ∇f(x₀)(x - x₀)
```

Valid only for points **close to** x₀.

### Main Idea

Apply linear regression using only the **K nearest neighbors** to each test point:
- Each test point gets its own weight vector w(x₀)
- Training matrix has K rows (selected neighbors) instead of N_train rows
- **Local linearity**: Better captures non-linear relationships

### Dataset Division

**Three-way split**:
- **Training set**: 50% (true training data for building models)
- **Validation set**: 25% (hyperparameter optimization - selecting K)
- **Test set**: 25% (final performance evaluation)

**Effective split**: 75% training + validation, 25% test

### Algorithm Steps

#### For Each Validation/Test Point x:

1. **Compute distances** to all training points:
   ```python
   distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
   # or square distance (no sqrt needed for comparison)
   distances_sq = np.sum((X_train - x)**2, axis=1)
   ```

2. **Select K nearest neighbors**:
   ```python
   nearest_indices = np.argsort(distances)[:K]
   A = X_train[nearest_indices]  # K × F matrix
   y_neighbors = y_train[nearest_indices]  # K × 1 vector
   ```

3. **Train local linear model** (Ridge regression):
   ```python
   epsilon = 1e-8
   w = np.linalg.inv(A.T @ A + epsilon * I) @ A.T @ y_neighbors
   ```

4. **Predict**:
   ```python
   y_pred = x.T @ w
   error = y_true - y_pred
   ```

5. **Aggregate**: Compute MSE over all validation/test points

### K Optimization Process

1. **Initial exploration**: Try extreme values (e.g., K=5, K=50, K=200)
2. **Define range**: Set reasonable K_min and K_max based on dataset size
3. **Grid search**: Loop through K values with appropriate step
4. **Validation**: Compute MSE for each K on validation set
5. **Selection**: Choose K_opt with minimum validation MSE
6. **Visualization**: Plot validation MSE vs K

**Considerations**:
- **Small K**: Low bias, high variance (overfitting risk)
- **Large K**: High bias, low variance (underfitting risk)
- **Optimal K**: Balance between bias and variance

### Comparison with Standard LLS

**Task**: Train standard LLS on full 75% training+validation data
- Use same test set for fair comparison
- Compare metrics: MSE, R², correlation coefficient
- Analyze error distributions (histograms)
- Evaluate scatter plots (true vs predicted)

**Expected insights**:
- KNN-LLS may capture local patterns better
- Standard LLS has single global model
- Trade-offs: computational cost vs accuracy
- Dataset-specific performance

---

## Lab 3: Image Segmentation of Skin Moles

### Medical Context

**Melanoma Detection** using dermatological imaging:
- **Classification**: Low risk, medium risk, melanoma (high risk)
- **ABCDE criteria** used by dermatologists:
  - **A**symmetry: Irregular shape
  - **B**order: Irregular, jagged edges
  - **C**olor: Multiple colors or uneven distribution
  - **D**iameter: Larger size
  - **E**volution: Changes over time

### Lab Objective

**Automatic mole segmentation** to extract quantitative features:
- Identify mole region in dermoscopic images
- Measure border properties (indentation ratio)
- Enable objective feature extraction for diagnosis
- Support automated melanoma risk assessment

### Dataset

**File**: `images.zip` containing JPEG images:
- `low_risk_n.jpg`: Low melanoma probability
- `medium_risk_n.jpg`: Medium melanoma probability
- `melanoma_n.jpg`: High melanoma probability

**Image format**: RGB (583×583×3 typical size)
- Each pixel: [R, G, B] values (0-255, uint8)
- [0,0,0]: Black
- [255,255,255]: White

### Segmentation Pipeline

#### Step 1: Color Quantization with K-Means

**Objective**: Reduce image to 3 representative colors

```python
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

# Load image
im_or = mpimg.imread(filename)  # Shape: (N1, N2, 3)

# Reshape for K-means (requires 2D input)
N1, N2, N3 = im_or.shape
im_2D = im_or.reshape((N1 * N2, N3))  # (N1*N2) × 3

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(im_2D)

# Get results
centroids = kmeans.cluster_centers_.astype('uint8')  # 3 colors
labels = kmeans.labels_  # Cluster assignment for each pixel
```

**Output**: 
- 3 centroids representing quantized colors (typically: mole, skin, shadow)
- Cluster labels for each pixel

**Quantized image reconstruction**:
```python
im_2D_quant = im_2D.copy()
for kc in range(3):
    im_2D_quant[kmeans.labels_ == kc, :] = centroids[kc, :]
im_quant = im_2D_quant.reshape((N1, N2, N3))
```

#### Step 2: Identify Darkest Cluster (Mole)

**Convert RGB to grayscale**:
```python
conv_to_gray = np.array([0.2125, 0.7154, 0.0721])
centroids_gray = centroids @ conv_to_gray
i_col = centroids_gray.argmin()  # Index of darkest cluster
```

**Extract mole pixel positions**:
```python
im_clust = kmeans.labels_.reshape(N1, N2)
mole_pos = np.argwhere(im_clust == i_col)  # (N_pixels, 2) array
```

**Problem**: May include shadows or other dark regions, not just the mole

#### Step 3: Spatial Clustering with DBSCAN

**Objective**: Separate spatially distinct dark regions

```python
from sklearn.cluster import DBSCAN

clusters = DBSCAN(eps=?, min_samples=?).fit(mole_pos)
```

**Parameters to tune**:
- `eps`: Maximum distance for neighborhood
- `min_samples`: Minimum points for core point

**Output**:
- Cluster labels: 0, 1, 2, ..., N-1 (valid clusters)
- Label -1: Outliers (isolated pixels)

#### Step 4: Select Mole Cluster

**Selection criteria**:

1. **Minimum size**: Mole should have ≥ 1000 pixels (~30×30 pixels)
   ```python
   cluster_sizes = [(clusters.labels_ == i).sum() 
                    for i in range(max(clusters.labels_) + 1)]
   valid_clusters = [i for i, size in enumerate(cluster_sizes) 
                     if size >= 1000]
   ```

2. **Compactness** (low moment of inertia):
   ```python
   def moment_of_inertia(points):
       center = points.mean(axis=0)
       return np.sum((points - center)**2)
   ```
   - Mole has compact, roughly circular shape
   - Calculate for each valid cluster
   - Select cluster with lowest moment of inertia

**Extract final mole**:
```python
i_mole = # selected cluster index
true_mole_pos = mole_pos[clusters.labels_ == i_mole]

# Create binary mask
im_mole_pos = np.zeros((N1, N2))
x, y = true_mole_pos[:, 0], true_mole_pos[:, 1]
im_mole_pos[x, y] = 1

# Extract mole with original colors
im_only_mole = 255 * np.ones_like(im_or)  # White background
im_only_mole[x, y, :] = im_or[x, y, :]
```

#### Step 5: Border Refinement (Optional)

**Smoothing filters**:
- Morphological operations (erosion, dilation)
- Median filtering
- Gaussian smoothing

**Border extraction**:
- Edge detection algorithms
- Contour tracing
- Gradient-based methods

### Feature Extraction

Once segmented, compute diagnostic features:

1. **Border irregularity**:
   ```
   ratio = perimeter / (2π√(area/π))
   ```
   - Perfect circle: ratio = 1
   - Higher ratio → more irregular border

2. **Asymmetry index**: Compare mole halves

3. **Color variation**: Analyze RGB distribution within mole

4. **Size**: Compute area in mm²

5. **Shape descriptors**: Circularity, compactness, eccentricity

### Implementation Tasks

1. **Load and visualize**: All mole images (low risk, medium risk, melanoma)
2. **K-means clustering**: Implement color quantization
3. **Darkest cluster extraction**: Identify mole candidate pixels
4. **DBSCAN clustering**: Separate spatial regions
5. **Mole selection**: Apply size and compactness criteria
6. **Visualization**: Display segmented moles
7. **Feature computation**: Calculate border irregularity
8. **Analysis**: Compare features across risk categories

---

## Lab 4: Classification of Chronic Kidney Disease

### Medical Background

**Chronic Kidney Disease (CKD)**: Progressive loss of kidney function
- Affects millions worldwide
- Early detection crucial for treatment
- Diagnosis based on blood and urine tests

### Lab Structure

**Two parts**:
1. **Data cleaning**: Handle missing values, categorical encoding
2. **Decision tree classification**: Build interpretable diagnostic model

---

### Part 1: Data Cleaning and Preparation

#### Dataset Overview

**Source**: UCI Machine Learning Repository
- **File**: `chronic_kidney_disease.arff` (ARFF format)
- **Samples**: 400 patients
  - 250 CKD patients
  - 150 healthy (notckd)
- **Features**: 24 clinical features + 1 class label
  - 11 numerical features
  - 13 categorical features

#### Features Description

**Numerical Features**:
- `age`: Patient age (years)
- `bp`: Blood pressure (mm/Hg)
- `bgr`: Blood glucose random (mg/dl)
- `bu`: Blood urea (mg/dl)
- `sc`: Serum creatinine (mg/dl)
- `sod`: Sodium (mEq/L)
- `pot`: Potassium (mEq/L)
- `hemo`: Hemoglobin (gms)
- `pcv`: Packed cell volume
- `wbcc`: White blood cell count
- `rbcc`: Red blood cell count (million/cmm)

**Categorical Features**:
- `sg`: Specific gravity {1.005, 1.010, 1.015, 1.020, 1.025}
- `al`: Albumin {0, 1, 2, 3, 4, 5}
- `su`: Sugar {0, 1, 2, 3, 4, 5}
- `rbc`: Red blood cells {normal, abnormal}
- `pc`: Pus cell {normal, abnormal}
- `pcc`: Pus cell clumps {present, notpresent}
- `ba`: Bacteria {present, notpresent}
- `htn`: Hypertension {yes, no}
- `dm`: Diabetes mellitus {yes, no}
- `cad`: Coronary artery disease {yes, no}
- `appet`: Appetite {good, poor}
- `pe`: Pedal edema {yes, no}
- `ane`: Anemia {yes, no}

**Class**: `classk` {ckd, notckd}

#### Data Loading Challenges

**Issues in original file**:
1. First 29 lines contain metadata (skip them)
2. Extra commas in some rows (lines 99, 102)
3. Double commas instead of single (line 399)
4. Missing values marked as `?`
5. Inconsistent whitespace (e.g., " yes", "yes ", "\tyes")

**Pandas solution**:
```python
import pandas as pd

feat_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
              'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 
              'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classk']

xx = pd.read_csv("chronic_kidney_disease_v2.arff", 
                 sep=',', 
                 skiprows=29, 
                 names=feat_names,
                 header=None, 
                 na_values=['?', '\t?'])
```

**Manual preprocessing**: Edit original file to fix lines 99, 102, 399 → save as `chronic_kidney_disease_v2.arff`

#### Categorical Encoding

**Convert text to numbers** (required for scikit-learn):

```python
mapping = {
    'normal': 0, 'abnormal': 1,
    'present': 1, 'notpresent': 0,
    'yes': 1, ' yes': 1, '\tyes': 1,
    'no': 0, '\tno': 0,
    'ckd': 1, 'ckd\t': 1, 'notckd': 0,
    'poor': 1, 'good': 0
}

xx = xx.replace(mapping.keys(), mapping.values())
```

**Verify encoding**:
```python
print(xx.nunique())  # Check cardinality of each feature
```

#### Missing Value Analysis

**Check missing data**:
```python
xx.info()  # Shows non-null counts per column
missing_counts = xx.isnull().sum()
missing_percentages = (missing_counts / len(xx)) * 100
```

**Common strategies**:
1. **Remove samples** with too many missing features
2. **Remove features** with too many missing values
3. **Imputation**:
   - Mean/median (numerical features)
   - Mode (categorical features)
   - KNN imputation
   - Predictive models

```python
# Example: Mean imputation
xx['bu'].fillna(xx['bu'].mean(), inplace=True)

# Example: Mode imputation
xx['rbc'].fillna(xx['rbc'].mode()[0], inplace=True)
```

#### Feature Analysis

**Statistical summary**:
```python
xx.describe().T  # Statistics for numerical features
```

**Correlation analysis**:
```python
correlation_matrix = xx.corr()
# Identify highly correlated features (potential redundancy)
```

**Class distribution**:
```python
class_counts = xx['classk'].value_counts()
# Check for class imbalance
```

---

### Part 2: Decision Tree Classification

#### Why Decision Trees?

**Advantages**:
- **Interpretable**: Easy to visualize and explain to clinicians
- **Handle mixed data**: Both numerical and categorical features
- **No normalization needed**: Scale-invariant
- **Feature importance**: Identify most diagnostic features
- **Non-linear relationships**: Capture complex patterns

#### Decision Tree Fundamentals

**Structure**:
- **Root node**: Entire dataset
- **Internal nodes**: Feature-based splits
- **Branches**: Decision rules
- **Leaf nodes**: Class predictions

**Splitting criteria**:
- **Information Gain** (based on entropy)
- **Gini Impurity**
- **Mutual Information**

**Entropy**:
```
H(S) = -Σ p_i log₂(p_i)
```
Where p_i is the proportion of class i in set S

**Information Gain**:
```
IG(S, A) = H(S) - Σ |S_v|/|S| × H(S_v)
```
Where A is feature, S_v is subset after split on value v

#### Implementation with Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prepare data
X = xx.drop('classk', axis=1)  # Features
y = xx['classk']  # Target

# Handle remaining missing values (if any)
# Option 1: Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Option 2: Imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train decision tree
dt = DecisionTreeClassifier(
    criterion='entropy',  # or 'gini'
    max_depth=5,  # Limit tree depth to prevent overfitting
    min_samples_split=20,  # Minimum samples to split node
    min_samples_leaf=10,  # Minimum samples in leaf
    random_state=42
)

dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['notckd', 'ckd']))
```

#### Tree Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=X.columns, 
          class_names=['notckd', 'ckd'],
          filled=True, 
          rounded=True,
          fontsize=10)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Feature Importance

```python
# Get feature importances
importances = dt.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'][:10], 
         feature_importance_df['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

#### Hyperparameter Tuning

**Grid Search** for optimal parameters:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_dt = grid_search.best_estimator_
```

#### Overfitting Prevention

**Strategies**:
1. **Pruning**: Limit tree depth and node splits
2. **Cross-validation**: Assess generalization
3. **Ensemble methods**: Random Forest, Gradient Boosting

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
```

#### Model Interpretation

**Extract decision rules**:
```python
from sklearn.tree import export_text

tree_rules = export_text(dt, feature_names=list(X.columns))
print(tree_rules)
```

**Example interpretation**:
```
|--- sc <= 1.2
|   |--- hemo <= 12.5
|   |   |--- class: ckd
|   |--- hemo > 12.5
|   |   |--- class: notckd
|--- sc > 1.2
|   |--- class: ckd
```

### Implementation Tasks

1. **Load and clean data**: Handle ARFF format, missing values, encoding
2. **Exploratory analysis**: Statistics, correlations, distributions
3. **Missing value treatment**: Choose appropriate strategy
4. **Feature encoding**: Convert categorical to numerical
5. **Train-test split**: Proper data division
6. **Build decision tree**: Train classifier with appropriate parameters
7. **Visualization**: Plot tree structure
8. **Evaluation**: Compute metrics, confusion matrix
9. **Feature analysis**: Identify most important diagnostic features
10. **Optimization**: Tune hyperparameters for best performance
11. **Clinical interpretation**: Extract actionable medical insights

---

## Lab 5: ROC Analysis for COVID-19 Serological Tests

### Medical Context

**COVID-19 Diagnostic Testing**:
- **Naso-pharyngeal swabs**: Gold standard (high sensitivity and specificity)
- **Serological tests**: Blood tests detecting antibodies (IgG)
  - Faster results (especially early in pandemic)
  - Less invasive
  - Variable reliability

### Lab Objective

Compare two serological tests by:
1. Computing ROC (Receiver Operating Characteristic) curves
2. Setting optimal thresholds for positive/negative classification
3. Determining which test is more reliable

**Ground truth**: Naso-pharyngeal swab results

### Dataset

**File**: `covid_serological_results.csv`

**Columns**:
- `COVID_swab_res`: Swab test result
  - 0: Negative
  - 1: Uncertain (excluded from analysis)
  - 2: Positive
- `IgG_test1_titre`: Antibody level in Test #1 (range: 2.5 - 314)
- `IgG_test2_titre`: Antibody level in Test #2 (range: 0 - 9.71)

**Sample size**: 879 patients (after removing uncertain swab results)

**Note**: Different scale for each test (arbitrary units) → different thresholds needed

### Binary Classification Fundamentals

#### Confusion Matrix

For a given threshold:

|                    | Predicted Positive | Predicted Negative |
|--------------------|-------------------|--------------------|
| **Actually Positive (D)** | True Positive (TP) | False Negative (FN) |
| **Actually Negative (H)** | False Positive (FP) | True Negative (TN) |

#### Performance Metrics

**Sensitivity** (True Positive Rate, Recall):
```
Sensitivity = P(T+ | D) = TP / (TP + FN)
```
- Proportion of sick patients correctly identified
- High sensitivity → few false negatives

**Specificity** (True Negative Rate):
```
Specificity = P(T- | H) = TN / (TN + FP)
```
- Proportion of healthy patients correctly identified
- High specificity → few false positives

**False Positive Rate**:
```
FPR = P(T+ | H) = 1 - Specificity = FP / (TN + FP)
```
- Proportion of healthy patients incorrectly classified as positive

**Trade-off**: Lowering threshold increases sensitivity but decreases specificity

### Data Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
xx = pd.read_csv("covid_serological_results.csv")

# Remove uncertain swab results
xx = xx[xx.COVID_swab_res != 1]

# Remap: 2 → 1 (positive)
xx.COVID_swab_res[xx.COVID_swab_res == 2] = 1

# Extract arrays
swab = xx.COVID_swab_res.values  # Ground truth
Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values
```

**Exploratory analysis**:
```python
xx.describe()
pd.plotting.scatter_matrix(xx, figsize=(10, 10))
plt.show()
```

### Outlier Removal

**Identify outliers** using DBSCAN on normalized data:

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# Min-max normalization (0 to 1)
scaler = MinMaxScaler()
xx_norm = scaler.fit_transform(xx[['COVID_swab_res', 
                                     'IgG_Test1_titre', 
                                     'IgG_Test2_titre']])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5)
labels = dbscan.fit_predict(xx_norm)

# Remove outliers (label == -1)
xx_clean = xx[labels != -1]
swab = xx_clean.COVID_swab_res.values
Test1 = xx_clean.IgG_Test1_titre.values
Test2 = xx_clean.IgG_Test2_titre.values
```

**Note**: Normalized data only for outlier detection; use original data for analysis

### Computing Sensitivity and Specificity

**For a single threshold**:

```python
def compute_metrics(test_values, ground_truth, threshold):
    """
    Compute sensitivity and specificity for given threshold.
    
    Parameters:
    - test_values: Array of test results (IgG levels)
    - ground_truth: Array of true labels (0: negative, 1: positive)
    - threshold: Classification threshold
    
    Returns:
    - sensitivity, specificity
    """
    # Separate by ground truth
    x_positive = test_values[ground_truth == 1]  # Sick patients
    x_negative = test_values[ground_truth == 0]  # Healthy patients
    
    N_positive = len(x_positive)
    N_negative = len(x_negative)
    
    # True positives and sensitivity
    TP = np.sum(x_positive > threshold)
    sensitivity = TP / N_positive
    
    # True negatives and specificity
    TN = np.sum(x_negative < threshold)
    specificity = TN / N_negative
    
    return sensitivity, specificity

# Example
sensitivity, specificity = compute_metrics(Test2, swab, threshold=5)
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
```

**For multiple thresholds**:

```python
def sensitivity_specificity_curve(test_values, ground_truth):
    """
    Compute sensitivity and specificity for all possible thresholds.
    
    Returns:
    - thresholds, sensitivities, specificities
    """
    # Use sorted test values as thresholds (include 0)
    thresholds = np.sort(np.unique(test_values))
    thresholds = np.concatenate([[0], thresholds])
    
    sensitivities = []
    specificities = []
    
    for thresh in thresholds:
        sens, spec = compute_metrics(test_values, ground_truth, thresh)
        sensitivities.append(sens)
        specificities.append(spec)
    
    return thresholds, np.array(sensitivities), np.array(specificities)

# Compute for Test2
thresholds, sensitivities, specificities = sensitivity_specificity_curve(
    Test2, swab)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', linewidth=2)
plt.plot(thresholds, specificities, label='Specificity', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Sensitivity and Specificity vs Threshold (Test 2)')
plt.legend()
plt.grid(True)
plt.show()
```

### ROC Curve Construction

**ROC**: Sensitivity vs False Positive Rate

```python
def plot_roc_curve(test_values, ground_truth, test_name):
    """
    Plot ROC curve for given test.
    
    Returns AUC (Area Under Curve).
    """
    thresholds, sensitivities, specificities = \
        sensitivity_specificity_curve(test_values, ground_truth)
    
    # False Positive Rate = 1 - Specificity
    fpr = 1 - specificities
    tpr = sensitivities  # True Positive Rate = Sensitivity
    
    # Plot ROC
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'{test_name}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')  # Diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title(f'ROC Curve - {test_name}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    
    # Compute AUC
    auc = compute_auc(fpr, tpr)
    print(f"AUC for {test_name}: {auc:.4f}")
    
    return fpr, tpr, thresholds, auc

# Plot for Test2
fpr2, tpr2, thresh2, auc2 = plot_roc_curve(Test2, swab, 'Test 2')
```

**Interpretation**:
- **Ideal point**: (0, 1) - perfect classifier (top-left corner)
- **Diagonal line**: Random classifier (AUC = 0.5)
- **Better curve**: Closer to top-left corner, higher AUC

### Area Under ROC Curve (AUC)

**Trapezoidal rule** for numerical integration:

```python
def compute_auc(x, y):
    """
    Compute area under curve using trapezoidal rule.
    
    Parameters:
    - x: x-coordinates (sorted)
    - y: y-coordinates
    
    Returns:
    - area: Area under curve
    """
    # Sort by x (if not already sorted)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Trapezoidal integration
    area = 0
    for i in range(len(x_sorted) - 1):
        dx = x_sorted[i+1] - x_sorted[i]
        avg_height = (y_sorted[i] + y_sorted[i+1]) / 2
        area += dx * avg_height
    
    return area
```

**Verification with scikit-learn**:
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr_sk, tpr_sk, thresh_sk = roc_curve(swab, Test2, pos_label=1)
auc_sk = roc_auc_score(swab, Test2)

print(f"Scikit-learn AUC: {auc_sk:.4f}")
print(f"Manual AUC: {compute_auc(fpr2, tpr2):.4f}")
```

### Threshold Selection Strategies

#### Strategy 1: Fixed Sensitivity

Set required sensitivity (e.g., 95%), accept resulting specificity:

```python
def threshold_for_sensitivity(test_values, ground_truth, target_sens):
    """Find threshold that achieves target sensitivity."""
    thresh, sens, spec = sensitivity_specificity_curve(test_values, ground_truth)
    
    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(sens - target_sens))
    
    return thresh[idx], sens[idx], spec[idx]

# Example: 95% sensitivity
thresh_95, sens_95, spec_95 = threshold_for_sensitivity(Test2, swab, 0.95)
print(f"Threshold for 95% sensitivity: {thresh_95:.2f}")
print(f"Resulting specificity: {spec_95:.4f}")
```

#### Strategy 2: Fixed Specificity

Set required specificity (e.g., 95%), accept resulting sensitivity:

```python
def threshold_for_specificity(test_values, ground_truth, target_spec):
    """Find threshold that achieves target specificity."""
    thresh, sens, spec = sensitivity_specificity_curve(test_values, ground_truth)
    
    idx = np.argmin(np.abs(spec - target_spec))
    return thresh[idx], sens[idx], spec[idx]
```

#### Strategy 3: Equal Sensitivity and Specificity

**Balanced performance**:

```python
def threshold_equal_sens_spec(test_values, ground_truth):
    """Find threshold where sensitivity equals specificity."""
    thresh, sens, spec = sensitivity_specificity_curve(test_values, ground_truth)
    
    # Find where |sensitivity - specificity| is minimum
    diff = np.abs(sens - spec)
    idx = np.argmin(diff)
    
    return thresh[idx], sens[idx], spec[idx]

# Apply to Test2
thresh_eq, sens_eq, spec_eq = threshold_equal_sens_spec(Test2, swab)
print(f"Equal sens/spec threshold: {thresh_eq:.2f}")
print(f"Sensitivity = Specificity = {sens_eq:.4f}")
```

#### Strategy 4: Optimal Point (Youden's Index)

**Maximize**: Sensitivity + Specificity - 1

```python
def threshold_youden(test_values, ground_truth):
    """Find threshold maximizing Youden's J statistic."""
    thresh, sens, spec = sensitivity_specificity_curve(test_values, ground_truth)
    
    # Youden's index: J = sensitivity + specificity - 1
    youden = sens + spec - 1
    idx = np.argmax(youden)
    
    return thresh[idx], sens[idx], spec[idx], youden[idx]

thresh_opt, sens_opt, spec_opt, j_opt = threshold_youden(Test2, swab)
print(f"Optimal threshold (Youden): {thresh_opt:.2f}")
print(f"Sensitivity: {sens_opt:.4f}, Specificity: {spec_opt:.4f}")
print(f"Youden's J: {j_opt:.4f}")
```

### Comparison of Two Tests

**Task**: Compare Test1 vs Test2

```python
# Compute ROC for both tests
fpr1, tpr1, _, auc1 = plot_roc_curve(Test1, swab, 'Test 1')
fpr2, tpr2, _, auc2 = plot_roc_curve(Test2, swab, 'Test 2')

# Compare on same plot
plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, linewidth=2, label=f'Test 1 (AUC={auc1:.3f})')
plt.plot(fpr2, tpr2, linewidth=2, label=f'Test 2 (AUC={auc2:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('Sensitivity')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Decision
if auc1 > auc2:
    print(f"Test 1 is better (AUC: {auc1:.4f} vs {auc2:.4f})")
elif auc2 > auc1:
    print(f"Test 2 is better (AUC: {auc2:.4f} vs {auc1:.4f})")
else:
    print("Tests are equivalent")
```

### Critical Thinking

**For COVID-19 testing, which is more important?**

**Scenario Analysis**:

1. **High sensitivity priority** (minimize false negatives):
   - **Rationale**: Missing infected patients → disease spread
   - **Application**: Screening in high-risk populations
   - **Trade-off**: More false positives → unnecessary isolation/treatment

2. **High specificity priority** (minimize false positives):
   - **Rationale**: Avoid unnecessary panic/treatment
   - **Application**: Confirmatory testing
   - **Trade-off**: More false negatives → missed cases

3. **Balanced approach**:
   - **Rationale**: Both errors have consequences
   - **Application**: General population screening with follow-up testing

**Recommendation**: For COVID-19, **high sensitivity** typically preferred to prevent transmission, with confirmatory testing if positive.

### Implementation Tasks

1. **Load and explore data**: CSV reading, basic statistics
2. **Data cleaning**: Remove uncertain swab results, outlier detection
3. **Implement metrics function**: Sensitivity, specificity for single threshold
4. **Curve computation**: Sweep through all thresholds
5. **Visualize**: Plot sensitivity and specificity vs threshold
6. **ROC construction**: Plot TPR vs FPR
7. **AUC calculation**: Implement trapezoidal integration
8. **Verification**: Compare with scikit-learn
9. **Threshold selection**: Implement all four strategies
10. **Test comparison**: Analyze Test1 vs Test2
11. **Critical analysis**: Justify threshold choice for COVID-19 context

---

## Lab 6: Independent Component Analysis (ICA) and EEG Signal Processing

### Background

**Electroencephalography (EEG)**: Recording electrical activity of the brain
- **Non-invasive**: Electrodes placed on scalp
- **High temporal resolution**: Millisecond precision
- **Applications**: 
  - Clinical diagnosis (epilepsy, sleep disorders)
  - Brain-computer interfaces
  - Cognitive neuroscience research

### Challenge: Signal Mixing

**Problem**: EEG electrodes record **mixed signals**:
- Multiple brain sources (cortical regions)
- Eye movements (EOG artifacts)
- Muscle activity (EMG artifacts)
- Cardiac signals (ECG artifacts)
- Line noise (50/60 Hz)

**Solution**: Independent Component Analysis (ICA) separates mixed signals into independent sources

### Lab Structure

1. **Artificial example**: Understand ICA with synthetic signals
2. **EEGLAB**: Professional toolbox for EEG analysis
3. **Real EEG data**: Apply ICA to separate brain sources from artifacts

---

### Part 1: ICA on Artificial Signals

#### Signal Generation

**Four independent source signals**:

1. **Sinusoidal**: `s1(t) = sin(2πf₁t)`
2. **Square wave**: Alternating min/max values
3. **Sawtooth wave**: Linear ramp
4. **Triangular wave**: Symmetric triangular shape

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.decomposition import FastICA, PCA

# Time vector
fs = 1000  # Sampling frequency
t = np.linspace(0, 4, fs * 4)

# Generate sources
s1 = np.sin(2 * np.pi * 5 * t)  # Sinusoidal 5 Hz
s2 = scipy_signal.square(2 * np.pi * 2 * t)  # Square wave 2 Hz
s3 = scipy_signal.sawtooth(2 * np.pi * 3 * t)  # Sawtooth 3 Hz
s4 = scipy_signal.sawtooth(2 * np.pi * 4 * t, 0.5)  # Triangle 4 Hz

# Combine into matrix (N_samples × N_sources)
S = np.c_[s1, s2, s3, s4]

# Plot original signals
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
for i in range(4):
    axes[i].plot(t[:1000], S[:1000, i])
    axes[i].set_ylabel(f'Source {i+1}')
plt.xlabel('Time (s)')
plt.suptitle('Original Independent Sources')
plt.tight_layout()
plt.show()
```

#### Signal Mixing

**Random mixing matrix**:

```python
# Generate random mixing matrix A (4×4)
np.random.seed(42)
A = np.random.rand(4, 4)

# Mix signals: X = S @ A.T
# Each observed signal is a linear combination of sources
X = S @ A.T

# Plot mixed signals
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
for i in range(4):
    axes[i].plot(t[:1000], X[:1000, i])
    axes[i].set_ylabel(f'Mixed {i+1}')
plt.xlabel('Time (s)')
plt.suptitle('Mixed Observed Signals')
plt.tight_layout()
plt.show()
```

**Mixed signals** X now contain linear combinations of all sources.

#### Probability Distributions

**Analyze PDF of each source**:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i in range(4):
    axes[i].hist(S[:, i], bins=50, density=True, alpha=0.7)
    axes[i].set_title(f'Source {i+1} PDF')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Probability Density')

plt.tight_layout()
plt.show()
```

**Key observation**: 
- Sinusoidal: Bell-shaped (non-Gaussian)
- Square wave: Bimodal (two peaks)
- Sawtooth: Uniform distribution
- Triangle: Uniform distribution

**ICA assumption**: Sources are **non-Gaussian** (crucial for separation)

#### ICA with FastICA

```python
# Apply FastICA
ica = FastICA(n_components=4, random_state=42, max_iter=1000)
S_ica = ica.fit_transform(X)  # Recovered sources

# Estimated unmixing matrix
W = ica.components_  # 4×4 matrix

print("Mixing matrix A:")
print(A)
print("\nEstimated unmixing matrix W:")
print(W)
print("\nW @ A (should be close to permutation matrix):")
print(W @ A)

# Plot recovered sources
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
for i in range(4):
    axes[i].plot(t[:1000], S_ica[:1000, i])
    axes[i].set_ylabel(f'ICA {i+1}')
plt.xlabel('Time (s)')
plt.suptitle('FastICA Recovered Sources')
plt.tight_layout()
plt.show()
```

**Note**: Recovered sources may have:
- **Permutation ambiguity**: Order may differ from original
- **Scaling ambiguity**: Amplitude may differ
- **Sign ambiguity**: May be inverted

**Solution**: Visual inspection or correlation matching

#### Comparison with PCA

```python
# Apply PCA
pca = PCA(n_components=4)
S_pca = pca.fit_transform(X)

# Plot PCA components
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
for i in range(4):
    axes[i].plot(t[:1000], S_pca[:1000, i])
    axes[i].set_ylabel(f'PCA {i+1}')
plt.xlabel('Time (s)')
plt.suptitle('PCA Components')
plt.tight_layout()
plt.show()
```

**PCA vs ICA**:
- **PCA**: Maximizes variance, assumes Gaussian, orthogonal components
- **ICA**: Maximizes independence, assumes non-Gaussian, not necessarily orthogonal
- **Result**: ICA better recovers original sources when sources are independent and non-Gaussian

---

### Part 2: EEGLAB - Professional EEG Analysis Toolbox

#### Installation

**Requirements**: MATLAB with Signal Processing Toolbox

**Installation steps**:
1. Download EEGLAB from https://sccn.ucsd.edu/eeglab/downloadtoolbox.php
2. Unzip to folder (e.g., `C:/MATLAB/eeglab/`)
3. In MATLAB, navigate to EEGLAB folder
4. Type `eeglab` in command window

**Interface**: GUI with menus for all EEG processing operations

#### Dataset Description

**Experiment**:
- Subject views screen with 5 possible square locations
- Subject presses button when square appears
- 32 EEG channels recorded
- 80 trials (epochs)

**Data structure**:
- **Epochs**: 3-second segments (-1 to +2 seconds around square appearance)
- **Event markers**:
  - Green lines: Square appearance time (t=0)
  - Red lines: Button press time
- **Sampling rate**: 128 Hz (384 samples per epoch)

**Loading data**:
```
File → Load existing dataset → sample_data/eeglab_data_epochs_ica.set
```

#### EEG Visualization

**Scrolling data view**:
```
Plot → Channel data (scroll)
```
- Shows all 32 channels
- Green/red event markers visible
- Can scroll through all epochs

**Electrode locations**:
```
Plot → Channel locations → By name
```
- 2D projection of electrode positions on scalp
- Standard 10-20 system naming convention
- Frontal (F), Central (C), Parietal (P), Occipital (O) regions

#### Frequency Analysis

**Power spectrum**:
```
Plot → Channel Spectra and Maps
```
- Parameter 5: Frequency range `[0 64]` (Nyquist = fs/2 = 64 Hz)
- Power in dB: `10 log₁₀(Power)`
- Shows frequency content of each channel

**EEG frequency bands**:
- **δ (Delta)**: 0.5-4 Hz (deep sleep)
- **θ (Theta)**: 4-7 Hz (drowsiness, meditation)
- **α (Alpha)**: 8-15 Hz (relaxed, eyes closed)
- **β (Beta)**: 16-31 Hz (active thinking, concentration)
- **γ (Gamma)**: >31 Hz (cognitive processing)

#### Filtering

**Purpose**:
- Remove DC offset and slow trends (high-pass filter)
- Remove line noise at 60 Hz (notch or low-pass filter)
- Preserve EEG signals (1-50 Hz)

**Apply band-pass filter** (1-50 Hz):
```
Tools → Filter the data → Basic FIR filter
```
- Lower edge: `1` Hz
- Upper edge: `50` Hz
- FIR filter: Linear phase (constant delay)

**Result**: New filtered dataset created

**Verify filtering**:
```
Plot → Channel Spectra and Maps
```
- Power spectrum now limited to 1-50 Hz
- 60 Hz line noise removed

#### Event-Related Potentials (ERP)

**ERP**: Average signal across all epochs

**Compute and plot**:
```
Plot → Channel ERPs → With scalp maps
```

**Interpretation**:
- Shows average brain response to stimulus (square appearance)
- Different components (peaks/troughs) at different latencies:
  - **P100**: Positive peak ~100 ms (early visual processing)
  - **N200**: Negative peak ~200 ms (attention)
  - **P300**: Positive peak ~300 ms (cognitive processing, decision)
- Scalp maps show spatial distribution at specific latencies

**Modify latency**: Change parameter 2 from `NaN` to desired time (e.g., `200` for 200 ms)

#### ERP Image

**Visualization**:
```
Plot → Channel ERP Image
```
- Select channel (e.g., channel 1: Fpz, middle front)
- Shows:
  - **Top**: Average ERP (thick line)
  - **Bottom**: Color-coded image of all 80 individual trials
  - Each row = one trial
  - Color intensity = amplitude

**Insights**:
- Trial-to-trial variability
- Consistency of ERP components
- Outlier trials

#### Time-Frequency Analysis

**Spectrogram**: Power spectrum evolving over time

```
Plot → Channel time-frequency
```
- Select channel
- Window: Short-time FFT on sliding window
- Color represents power at each (time, frequency) point

**Interpretation** (example for Fpz):
- At ~300 ms post-stimulus: Increased power in low frequencies (<10 Hz)
- Corresponds to θ waves during cognitive processing

**More details**: https://eeglab.org/tutorials/08_Plot_data/Time-Frequency_decomposition.html

---

### Part 3: ICA on Real EEG Data

#### Why ICA for EEG?

**EEG signal = mixture**:
- Each scalp electrode records a weighted sum of multiple brain sources
- Artifacts (eye blinks, muscle, heartbeat) also mixed in
- Traditional spatial filters cannot fully separate sources

**ICA assumption**:
- Brain sources are spatially fixed and temporally independent
- Artifacts are also independent sources
- ICA can separate them

#### Applying ICA in EEGLAB

**Start ICA**:
```
Tools → Decompose Data by ICA
```
- Algorithm: Infomax (not FastICA)
- Leave default parameters
- Computes 32 independent components (ICs) for 32 channels

**Algorithm details**:
- Gradient descent optimization
- Maximizes joint entropy after sigmoid transformation
- Minimizes mutual information between components

**During computation**, MATLAB displays:
- **Learning rate**: Decreased during optimization
- **Weight change**: `||W_new - W_old||` (norm difference)
- **Angle**: Between current and previous W
  - If angle > 60°: Oscillating around optimum → reduce learning rate by ×0.98

**Convergence**: Typically takes several iterations

#### Visualizing ICA Components

**Component maps** (spatial filters):
```
Plot → Component maps → In 2D
```

**Interpretation**:
- Each map shows the **spatial pattern** (weights) for one IC
- Positive (red) and negative (blue) regions
- **Dipolar patterns**: Likely brain sources (e.g., frontal, parietal)
- **Frontal edge**: Eye movement artifacts
- **Temporal**: Muscle artifacts
- **Distributed**: Possible line noise or bad channels

**Component time series**:
```
Plot → Component activations (scroll)
```
- Shows IC time courses (like channel data, but for ICs)
- Easier to identify artifacts (stereotyped patterns)

**Component ERPs**:
```
Plot → Component ERPs → With component maps
```
- Average IC activity across epochs
- Brain ICs show ERP-like waveforms
- Artifact ICs show artifact-specific patterns

**Component spectra**:
```
Plot → Component spectra and maps
```
- Frequency content of each IC
- Eye artifacts: Low frequency (<4 Hz)
- Muscle artifacts: High frequency (>20 Hz)
- Line noise: Sharp peak at 60 Hz

#### Artifact Identification and Removal

**Identify artifact components**:

1. **Eye blinks/movements**:
   - Frontal topography (Fp1, Fp2 region)
   - Low frequency (< 4 Hz)
   - Stereotyped waveform (sharp peaks)

2. **Muscle activity**:
   - Temporal topography
   - High frequency (> 20 Hz)
   - Irregular, high-amplitude bursts

3. **Cardiac**:
   - Sometimes visible in frontal channels
   - ~1 Hz rhythmic pattern

**Reject artifact ICs**:
```
Tools → Reject data using ICA → Reject components by map
```
- Select IC numbers to reject (e.g., [1, 3, 5])
- EEGLAB removes these ICs from data
- **Reconstructs** EEG without artifacts

**Comparison**:
```
Plot → Channel data (scroll)
```
- Before ICA: Visible eye blinks, muscle noise
- After ICA: Cleaner EEG, preserved brain signals

#### Advanced: Source Localization

**DIPFIT** plugin (if available):
```
Tools → Locate dipoles using DIPFIT
```
- Fits equivalent current dipoles to IC spatial patterns
- Estimates brain regions generating each IC
- Requires head model and electrode locations

**Interpretation**: Localizes which brain areas contribute to each IC

### ICA Theory Recap

**Mathematical formulation**:
```
X = A × S
```
- X: Observed signals (N_channels × N_samples)
- A: Mixing matrix (N_channels × N_sources)
- S: Independent sources (N_sources × N_samples)

**Goal**: Estimate unmixing matrix W such that:
```
S_estimated = W × X
```

**Assumptions**:
1. Sources are **statistically independent**
2. Sources are **non-Gaussian** (at most one can be Gaussian)
3. Mixing is **linear** and **instantaneous**

**Algorithms**:
- **FastICA**: Fast fixed-point algorithm (Python default)
- **Infomax**: Gradient-based, maximizes entropy (EEGLAB default)
- **JADE**: Joint Approximate Diagonalization of Eigenmatrices

**Ambiguities**:
- **Permutation**: Can't determine source order
- **Scaling**: Can't determine absolute amplitude
- **Sign**: Source may be flipped

**Not a problem** for most applications (artifact removal, source identification)

### Implementation Tasks

#### Artificial Example:
1. **Generate sources**: Four distinct waveforms
2. **Analyze PDFs**: Verify non-Gaussian distributions
3. **Mix signals**: Random mixing matrix
4. **Apply FastICA**: Recover sources
5. **Compare with PCA**: Understand differences
6. **Visualize**: All signals and histograms

#### EEGLAB Real Data:
1. **Install EEGLAB**: Set up MATLAB environment
2. **Load dataset**: Sample EEG data with events
3. **Visualize raw data**: Scrolling view, electrode locations
4. **Frequency analysis**: Power spectra, identify noise
5. **Filter data**: Apply 1-50 Hz band-pass filter
6. **ERP analysis**: Compute and visualize ERPs
7. **Time-frequency**: Analyze spectrograms
8. **Run ICA**: Decompose into independent components
9. **Identify artifacts**: Visual inspection of IC maps and time series
10. **Remove artifacts**: Reject artifact ICs, reconstruct clean EEG
11. **Validate**: Compare cleaned data with original

### Expected Outcomes

- Understanding of signal separation principles
- Ability to identify EEG artifacts
- Practical experience with professional EEG analysis software
- Knowledge of ICA applications in neuroscience and beyond

---

## Exercise C4.5: Decision Tree Behavior Analysis

### Overview

This exercise explores the behavior of **C4.5 decision trees** (implemented in scikit-learn as `DecisionTreeClassifier` with `criterion='entropy'`) and how the **number of training samples** affects classification accuracy and decision boundary formation.

### Theoretical Background

#### Decision Tree Characteristics

**Partition Strategy**:
- Decision trees divide the feature space into **hyper-rectangular regions**
- Each region corresponds to a leaf node with a class prediction
- Splits are axis-aligned (parallel to feature axes)

**Training Sample Impact**:
- **Small N_train**: 
  - Fewer splits → larger rectangular regions
  - Coarse approximation of decision boundary
  - Visible rectangular structure in predictions
  - Risk of underfitting
  
- **Large N_train**: 
  - More splits → smaller, numerous rectangular regions
  - Finer approximation of decision boundary
  - Smoother-looking decision regions (though still rectangular)
  - Better captures complex patterns
  - Risk of overfitting (if not regularized)

#### C4.5 Algorithm (Entropy-Based)

**Splitting criterion**: Information Gain using entropy

**Entropy** (measure of impurity):
```
H(S) = -Σ p_i log₂(p_i)
```
Where p_i is the proportion of class i in set S

**Information Gain**:
```
IG(S, A) = H(S) - Σ (|S_v|/|S|) × H(S_v)
```
Where:
- S: Current node samples
- A: Feature to split on
- S_v: Subset of samples after split on feature A

**Algorithm selects**: Feature and threshold that maximize information gain at each node

### Experimental Setup

#### Synthetic Dataset

**Non-linear decision boundary**:
```python
y = sign(-2 × sign(x₁) × |x₁|^(2/3) + 4 × x₂²)
```

**Characteristics**:
- Highly non-linear boundary
- Cannot be separated by simple linear classifier
- Tests decision tree's ability to approximate complex shapes

**Feature space**: x₁, x₂ ∈ [-1, 1]

**Classes**: 
- Class -1 (red points)
- Class +1 (blue points)

#### Experiments with Variable Training Size

**Three training set sizes**:
1. N_train = 100 (small sample)
2. N_train = 1,000 (medium sample)
3. N_train = 10,000 (large sample)

**Test set**: N_test = 20,000 (fixed, for consistent evaluation)

### Implementation

```python
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def main():
    # Experiment with different training sizes
    N_train = [100, 1000, 10000]
    
    for n in N_train:
        # Generate training data
        x_train = np.array([
            (random.uniform(-1, 1), random.uniform(-1, 1)) 
            for _ in range(n)
        ])
        
        # True labels (non-linear boundary)
        y_train = np.sign(
            -2 * np.sign(x_train[:, 0]) * 
            np.abs(x_train[:, 0])**(2/3) + 
            4 * x_train[:, 1]**2
        )
        
        # Visualize training data
        colors = ['red' if z == -1 else 'blue' for z in y_train]
        plt.figure(figsize=(10, 10), num='Training Data')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, alpha=0.6)
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(f'Training Data (N = {n})')
        plt.grid(True)
        plt.show()
        
        # Train C4.5 decision tree (entropy criterion)
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(x_train, y_train)
        
        # Optional: Visualize tree structure
        # plt.figure(figsize=(20, 10))
        # plot_tree(clf, filled=True, 
        #          feature_names=['x₁', 'x₂'], 
        #          class_names=['-1', '1'])
        # plt.show()
        
        # Generate test data
        N_test = 20000
        x_test = np.array([
            (random.uniform(-1, 1), random.uniform(-1, 1)) 
            for _ in range(N_test)
        ])
        
        # Predict on test data
        y_pred = clf.predict(x_test)
        
        # True labels for test data
        y_true = np.sign(
            -2 * np.sign(x_test[:, 0]) * 
            np.abs(x_test[:, 0])**(2/3) + 
            4 * x_test[:, 1]**2
        )
        
        # Evaluate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy with N_train = {n}: {accuracy:.4f}")
        
        # Visualize predictions (shows rectangular decision regions)
        colors = ['red' if z == -1 else 'blue' for z in y_pred]
        plt.figure(figsize=(10, 10), num='Test Predictions')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=colors, 
                   alpha=0.3, s=1)
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(f'Decision Tree Predictions (N_train = {n})')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
```

### Expected Observations

#### 1. Training Data Visualization

**All three cases** show the true non-linear decision boundary through the color distribution of training points.

**Observations**:
- Red and blue regions form a complex, curved boundary
- The mathematical function creates a specific pattern in the 2D space
- Boundary is smooth and non-linear (not axis-aligned)

#### 2. Prediction Visualization (Key Insight)

**N_train = 100** (Small Sample):
- **Appearance**: Large, clearly visible rectangular regions
- **Decision boundary**: Very coarse approximation
- **Rectangles**: Easy to see individual rectangular partitions
- **Accuracy**: Lowest (~60-70% typical)
- **Interpretation**: Insufficient data to capture boundary complexity

**N_train = 1,000** (Medium Sample):
- **Appearance**: Smaller rectangles, still somewhat visible
- **Decision boundary**: Better approximation
- **Rectangles**: Noticeable but less obvious than N=100
- **Accuracy**: Moderate (~75-85% typical)
- **Interpretation**: Captures main patterns, misses fine details

**N_train = 10,000** (Large Sample):
- **Appearance**: Very small rectangles, appears almost smooth
- **Decision boundary**: Fine approximation of true boundary
- **Rectangles**: Hard to see individual partitions (mosaic effect)
- **Accuracy**: Highest (~85-95% typical)
- **Interpretation**: Closely follows true non-linear boundary

#### 3. Accuracy Trend

**Expected pattern**:
```
Accuracy(N=100) < Accuracy(N=1000) < Accuracy(N=10000)
```

**Why?**
- More training samples → more opportunities to learn boundary details
- Decision tree can create more refined partitions
- Better generalization to unseen test data

**Limitation**: 
- Even with large N, rectangular partitions cannot perfectly match smooth curves
- Asymptotic accuracy limit exists (< 100%)

### Analysis Questions

1. **Why rectangular regions?**
   - Decision trees use axis-aligned splits (x₁ < threshold or x₂ < threshold)
   - Each split creates rectangular sub-regions
   - No diagonal or curved boundaries possible

2. **Why does accuracy improve with more data?**
   - More samples provide better coverage of feature space
   - Tree can identify more informative splits
   - Better statistical estimates of class probabilities in each region

3. **What are the limitations?**
   - **Axis-aligned constraint**: Cannot efficiently represent diagonal or circular boundaries
   - **Overfitting risk**: Very deep trees with large N can memorize noise
   - **Discontinuous boundaries**: Predictions change abruptly at split points

4. **How to improve?**
   - **Ensemble methods**: Random Forest, Gradient Boosting (multiple trees voting)
   - **Regularization**: Limit tree depth, minimum samples per leaf
   - **Feature engineering**: Add polynomial features, interactions
   - **Alternative models**: Neural networks, SVM with RBF kernel (for smooth boundaries)

### Extensions and Experiments

#### Extension 1: Visualize Tree Structure

Uncomment the tree visualization code to see:
- Number of nodes (increases with N_train)
- Depth of tree (increases with N_train)
- Split decisions at each node
- Leaf node class distributions

```python
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, 
         feature_names=['x₁', 'x₂'], 
         class_names=['-1', '1'])
plt.title(f'Decision Tree Structure (N_train = {n})')
plt.show()
```

#### Extension 2: Add Tree Regularization

Compare unrestricted tree with regularized versions:

```python
# Unrestricted (current implementation)
clf_unrestricted = DecisionTreeClassifier(criterion='entropy')

# Regularized trees
clf_depth5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf_samples = DecisionTreeClassifier(criterion='entropy', 
                                     min_samples_split=100,
                                     min_samples_leaf=50)

# Train all and compare accuracies
```

**Expected**: Regularization prevents overfitting with large N_train

#### Extension 3: Compare with Other Algorithms

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# SVM with RBF kernel (smooth boundaries)
svm = SVC(kernel='rbf', gamma='scale')

# Neural network (flexible boundaries)
nn = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)

# Compare accuracies and visualize decision boundaries
```

**Question**: Which algorithm best approximates the true non-linear boundary?

#### Extension 4: Feature Space Complexity

Try different mathematical boundaries:

```python
# Linear (easy for decision tree)
y = sign(x₁ + x₂)

# Circular (hard for decision tree, easy for SVM-RBF)
y = sign(x₁² + x₂² - 0.5)

# XOR (requires multiple splits)
y = sign(x₁ * x₂)
```

**Analysis**: How does decision tree accuracy vary with boundary complexity?

### Learning Objectives

1. **Understand decision tree partitioning**: Rectangular regions, axis-aligned splits
2. **Sample size impact**: More data → finer partitions → better accuracy
3. **Visualization interpretation**: Connect rectangular structure to tree splits
4. **Model limitations**: Recognize when axis-aligned splits are insufficient
5. **Algorithm comparison**: Appreciate trade-offs between different classifiers

### Implementation Tips

**Efficient visualization**:
```python
# For large test sets, reduce point size and increase transparency
plt.scatter(x_test[:, 0], x_test[:, 1], c=colors, 
           alpha=0.1, s=0.5)  # Small, transparent points
```

**Save figures**:
```python
plt.savefig(f'decision_boundary_N{n}.png', dpi=300, bbox_inches='tight')
```

**Print tree statistics**:
```python
print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")
print(f"Number of nodes: {clf.tree_.node_count}")
```

### Conclusion

This exercise demonstrates a fundamental property of decision trees: their **piecewise constant** nature through **rectangular partitioning**. While increasing training samples improves approximation quality, the axis-aligned constraint remains. Understanding these limitations guides informed model selection for real-world healthcare applications.

---

## Summary

This laboratory course covers a comprehensive range of ICT applications in healthcare:

1. **Regression** (Labs 1-2): Predictive modeling for disease severity
2. **Image Processing** (Lab 3): Automated medical image segmentation
3. **Classification** (Lab 4): Diagnostic decision trees
4. **Statistical Analysis** (Lab 5): Medical test evaluation and comparison
5. **Signal Processing** (Lab 6): Brain signal analysis and artifact removal
6. **Decision Tree Analysis** (Exercise C4.5): Understanding classifier behavior and limitations

**Skills developed**:
- Python programming (Pandas, NumPy, Scikit-learn, Matplotlib)
- MATLAB (EEGLAB for EEG analysis)
- Machine learning algorithms
- Data cleaning and preprocessing
- Statistical evaluation metrics
- Medical domain knowledge
- Critical thinking about healthcare applications

**Tools and libraries**:
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, SciPy
- **MATLAB**: EEGLAB toolbox
- **Datasets**: Real medical data from UCI repository and clinical studies
