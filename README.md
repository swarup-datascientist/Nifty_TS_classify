# Nifty_TS_classify
Stock data TS classification using Deep Learning Models

# Stock Movement Prediction with Hybrid TCN-Attention Model

This project implements a deep learning model combining Temporal Convolutional Networks (TCN) and Attention mechanisms to predict stock movements from time-series data.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Algorithms](#algorithms)
3. [Implementation Details](#implementation-details)
4. [Data Processing](#data-processing)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Threshold Optimization](#threshold-optimization)
9. [Usage](#usage)

## Project Structure
project/
├── best_model.h5 # Best model weights during training
├── final_model.h5 # Final model with threshold optimization
├── nifty_model.py # Main training script
└── README.md # This documentation



## Algorithms

### 1. Hybrid TCN-Attention Model
- **Temporal Convolutional Network (TCN)**:
  - Uses dilated causal convolutions to capture long-range dependencies
  - Residual blocks with skip connections for stable training
- **Multi-head Attention**:
  - Captures global dependencies between time steps
  - Helps identify important temporal patterns

### 2. Weighted Focal Loss
- Addresses class imbalance by:
  - Applying class-specific weights
  - Focusing on hard-to-classify examples (γ=2.0)

### 3. Threshold Optimization
- Per-class thresholds optimized using precision-recall curves
- Maximizes F1-score for each class independently

## Implementation Details

### Key Components
- **Data Augmentation**: Random oversampling of minority classes
- **Sequence Creation**: Sliding window (75 timesteps) over raw data
- **Normalization**: MinMax scaling (0-1 range)
- **Early Stopping**: Patience of 20 epochs on validation accuracy
- **Model Checkpointing**: Saves best weights based on validation accuracy

## Data Processing


# 1. Load and preprocess data
df = pd.read_csv("aggregated_master.csv")

# 2. Encode categorical features
for col in categorical_cols:
    df_features[col] = LabelEncoder().fit_transform(df_features[col])

# 3. Normalize features
scaler = MinMaxScaler()
df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

# 4. Create sequences (window_size=75)
for i in range(len(df) - window_size):
    window = df_features.iloc[i:i+window_size].values
    label = df[target].iloc[i + window_size]
    X_sequences.append(window)
    y_labels.append(label)

# Model Architecture:

def residual_block(x, filters, kernel_size, dilation_rate):
    # 1D convolution with skip connection
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    
    # Projection shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    return Activation('relu')(Add()([shortcut, x]))

# Complete model:
# Input -> Conv1D -> 3 Residual Blocks (dilation=1,2,4) -> 
# MultiHeadAttention -> GlobalAveragePooling -> Dense Layers

# Training:
# Class-weighted Focal Loss
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (num_classes * class_counts)

# Model compilation
base_model.compile(
    optimizer=Adam(learning_rate=0.0005, clipvalue=0.5),
    loss=WeightedFocalLoss(gamma=2.0, class_weights=class_weights_dict),
    metrics=['accuracy']
)

# Training with early stopping
history = base_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=20),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluation
Metrics Reported:
Precision, Recall, F1-score per class

Confusion matrices

Training vs validation accuracy

Threshold Optimization:
for class_idx in range(num_classes):
    precision, recall, thresh = precision_recall_curve(
        (y_val == class_idx).astype(int), 
        probs_val[:, class_idx]
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    thresholds[class_idx] = thresh[np.argmax(f1_scores)]


# Results:
Model achieves improved performance on minority classes through:

Class-weighted focal loss

Random oversampling

Per-class threshold optimization

Hybrid architecture captures both local (TCN) and global (Attention) temporal patterns

