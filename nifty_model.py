import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import CustomObjectScope
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler


# === 1. Define Custom Objects ===
class WeightedFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, class_weights=None, name="weighted_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else {}

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        pt = tf.exp(-ce)
        weights = tf.gather(list(self.class_weights.values()), y_true)
        loss = weights * tf.pow(1 - pt, self.gamma) * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        return {"gamma": self.gamma, "class_weights": self.class_weights}


class ThresholdModel(tf.keras.Model):

    def __init__(self, base_model, thresholds, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.thresholds = thresholds

    def call(self, inputs):
        logits = self.base_model(inputs)
        return tf.nn.softmax(logits)

    def predict_with_thresholds(self, X):
        probs = self.predict(X)
        predictions = np.zeros_like(probs)
        for class_idx, threshold in self.thresholds.items():
            predictions[:, class_idx] = (probs[:, class_idx] >= threshold).astype(int)
        return np.argmax(predictions * probs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_model": tf.keras.models.clone_model(self.base_model),
            "thresholds": self.thresholds
        })
        return config

    @classmethod
    def from_config(cls, config):
        # First create the base model
        base_model = Model.from_config(config['base_model'].get_config())
        # Then create the ThresholdModel instance
        return cls(base_model, config['thresholds'])


# Register custom objects
custom_objects = {
    'WeightedFocalLoss': WeightedFocalLoss,
    'ThresholdModel': ThresholdModel
}

# === 2. Data Loading and Preprocessing ===
print("Loading data...")
df = pd.read_csv("aggregated_master.csv")

# Prepare features and target
df_features = df.drop(columns=['last_traded_time', 'date'])
target = 'result'

# Encode categorical features
categorical_cols = df_features.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    df_features[col] = LabelEncoder().fit_transform(df_features[col])

# Normalize features
scaler = MinMaxScaler()
feature_cols = [col for col in df_features.columns if col != target]
df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

# === 3. Create Sequences ===
window_size = 75
X_sequences = []
y_labels = []

for i in range(len(df) - window_size):
    window = df_features.iloc[i:i + window_size].values
    label = df[target].iloc[i + window_size]
    X_sequences.append(window)
    y_labels.append(label)

X_padded = np.array(X_sequences)
y_array = np.array(y_labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_array)
num_classes = len(label_encoder.classes_)
num_features = X_padded.shape[2]

print(f"Input shape: {X_padded.shape}")
print("Class distribution:", {cls: count for cls, count in zip(label_encoder.classes_, np.bincount(y_encoded))})

# === 4. Data Augmentation for Minority Classes ===
n_samples, n_timesteps, n_features = X_padded.shape
X_flat = X_padded.reshape(n_samples, -1)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y_encoded)
X_resampled = X_resampled.reshape(-1, n_timesteps, n_features)

# === 5. Train/Validation/Test Split ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# === 6. Build Hybrid TCN + Attention Model ===
def residual_block(x, filters, kernel_size, dilation_rate):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Add()([shortcut, x])
    return Activation('relu')(x)


inputs = Input(shape=(window_size, num_features))
x = Conv1D(64, 3, padding='same')(inputs)

for dilation_rate in [1, 2, 4]:
    x = residual_block(x, 64, 3, dilation_rate)

attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = LayerNormalization()(Add()([x, attention_output]))
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
logits = Dense(num_classes, activation='linear')(x)
base_model = Model(inputs, logits)

# === 7. Training ===
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (num_classes * class_counts)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

base_model.compile(
    optimizer=Adam(learning_rate=0.0005, clipvalue=0.5),
    loss=WeightedFocalLoss(gamma=2.0, class_weights=class_weights_dict),
    metrics=['accuracy']
)

history = base_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    ],
    verbose=1
)


# === 8. Evaluation ===
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Training data evaluation
y_train_pred = np.argmax(base_model.predict(X_train), axis=1)
print("\n=== Training Data ===")
print("Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=[str(cls) for cls in label_encoder.classes_]))
plot_confusion_matrix(y_train, y_train_pred, "Training Data Confusion Matrix")

# Test data evaluation
y_test_pred = np.argmax(base_model.predict(X_test), axis=1)
print("\n=== Test Data ===")
print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=[str(cls) for cls in label_encoder.classes_]))
plot_confusion_matrix(y_test, y_test_pred, "Test Data Confusion Matrix")

# === 9. Threshold Optimization ===
logits_val = base_model.predict(X_val)
probs_val = tf.nn.softmax(logits_val).numpy()

thresholds = {}
for class_idx in range(num_classes):
    precision, recall, thresh = precision_recall_curve(
        (y_val == class_idx).astype(int),
        probs_val[:, class_idx]
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    thresholds[class_idx] = thresh[np.argmax(f1_scores)]

print("\nOptimal thresholds per class:")
for class_idx, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name}: {thresholds[class_idx]:.3f}")

# === 10. Create and Save Final Model ===
final_model = ThresholdModel(base_model, thresholds)

with CustomObjectScope(custom_objects):
    final_model.save("final_model.h5")
    print("\nModel saved to 'final_model.h5'")
