import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# ==============================
# DATA GENERATORS
# ==============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_data = valid_gen.flow_from_directory(
    "valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print("Classes:", train_data.class_indices)

# ==============================
# MODEL (TRANSFER LEARNING)
# ==============================
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # IMPORTANT: freeze first

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6),
    ModelCheckpoint("mobilenet_best.h5", save_best_only=True)
]

# ==============================
# PHASE 1 — TRAIN CLASSIFIER
# ==============================
print("\n🔹 Phase 1: Training classifier head\n")

history_1 = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# ==============================
# PHASE 2 — FINE-TUNING (STRONGLY RECOMMENDED)
# ==============================
print("\n🔹 Phase 2: Fine-tuning MobileNet layers\n")

# Unfreeze top layers only (safe fine-tuning)
FINE_TUNE_AT = int(len(base_model.layers) * 0.7)

for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # VERY IMPORTANT
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history_1.epoch[-1] + 1,
    callbacks=callbacks
)

# ==============================
# TEST EVALUATION
# ==============================
test_loss, test_acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ==============================
# SAVE FINAL MODEL
# ==============================
model.save("mobilenet_final.h5")
print("✅ Model saved as mobilenet_final.h5")
