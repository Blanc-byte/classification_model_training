import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

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
# ATTENTION MODULES (KERAS-SAFE)
# ==============================
def channel_attention(x, ratio=8):
    channel = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_dense_1 = layers.Dense(channel // ratio, activation="relu")
    shared_dense_2 = layers.Dense(channel, activation="sigmoid")

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    scale = layers.Add()([avg_out, max_out])
    scale = layers.Reshape((1, 1, channel))(scale)

    return layers.Multiply()([x, scale])


def spatial_attention(x):
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(x)

    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(x)

    concat = layers.Concatenate()([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    return layers.Multiply()([x, attention])

# ==============================
# MULTI-SCALE BLOCK
# ==============================
def multi_scale_block(x, filters):
    conv1 = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    conv3 = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    conv5 = layers.Conv2D(filters, 5, padding="same", activation="relu")(x)
    return layers.Concatenate()([conv1, conv3, conv5])

# ==============================
# DAMLSNet ARCHITECTURE
# ==============================
def DAMLSNet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = multi_scale_block(x, 32)
    x = channel_attention(x)
    x = spatial_attention(x)
    x = layers.MaxPooling2D()(x)

    x = multi_scale_block(x, 64)
    x = channel_attention(x)
    x = spatial_attention(x)
    x = layers.MaxPooling2D()(x)

    x = multi_scale_block(x, 128)
    x = channel_attention(x)
    x = spatial_attention(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="DAMLSNet")

# ==============================
# BUILD MODEL
# ==============================
model = DAMLSNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES
)

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
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6),
    ModelCheckpoint("damlsnet_best.h5", save_best_only=True)
]

# ==============================
# TRAIN
# ==============================
print("\n🔹 Training DAMLSNet\n")

model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# TEST
# ==============================
test_loss, test_acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ==============================
# SAVE FINAL MODEL
# ==============================
model.save("damlsnet_final.h5")
print("✅ Model saved as damlsnet_final.h5")
