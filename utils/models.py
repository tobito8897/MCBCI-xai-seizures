import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, \
                                    Dropout, BatchNormalization, \
                                    Input, Conv1D, MaxPooling1D, Concatenate, \
                                    GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping


# Necesario para usar Theano como backend
os.environ["MKL_THREADING_LAYER"] = "GNU"


def early_stop_wang(patience):
    return EarlyStopping(monitor="val_loss",
                         patience=patience,
                         verbose=2)


def Net_Wang_1d(input_dim: int):
    """
    Creation of model object, size (width, length)
    """
    input_shape = Input(shape=input_dim)

    tower_1 = Conv1D(32, 3, strides=2,
                     activation='relu')(input_shape)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_1)
    tower_1 = Conv1D(64, 3, padding='same', strides=2,
                     activation='relu')(tower_1)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_1)
    tower_1 = Conv1D(128, 3, padding='same', strides=1,
                     activation='relu')(tower_1)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_1)

    tower_2 = Conv1D(32, 5, strides=2,
                     activation='relu')(input_shape)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_2)
    tower_2 = Conv1D(64, 5, padding='same', strides=2,
                     activation='relu')(tower_2)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_2)
    tower_2 = Conv1D(128, 3, padding='same', strides=1,
                     activation='relu')(tower_2)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling1D(pool_size=3, strides=1,
                           padding='same')(tower_2)

    merged = Concatenate(axis=1)([tower_1, tower_2])
    merged = GlobalAveragePooling1D()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.25)(merged)
    out = Dense(2, activation='softmax')(merged)

    model = Model(input_shape, out)
    model.compile(optimizer="Adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    return model


def Net_Wang_2d(input_dim: int):
    """
    Creation of model object, size (width, length)
    """
    input_shape = Input(shape=input_dim)

    tower_1 = Conv2D(32, 3, strides=2,
                     activation='relu')(input_shape)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_1)
    tower_1 = Conv2D(64, 3, padding='same', strides=2,
                     activation='relu')(tower_1)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_1)
    tower_1 = Conv2D(128, 3, padding='same', strides=1,
                     activation='relu')(tower_1)
    tower_1 = BatchNormalization(axis=-1)(tower_1)
    tower_1 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_1)

    tower_2 = Conv2D(32, 5, strides=2,
                     activation='relu')(input_shape)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_2)
    tower_2 = Conv2D(64, 5, padding='same', strides=2,
                     activation='relu')(tower_2)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_2)
    tower_2 = Conv2D(128, 3, padding='same', strides=1,
                     activation='relu')(tower_2)
    tower_2 = BatchNormalization(axis=-1)(tower_2)
    tower_2 = MaxPooling2D(pool_size=3, strides=1,
                           padding='same')(tower_2)

    merged = Concatenate(axis=1)([tower_1, tower_2])
    merged = GlobalAveragePooling2D()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.25)(merged)
    out = Dense(2, activation='softmax')(merged)

    model = Model(input_shape, out)
    model.compile(optimizer="Adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    return model
