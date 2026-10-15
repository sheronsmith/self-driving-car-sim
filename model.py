import keras
from keras import layers


# Nvidia's end-to-end architecture from the 2016 paper (Bojarski et al.)
# 5 conv layers -> flatten -> 3 dense layers -> 1 steering output
# ~250k weights total

def nvidia_model(input_shape=(66, 200, 3), drop_rate=0.5):
    model = keras.Sequential()

    # normalise pixels to [-0.5, 0.5]
    model.add(layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))

    # conv layers — first three use 5x5 filters with stride 2, last two use 3x3
    model.add(layers.Conv2D(24, 5, strides=2, activation='elu'))
    model.add(layers.Conv2D(36, 5, strides=2, activation='elu'))
    model.add(layers.Conv2D(48, 5, strides=2, activation='elu'))
    model.add(layers.Conv2D(64, 3, activation='elu'))
    model.add(layers.Conv2D(64, 3, activation='elu'))
    model.add(layers.Dropout(drop_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='elu'))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(50, activation='elu'))
    model.add(layers.Dense(10, activation='elu'))

    # single output — the predicted steering angle
    model.add(layers.Dense(1))

    return model


if __name__ == '__main__':
    m = nvidia_model()
    m.summary()
