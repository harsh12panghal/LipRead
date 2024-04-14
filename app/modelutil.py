import os
import tensorflow
import keras
# from keras.models import Sequential
# from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.initializers import Orthogonal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# initializer = Orthogonal()
# path = tensorflow.keras.models.load_model(os.path.join('..', 'models96', 'checkpoint'))
#
# path.save('saved_model.h5')


def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model_checkpoint_path = (os.path.join('..', 'model96', 'checkpoint'))
    checkpoint = tensorflow.train.Checkpoint(model=model)

    checkpoint_manager = tensorflow.train.CheckpointManager(checkpoint, directory=model_checkpoint_path, max_to_keep=1)
    # # Restore the latest checkpoint
    # if checkpoint_manager.latest_checkpoint:
    #     checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    #     print("Model loaded from checkpoint.")

    # model = keras.layers.TFSMLayer('../models96/checkpoint', call_endpoint='serving_default')
    # model.load_weights(os.path.join('..', 'models96', 'checkpoint'))

    return model

