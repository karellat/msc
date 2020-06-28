import tensorflow_hub as hub
import tensorflow as tf


def transfer_model(feature_extractor_url,
                   input_shape=(193, 193, 3),
                   feature_trainable=False,
                   fc_count=1024):
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=input_shape,
                                             trainable=feature_trainable)
    return tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(fc_count, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


def factory(model_name, **model_args):
    if model_name.lower() == "transfer":
        transfer_model(**model_args)
    else:
        raise Exception(f"Unknown 3d model : {model_name}")
