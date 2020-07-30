import tensorflow_hub as hub
import tensorflow as tf


def transfer_model(feature_extractor_url,
                   input_shape=(193, 193, 3),
                   feature_trainable=False,
                   fc_count=1024,
                   num_outputs=3,
                   dense_activation='relu',
                   output_activation='softmax',
                   dropout=0.5):
    """
    Create 2D using model from tensorflow hub as feature extractor
    Parameters
    ----------
    feature_extractor_url : str
        Url of the tensorflow hub feature extractor
    input_shape : tuple
        Expected input shape
    feature_trainable: bool
        False if fixed weights of feature extractor
    fc_count : int
        Number of full connected units
    num_outputs : int
        Number of network outputs
    dense_activation : str
        Name of activation function of full connected
    output_activation : str
        Output activation function
    dropout : float
        Probability of dropout of unit

    Returns
    -------
    tf.keras.Model
    """
    assert num_outputs > 1
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=input_shape,
                                             trainable=feature_trainable)
    return tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(fc_count, activation=dense_activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_outputs, activation=output_activation)
    ])


def factory(model_name, **model_args):
    if model_name.lower() == "transfer":
        return transfer_model(**model_args)
    else:
        raise Exception(f"Unknown 3d model : {model_name}")
