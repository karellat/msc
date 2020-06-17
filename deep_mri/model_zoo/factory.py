from deep_mri.model_zoo import models_3d, encoders, models_2d

MODEL_TYPE_DELIMETER = "_"


def model_factory(model_name, **model_args):
    assert len(model_name.split(MODEL_TYPE_DELIMETER)) == 2, "Model name can have only single delimeter"
    mod_type, mod_name = model_name.split(MODEL_TYPE_DELIMETER)
    if mod_type.lower() == "3d":
        return models_3d.factory(mod_name, **model_args)
    elif mod_type.lower() == "encoder":
        return encoders.factory(mod_name, **model_args)
    elif mod_type.lower() == "2d":
        return models_2d.factory(mod_name, **model_args)
    else:
        raise Exception(f"Unknown model name: {model_name}")