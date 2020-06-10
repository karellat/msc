from deep_mri.train.training_encoder import train
from datetime import datetime

batch_size = [2, 4, 8, 32, 64, 128]
epochs = 100

t = datetime.strftime("%Y-%m-%d-%H-%M")
for b in batch_size:
    name = f"{t}-b{b}"
    train(batch_size=b, epochs=epochs, model_name=name)
