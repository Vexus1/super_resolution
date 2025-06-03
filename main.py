import os
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras

from src.models.srcnn import SRCNN
from src.data.dataset import (TrainsetConfig, TrainDataset,
                              PairedConfig, PairedDataset) 
from src.metrics import PSNR

log_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)

tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(log_dir),
    update_freq="epoch",         
    histogram_freq=1,             
    write_graph=True,             
    write_images=True           
)

def main() -> None:
    train_dataset = TrainDataset(
        TrainsetConfig(dir=Path("dataset/train"), batch_size=128)
    ).build()
    val_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/validation"), use_lr=False)
    ).build()
    test_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/test"), use_lr=False)
    ).build()

    model = SRCNN.variant_915(filters=(128, 64, 1), input_channels=1).model

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.Huber(delta=0.01),
                  metrics=[PSNR(max_val=1.0, shave=0)])

    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor="val_psnr", 
                                                    mode="max",
                                                    factor=0.5,
                                                    patience=5,           
                                                    min_lr=1e-6,
                                                    verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint("best.keras",
                                                 save_best_only=True,
                                                 monitor="val_psnr",
                                                 mode="max")
    model.fit(train_dataset,
              steps_per_epoch=1000,
              epochs=100,
              validation_data=val_dataset,
              callbacks=[tensorboard_cb, lr_schedule, checkpoint])
    print("Final evaluation on Set5 (test):")
    model.evaluate(test_dataset)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
