import os
from pathlib import Path

import tensorflow as tf
import keras

from src.models.srcnn import SRCNN
from src.data.dataset import TrainsetConfig, TrainDataset, PairedConfig, PairedDataset 
from src.metrics import PSNR

def main() -> None:
    train_dataset = TrainDataset(
        TrainsetConfig(dir=Path("dataset/train"), batch_size=64)
    ).build()
    val_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/validation"), use_lr=False)
    ).build()
    test_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/test"), use_lr=False)
    ).build()

    model = SRCNN.variant_915(filters=(64, 32, 1), input_channels=1).model
    lr_schedule = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                    patience=10,
                                                    min_lr=1e-6)
    checkpoint = keras.callbacks.ModelCheckpoint("best.keras",
                                                 save_best_only=True,
                                                 monitor="val_psnr",
                                                 mode="max")
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="mse",
                  metrics=[PSNR(max_val=1.0, shave=0)])

    model.fit(train_dataset,
              steps_per_epoch=1000,
              epochs=50,
              validation_data=val_dataset,
              callbacks=[lr_schedule, checkpoint])
    print("Final evaluation on Set5 (test):")
    model.evaluate(test_dataset)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
