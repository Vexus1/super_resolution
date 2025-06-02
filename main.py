import os
from pathlib import Path

import tensorflow as tf
import keras

from src.models.srcnn import SRCNN
from src.data.dataset import TrainsetConfig, TrainDataset, PairedConfig, PairedDataset 
from src.metrics import PSNR


def main() -> None:
    train_dataset = TrainDataset(
        TrainsetConfig(dir=Path("dataset/train"))
    ).build()
    val_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/validation"), use_lr=False)
    ).build()
    test_dataset = PairedDataset(
        PairedConfig(dir=Path("dataset/test"), use_lr=False)
    ).build()

    model = SRCNN.variant_915(filters=(64, 32, 3), input_channels=3).model
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=[PSNR(max_val=1.0)])

    model.fit(train_dataset,
              steps_per_epoch=1000,
              epochs=50,
              validation_data=val_dataset)
    print("Final evaluation on Set5 (test):")
    model.evaluate(test_dataset)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(main())
