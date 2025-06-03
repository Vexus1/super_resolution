import os
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras

from src.models.srcnn import SRCNN
from src.data.dataset import (TrainsetConfig, TrainDataset,
                              PairedConfig, PairedDataset) 
from src.metrics import PSNR
from hparams import ModelHP, TrainHP, DataHP, BlurHP

class Train:
    def __init__(self):
        self.model_hp = ModelHP()
        self.train_hp = TrainHP()
        self.data_hp = DataHP()
        self.blur_hp = BlurHP()
        self._datasets()
        self._model()
        self._callbacks()

    def _datasets(self) -> None:
        self.train_ds = TrainDataset(
            TrainsetConfig(
                dir=self.data_hp.root / "train",
                batch_size=self.train_hp.batch_size,
                scale=self.data_hp.scale,
                fsub=self.data_hp.fsub,
                shuffle_buffer=self.data_hp.shuffle_buffer,
                blur=self.blur_hp
            )
        ).build()
        self.val_ds = PairedDataset(
            PairedConfig(dir=self.data_hp.root / "validation",
                         blur=self.blur_hp, use_lr=False)
        ).build()
        self.test_ds = PairedDataset(
           PairedConfig(dir=self.data_hp.root / "test",
                        blur=self.blur_hp, use_lr=False)
        ).build()

    def _model(self) -> None:
        self.model = SRCNN.variant_935(
            filters=self.model_hp.filters,
            input_channels=self.model_hp.input_channels
        ).model
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.train_hp.lr_init),
            loss=keras.losses.Huber(delta=0.01),
            metrics=[PSNR(max_val=1.0, shave=0)]
        )

    def _callbacks(self) -> None:
        log_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                update_freq="epoch",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_psnr",
                mode="max",
                factor=self.train_hp.lr_factor,
                patience=self.train_hp.lr_patience,
                min_lr=self.train_hp.lr_min,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="best.keras",
                save_best_only=True,
                monitor="val_psnr",
                mode="max"
            )
        ]

    def fit(self) -> None:
        self.model.fit(
            self.train_ds,
            steps_per_epoch=self.train_hp.steps_per_epoch,
            epochs=self.train_hp.epochs,
            validation_data=self.val_ds,
            callbacks=self.callbacks
        )
    
    def evaluate(self) -> None:
        print("Final evaluation on Set5 (test):")
        self.model.evaluate(self.test_ds)

    def run(self) -> None:
        self.fit()
        self.evaluate()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    Train().run()
