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
    model_hp = ModelHP()
    train_hp = TrainHP()
    data_hp = DataHP()
    blur_hp = BlurHP()

    train_ds = TrainDataset(
    TrainsetConfig(
        dir=data_hp.root / "train",
        batch_size=train_hp.batch_size,
        scale=data_hp.scale,
        fsub=data_hp.fsub,
        shuffle_buffer=data_hp.shuffle_buffer,
        blur=blur_hp,                    
    )
    ).build()
    val_ds = PairedDataset(
        PairedConfig(dir=data_hp.root / "validation", blur=blur_hp, use_lr=False)
    ).build()
    test_ds = PairedDataset(
        PairedConfig(dir=data_hp.root / "test", blur=blur_hp, use_lr=False)
    ).build()

    model = SRCNN.variant_915(
        filters=model_hp.filters,
        input_channels=model_hp.input_channels,
    ).model

    model.compile(
        optimizer=keras.optimizers.Adam(train_hp.lr_init),
        loss=keras.losses.Huber(delta=0.01),
        metrics=[PSNR(max_val=1.0, shave=0)],
    )

    lr_sched = keras.callbacks.ReduceLROnPlateau(
        monitor="val_psnr",
        mode="max",
        factor=train_hp.lr_factor,
        patience=train_hp.lr_patience,
        min_lr=train_hp.lr_min,
        verbose=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        "best.keras", save_best_only=True,
        monitor="val_psnr", mode="max"
    )

    model.fit(
    train_ds,
    steps_per_epoch=train_hp.steps_per_epoch,
    epochs=train_hp.epochs,
    validation_data=val_ds,
    callbacks=[tensorboard_cb, lr_sched, checkpoint],
    )

    print("Final evaluation on Set5 (test):")
    model.evaluate(test_ds)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
