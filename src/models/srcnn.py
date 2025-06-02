from dataclasses import dataclass
from typing import Self

import keras

@dataclass
class SRCNN:
    filters: tuple[int, int, int]
    kernel_sizes: tuple[int, int, int]
    input_channels: int

    def __post_init__(self):
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Input(shape=(None, None, self.input_channels)),
            keras.layers.Conv2D(filters=self.filters[0],
                                kernel_size=self.kernel_sizes[0],
                                activation="relu",
                                padding="same"),
            keras.layers.Conv2D(filters=self.filters[1],
                                kernel_size=self.kernel_sizes[1],
                                activation='relu',
                                padding="same"),
            keras.layers.Conv2D(filters=self.filters[2],
                                kernel_size=self.kernel_sizes[2],
                                activation="linear",
                                padding="same")
        ])
        return model
    
    @classmethod
    def variant_915(cls, **kwargs) -> Self:
        return cls(kernel_sizes=(9, 1, 5), **kwargs)
    
    @classmethod
    def variant_935(cls, **kwargs) -> Self:
        return cls(kernel_sizes=(9, 3, 5), **kwargs) 
        
    @classmethod
    def variant_955(cls, **kwargs) -> Self:
        return cls(kernel_sizes=(9, 5, 5), **kwargs)
    
