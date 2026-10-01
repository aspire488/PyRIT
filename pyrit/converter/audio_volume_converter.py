# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import logging
import math
from typing import Any, Literal

import numpy as np
from scipy.io import wavfile

from pyrit.converter.converter import Converter, ConverterResult
from pyrit.memory import data_serializer_factory
from pyrit.models import PromptDataType

logger = logging.getLogger(__name__)


class AudioVolumeConverter(Converter):
    """
    Changes the volume of an audio file by scaling the amplitude.

    A volume_factor > 1.0 increases the volume (louder),
    while a volume_factor < 1.0 decreases it (quieter).
    A volume_factor of 1.0 leaves the audio unchanged.
    The converter scales all audio samples by the given factor and clips
    the result to the valid range for the original data type.
    Sample rate, bit depth, and number of channels are preserved.
    """

    SUPPORTED_INPUT_TYPES = ("audio_path",)
    SUPPORTED_OUTPUT_TYPES = ("audio_path",)
    AcceptedAudioFormats = Literal["wav"]

    def __init__(self, *, output_format: AcceptedAudioFormats = "wav", volume_factor: float = 1.5) -> None:
        if not math.isfinite(volume_factor) or volume_factor <= 0:
            raise ValueError("volume_factor must be finite and greater than 0.")
        self._output_format = output_format
        self._volume_factor = volume_factor

    def _apply_volume(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Scale audio samples by the volume factor and clip to the valid range.

        Unsigned 8-bit PCM is centered at 128, so scale around that midpoint
        rather than around zero.
        """
        if data.dtype == np.uint8:
            midpoint = 128.0
            scaled = (data.astype(np.float64) - midpoint) * self._volume_factor + midpoint
        else:
            scaled = data.astype(np.float64) * self._volume_factor

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scaled = np.clip(scaled, info.min, info.max)

        return scaled

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "audio_path") -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        try:
            audio_serializer = data_serializer_factory(
                category="prompt-memory-entries", data_type="audio_path", extension=self._output_format, value=prompt
            )
            audio_bytes = await audio_serializer.read_data_async()
            bytes_io = io.BytesIO(audio_bytes)
            sample_rate, data = wavfile.read(bytes_io)
            original_dtype = data.dtype

            if data.ndim == 1:
                volume_data = self._apply_volume(data).astype(original_dtype)
            else:
                channels = [self._apply_volume(data[:, ch]) for ch in range(data.shape[1])]
                volume_data = np.column_stack(channels).astype(original_dtype)

            output_bytes_io = io.BytesIO()
            wavfile.write(output_bytes_io, sample_rate, volume_data)
            converted_bytes = output_bytes_io.getvalue()
            await audio_serializer.save_data_async(data=converted_bytes)
            audio_serializer_file = str(audio_serializer.value)
            logger.info(
                "Volume changed by factor %.2f for [%s], and the audio was saved to [%s]",
                self._volume_factor, prompt, audio_serializer_file,
            )
        except Exception as e:
            logger.error("Failed to convert audio volume: %s", str(e))
            raise
        return ConverterResult(output_text=audio_serializer_file, output_type=input_type)
