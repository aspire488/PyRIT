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


class AudioEchoConverter(Converter):
    """
    Adds an echo effect to an audio file.

    The echo is created by mixing a delayed, attenuated copy of the signal back
    into the original. The delay and decay parameters control the timing and
    loudness of the echo respectively. Sample rate, bit depth, and channel
    count are preserved.
    """
    SUPPORTED_INPUT_TYPES = ("audio_path",)
    SUPPORTED_OUTPUT_TYPES = ("audio_path",)
    AcceptedAudioFormats = Literal["wav"]

    def __init__(self, *, output_format: AcceptedAudioFormats = "wav", delay: float = 0.3, decay: float = 0.5) -> None:
        if not math.isfinite(delay) or delay <= 0:
            raise ValueError("delay must be a finite number greater than 0.")
        if not math.isfinite(decay) or decay <= 0 or decay >= 1:
            raise ValueError("decay must be a finite number between 0 and 1 (exclusive).")
        self._output_format = output_format
        self._delay = delay
        self._decay = decay

    def _apply_echo(self, data: np.ndarray[Any, Any], sample_rate: int) -> np.ndarray[Any, Any]:
        """Apply echo to a 1-D audio signal, preserving unsigned PCM's midpoint."""
        delay_samples = int(self._delay * sample_rate)
        if data.dtype == np.uint8:
            midpoint = 128.0
            centered = data.astype(np.float64) - midpoint
            output = centered.copy()
            if delay_samples < len(data):
                output[delay_samples:] += self._decay * centered[: len(data) - delay_samples]
            output += midpoint
        else:
            output = data.astype(np.float64).copy()
            if delay_samples < len(data):
                output[delay_samples:] += self._decay * data[: len(data) - delay_samples].astype(np.float64)

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            output = np.clip(output, info.min, info.max)
        return output

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
                echo_data = self._apply_echo(data, sample_rate).astype(original_dtype)
            else:
                channels = [self._apply_echo(data[:, ch], sample_rate) for ch in range(data.shape[1])]
                echo_data = np.column_stack(channels).astype(original_dtype)

            output_bytes_io = io.BytesIO()
            wavfile.write(output_bytes_io, sample_rate, echo_data)
            converted_bytes = output_bytes_io.getvalue()
            await audio_serializer.save_data_async(data=converted_bytes)
            audio_serializer_file = str(audio_serializer.value)
            logger.info(
                "Echo effect (delay=%.3fs, decay=%.2f) applied to [%s], saved to [%s]",
                self._delay, self._decay, prompt, audio_serializer_file,
            )
        except Exception as e:
            logger.error("Failed to apply echo effect: %s", str(e))
            raise
        return ConverterResult(output_text=audio_serializer_file, output_type=input_type)
