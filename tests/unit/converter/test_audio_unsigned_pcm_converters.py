# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from pyrit.converter.audio_echo_converter import AudioEchoConverter
from pyrit.converter.audio_volume_converter import AudioVolumeConverter


def test_volume_unsigned_8bit_silence_stays_centered():
    converter = AudioVolumeConverter(volume_factor=0.5)
    data = np.full(8, 128, dtype=np.uint8)

    result = converter._apply_volume(data)

    np.testing.assert_array_equal(result, data.astype(np.float64))


def test_volume_unsigned_8bit_scales_around_midpoint():
    converter = AudioVolumeConverter(volume_factor=0.5)
    data = np.array([128, 138, 118], dtype=np.uint8)

    result = converter._apply_volume(data)

    np.testing.assert_allclose(result, [128.0, 133.0, 123.0])


def test_echo_unsigned_8bit_silence_stays_centered():
    converter = AudioEchoConverter(delay=0.0005, decay=0.5)
    data = np.full(8, 128, dtype=np.uint8)

    result = converter._apply_echo(data, sample_rate=8000)

    np.testing.assert_array_equal(result, data.astype(np.float64))


def test_echo_unsigned_8bit_scales_around_midpoint():
    converter = AudioEchoConverter(delay=0.0005, decay=0.5)
    data = np.array([128, 138, 118, 128, 128, 128, 128, 128], dtype=np.uint8)

    result = converter._apply_echo(data, sample_rate=8000)

    np.testing.assert_allclose(result, [128.0, 138.0, 118.0, 128.0, 133.0, 123.0, 128.0, 128.0])
