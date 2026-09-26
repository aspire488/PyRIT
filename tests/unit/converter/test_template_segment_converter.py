# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest

from pyrit.converter import TemplateSegmentConverter
from pyrit.models import SeedPrompt


def test_template_segment_converter_init_default():
    """Test initialization with default template."""
    converter = TemplateSegmentConverter()
    assert converter.prompt_template is not None
    assert len(converter.prompt_template.parameters) >= 2
    assert converter._number_parameters >= 2


def test_template_segment_converter_init_custom():
    """Test initialization with custom template."""
    custom_template = SeedPrompt(
        value="First part: {{ part1 }}\nSecond part: {{ part2 }}", parameters=["part1", "part2"]
    )
    converter = TemplateSegmentConverter(prompt_template=custom_template)
    assert converter.prompt_template == custom_template
    assert converter._number_parameters == 2
    rendered = custom_template.render_template_value(part1="test1", part2="test2")
    assert "First part: test1" in rendered
    assert "Second part: test2" in rendered


def test_template_segment_converter_init_invalid_template():
    invalid_template = SeedPrompt(value="Only one part: {{ part1 }}", parameters=["part1"])
    with pytest.raises(ValueError, match="Template must have at least two parameters"):
        TemplateSegmentConverter(prompt_template=invalid_template)


def test_template_segment_converter_init_missing_parameter():
    invalid_template = SeedPrompt(
        value="First part: {{ part1 }}\nSecond part: {{ part2 }} third {{ part3 }}",
        parameters=["part1", "part2"],
    )
    with pytest.raises(ValueError, match="Error validating template parameters"):
        TemplateSegmentConverter(prompt_template=invalid_template)


def test_template_segment_converter_init_template_with_whitespace():
    template = SeedPrompt(
        value="""
        First: {{ part1 }}
        Second: {{part2}}
        Third: {{ part3}}
        Fourth: {{part4 }}
        """,
        parameters=["part1", "part2", "part3", "part4"],
    )
    TemplateSegmentConverter(prompt_template=template)
    rendered = template.render_template_value(part1="test1", part2="test2", part3="test3", part4="test4")
    assert "First: test1" in rendered
    assert "Second: test2" in rendered
    assert "Third: test3" in rendered
    assert "Fourth: test4" in rendered


def test_template_segment_converter_input_output_support():
    converter = TemplateSegmentConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
    assert converter.output_supported("text") is True
    assert converter.output_supported("image_path") is False


async def test_template_segment_converter_convert_basic():
    template = SeedPrompt(value="First: {{ part1 }}\nSecond: {{ part2 }}", parameters=["part1", "part2"])
    converter = TemplateSegmentConverter(prompt_template=template)
    result = await converter.convert_async(prompt="Hello world", input_type="text")
    assert result.output_type == "text"
    assert "First:" in result.output_text
    assert "Second:" in result.output_text
    assert result.output_text == "First: Hello\nSecond: world"


async def test_template_segment_converter_convert_long_prompt():
    template = SeedPrompt(
        value="""
        Part 1: {{ part1 }}
        Part 2: {{ part2 }}
        Part 3: {{ part3 }}
        """,
        parameters=["part1", "part2", "part3"],
    )
    converter = TemplateSegmentConverter(prompt_template=template)
    long_prompt = "This is a longer prompt that should be split into three different segments for testing purposes"
    result = await converter.convert_async(prompt=long_prompt, input_type="text")
    assert result.output_type == "text"
    assert "Part 1:" in result.output_text
    assert "Part 2:" in result.output_text
    assert "Part 3:" in result.output_text
    assert len(result.output_text.split("\n")) == 5


async def test_template_segment_converter_convert_short_prompt():
    """Prompts with fewer words than template parameters should not raise ValueError."""
    template = SeedPrompt(
        value="""
        First: {{ part1 }}
        Second: {{ part2 }}
        Third: {{ part3 }}
        """,
        parameters=["part1", "part2", "part3"],
    )
    converter = TemplateSegmentConverter(prompt_template=template)
    result = await converter.convert_async(prompt="Hi there", input_type="text")
    assert result.output_type == "text"
    assert result.output_text == "First: Hi\nSecond: there\nThird: "


async def test_template_segment_converter_invalid_input_type():
    converter = TemplateSegmentConverter()
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="test", input_type="image_path")
