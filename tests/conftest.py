import pytest

from src.data import CustomPreprocessor


@pytest.fixture
def preprocessor():
    class_names = ["t1", "t2"]
    return CustomPreprocessor(class_names=class_names)
