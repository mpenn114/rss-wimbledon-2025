import argparse
from src.alternative_models.base_alt_model import BaseAlternativeModel
from typing import Type

class ModelRegistry:
    """Singleton registry to store and retrieve model classes."""
    _models = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._models[name.lower()] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type[BaseAlternativeModel]:
        return cls._models.get(name.lower())

    @classmethod
    def list_models(cls):
        return list(cls._models.keys())
    
# Shortcut for the decorator
register_model = ModelRegistry.register

def run_model(model_name: str, tournament: str, year: int, male: bool):
    """Instantiates and executes the requested model."""
    model_class = ModelRegistry.get_model(model_name)
    
    if not model_class:
        available = ", ".join(ModelRegistry.list_models())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    model_instance = model_class()
    model_instance.predict(tournament, year, male)
