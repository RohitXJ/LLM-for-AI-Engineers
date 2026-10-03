from abc import ABC, abstractmethod
from typing import Dict, Any
from core.models.alert import SystemAlert

class BaseIngestionAdapter(ABC):
    """
    Abstract Base Class for all Ingestion Plugins.
    Every plugin must implement the 'normalize' method to convert 
    source-specific JSON into a standard SystemAlert.
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """The name of the source (e.g., 'prometheus', 'datadog')"""
        pass

    @abstractmethod
    def normalize(self, payload: Dict[str, Any]) -> SystemAlert:
        """
        Translates raw incoming JSON payload to a SystemAlert model.
        """
        pass
