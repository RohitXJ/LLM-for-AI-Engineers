from typing import Optional, Literal, Any, List, Union
from pydantic import BaseModel, Field, field_validator
import re

class DocumentMetadata(BaseModel):
    """
    Schema for automated metadata extraction from research documents.
    
    Attributes:
        topic (str): The primary subject matter of the text.
        year (int): The primary year or era discussed in the text.
        complexity (Literal): The technical depth (Beginner, Intermediate, Advanced).
        audience (str): The intended reader.
        priority (Literal): The urgency or importance (Low, Medium, High).
    """
    topic: str = Field(
        description="The primary subject matter of the text (e.g., Cybersecurity, Financial Audit, HR Policy)."
    )
    year: int = Field(
        default=2026,
        description="The primary year or era discussed in the text. Defaults to 2026."
    )
    complexity: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        description="The technical depth of the content."
    )
    audience: str = Field(
        description="The intended reader (e.g., 'Stakeholders', 'Developers', 'Data Scientists')."
    )
    priority: Literal["Low", "Medium", "High"] = Field(
        description="The urgency or importance of the information."
    )

    @field_validator("year", mode="before")
    @classmethod
    def parse_year(cls, v: Any) -> int:
        """Extracts a 4-digit year from a string or returns the integer."""
        if isinstance(v, str):
            match = re.search(r"\d{4}", v)
            if match:
                return int(match.group())
        return v if isinstance(v, int) else 2026

    @field_validator("complexity", "priority", mode="before")
    @classmethod
    def normalize_literals(cls, v: Any) -> str:
        """Normalizes string inputs to match the literal requirements."""
        if isinstance(v, str):
            v = v.strip().capitalize()
            mapping = {
                "Moderate": "Intermediate",
                "High": "High",
                "Medium": "Medium",
                "Low": "Low",
                "Beginner": "Beginner",
                "Advanced": "Advanced",
                "Professional": "Advanced"
            }
            v_clean = v.split()[0]
            return mapping.get(v_clean, v_clean)
        return v

class SearchFilters(BaseModel):
    """
    Structured filters generated from a natural language user query.
    
    Attributes:
        topic (Optional[Union[str, List[str]]]): Topic(s) to filter by.
        year (Optional[int]): Specific year to filter by.
        complexity (Optional[str]): Complexity level to filter by.
        priority (Optional[str]): Priority level to filter by.
        department (Optional[Union[str, List[str]]]): Department(s) for policy searches.
        category (Optional[Union[str, List[str]]]): Category(s) for policy searches.
    """
    topic: Optional[Union[str, List[str]]] = Field(None, description="Topic filter.")
    year: Optional[int] = Field(None, description="Year filter.")
    complexity: Optional[str] = Field(None, description="Complexity filter.")
    priority: Optional[str] = Field(None, description="Priority filter.")
    department: Optional[Union[str, List[str]]] = Field(None, description="Department filter.")
    category: Optional[Union[str, List[str]]] = Field(None, description="Category filter.")
