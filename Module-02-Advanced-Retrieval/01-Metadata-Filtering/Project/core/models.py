from typing import Optional, Literal
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """
    Schema for automated metadata extraction from research documents.
    This ensures that every chunk of text is tagged with consistent, software-readable data.
    """
    topic: str = Field(
        description="The primary subject matter of the text (e.g., Cybersecurity, Financial Audit, HR Policy, Machine Learning)."
    )
    year: int = Field(
        default=2026,
        description="The primary year or era discussed in the text, or the publication year if no specific date is mentioned. default to 2026."
    )
    complexity: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        description="The technical depth of the content. Beginner (General), Intermediate (Practitioner), Advanced (Expert)."
    )
    audience: str = Field(
        description="The intended reader (e.g., 'Stakeholders', 'Developers', 'Data Scientists')."
    )
    priority: Literal["Low", "Medium", "High"] = Field(
        description="The urgency or importance of the information relative to a corporate or research setting."
    )

class SearchFilters(BaseModel):
    """
    The structured filter generated from a natural language user query.
    All fields are optional because a user might only filter by one or two criteria.
    """
    topic: Optional[str] = Field(None, description="Extract the topic filter if mentioned (e.g., 'security').")
    year: Optional[int] = Field(None, description="Extract the specific year if mentioned (e.g., 2023). default to 2026")
    complexity: Optional[str] = Field(None, description="Extract complexity if mentioned ('Beginner', 'Intermediate', 'Advanced').")
    priority: Optional[str] = Field(None, description="Extract priority if mentioned ('Low', 'Medium', 'High').")
