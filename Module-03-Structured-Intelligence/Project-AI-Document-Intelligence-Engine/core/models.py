from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ==========================================================
# Resume Models
# ==========================================================

class Education(BaseModel):
    degree: str
    institution: str
    graduation_year: int


class Experience(BaseModel):
    company: str
    role: str
    duration: str


class Project(BaseModel):
    title: str
    description: str


class Resume(BaseModel):
    name: str
    email: EmailStr
    phone: str
    location: str
    professional_summary: str

    education: list[Education]
    work_experience: list[Experience]
    projects: list[Project]

    skills: list[str]
    certifications: list[str]


# ==========================================================
# Invoice Models
# ==========================================================

class InvoiceItem(BaseModel):
    name: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str

    vendor: str
    customer: str

    items: list[InvoiceItem]

    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)

    payment_due: Optional[str] = None
    payment_method: Optional[str] = None
    payment_status: Optional[str] = None


# ==========================================================
# Support Email Models
# ==========================================================

class SupportEmail(BaseModel):
    subject: str

    customer_name: Optional[str] = None
    customer_email: Optional[EmailStr] = None

    issue: str

    product: Optional[str] = None

    refund_requested: bool = False

    priority: Optional[str] = None