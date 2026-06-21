from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ValidationError

# 1. Define an Enum for Invoice Status
class PaymentStatus(str, Enum):
    PAID = "paid"
    UNPAID = "unpaid"
    OVERDUE = "overdue"
    DRAFT = "draft"

# 2. Define a nested model for individual line items
class LineItem(BaseModel):
    description: str = Field(description="Description of the service or item purchased")
    quantity: int = Field(ge=1, description="Quantity of items purchased")
    unit_price: float = Field(gt=0, description="Price per single unit")

    # We can define helper properties on our models
    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price

# 3. Define the main Invoice model containing nested structures
class Invoice(BaseModel):
    invoice_number: str = Field(description="Unique invoice identifier, e.g., INV-2024-001")
    customer_name: str = Field(description="Full name of the client or company")
    status: PaymentStatus = Field(
        description="Current payment status. Must be one of: paid, unpaid, overdue, draft"
    )
    items: List[LineItem] = Field(
        description="List of individual line items included in this invoice"
    )

def main():
    print("--- Lesson 3.2.2: Nested Models, Lists, and Enums ---")

    # Simulating structured data extracted from an unstructured email
    extracted_invoice_data = {
        "invoice_number": "INV-8892",
        "customer_name": "Acme Corp India",
        "status": "unpaid",  # Matches one of our Enum values
        "items": [
            {
                "description": "Cloud Infrastructure Setup",
                "quantity": 1,
                "unit_price": 1500.00
            },
            {
                "description": "Database Migration Support",
                "quantity": 10,
                "unit_price": 120.00
            }
        ]
    }

    print("\n[1] Parsing nested Invoice data...")
    try:
        invoice = Invoice(**extracted_invoice_data)
        print("Invoice validated successfully!")
        print(f"Customer: {invoice.customer_name}")
        print(f"Status: {invoice.status.value.upper()}")
        
        # Calculate total invoice value programmatically
        grand_total = sum(item.total_price for item in invoice.items)
        print(f"Calculated Grand Total: ${grand_total:.2f}")
        
    except ValidationError as e:
        print("Validation failed:", e)

    # Scenario with an invalid Enum value (LLM hallucinated a status)
    bad_invoice_data = extracted_invoice_data.copy()
    bad_invoice_data["status"] = "pending_approval"  # Not in our PaymentStatus Enum!

    print("\n[2] Parsing Invoice with invalid Enum value...")
    try:
        Invoice(**bad_invoice_data)
    except ValidationError as e:
        print("Validation caught the invalid Enum value!")
        print(e)

if __name__ == "__main__":
    main()
