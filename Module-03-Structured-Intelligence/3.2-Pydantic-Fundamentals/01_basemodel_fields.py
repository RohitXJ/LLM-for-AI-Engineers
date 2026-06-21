from typing import Optional
from pydantic import BaseModel, Field, ValidationError

# 1. Define a basic schema for a Product in a Catalog
class Product(BaseModel):
    # Field() allows us to add descriptions. When using frameworks like Instructor,
    # these descriptions are passed directly to the LLM as instructions!
    id: str = Field(
        description="A unique alphanumeric identifier for the product, e.g., PROD-123"
    )
    name: str = Field(
        description="The official commercial name of the product"
    )
    price: float = Field(
        gt=0, # Greater than 0 validation built-in!
        description="The unit price of the product in USD"
    )
    description: Optional[str] = Field(
        default=None,
        description="An optional short summary of the product features"
    )
    is_in_stock: bool = Field(
        default=True,
        description="Availability status of the product"
    )

def main():
    print("--- Lesson 3.2.1: Pydantic BaseModel & Fields ---")
    
    # Scenario A: Valid Data (Simulating a perfect LLM response)
    valid_llm_output = {
        "id": "PROD-9081",
        "name": "Quantum Noise-Cancelling Headphones",
        "price": 299.99,
        "description": "Experience pure silence with adaptive ANC technology.",
        "is_in_stock": True
    }
    
    print("\n[1] Parsing valid LLM output...")
    try:
        product = Product(**valid_llm_output)
        print("Successfully validated!")
        print(f"Product Name: {product.name}")
        print(f"Product Price: ${product.price}")
        # We can easily export this back to a clean dictionary or JSON
        print("Serialized JSON:", product.model_dump_json(indent=2))
    except ValidationError as e:
        print("Validation failed unexpectedly:", e)

    # Scenario B: Invalid Data (Simulating an LLM hallucination or bad type)
    invalid_llm_output = {
        "id": "PROD-ERR",
        "name": "Broken Price Headphones",
        "price": -15.00,  # Violates gt=0!
        "is_in_stock": "Yes indeed"  # Pydantic will try to coerce this, but let's see what happens
    }
    
    print("\n[2] Parsing invalid LLM output...")
    try:
        Product(**invalid_llm_output)
    except ValidationError as e:
        print("Validation caught the error successfully!")
        # Pydantic provides structured error messages that we can feed back to the LLM!
        for error in e.errors():
            print(f" - Location: {error['loc']} | Error: {error['msg']} | Type: {error['type']}")

if __name__ == "__main__":
    main()
