from pydantic import BaseModel, Field
from typing import List, Optional

class FinancialMetric(BaseModel):
    """Represents a single financial figure from an annual or quarterly report."""
    metric_name: str = Field(..., description="The name of the metric, e.g., Revenue, Gross Profit")
    value: float = Field(..., description="The numerical value of the metric")
    currency: str = Field(default="USD", description="The currency, e.g., USD or CAD")
    fiscal_period: str = Field(..., description="The quarter or year, e.g., Q3 2023 or FY 2023")

class FinancialAnalysis(BaseModel):
    """The structured output for a comparative financial analysis."""
    company: str = Field(..., description="Name of the company, e.g., Shopify")
    metrics: List[FinancialMetric]
    summary_insight: str = Field(..., description="A brief executive summary of the findings")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the extraction (0 to 1)")