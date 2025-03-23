from pydantic import BaseModel, Field


class ExpandedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    expanded_query: str = Field(
        ...,
        description='A unique paraphrasing of the original question.',
    )
