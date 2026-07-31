from typing import Literal

from pydantic import BaseModel


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]

