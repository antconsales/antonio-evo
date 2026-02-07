"""
Response Builder - Assembles final response with audit metadata.

Every response goes through here before being returned.
"""

from typing import Dict, Any

from ..models.response import Response, ResponseMeta
from ..models.policy import PolicyDecision, Classification


class ResponseBuilder:
    """
    Response builder.

    Assembles response with:
    - Handler result
    - Policy decision
    - Classification
    - Timing metadata
    - Audit trail
    """

    def build(
        self,
        result: Response,
        decision: PolicyDecision,
        classification: Classification,
        elapsed_ms: int
    ) -> Dict[str, Any]:
        """
        Build final response dictionary.

        Attaches all metadata for audit.
        """

        # Update meta
        result.meta.elapsed_ms = elapsed_ms
        result.meta.classification = classification.to_dict()
        result.meta.policy_decision = decision.to_dict()

        return result.to_dict()
