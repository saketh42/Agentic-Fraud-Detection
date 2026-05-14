from pydantic import BaseModel, Field
from typing import List, Optional, Any


class TransactionInput(BaseModel):
    transaction_id: str
    is_fraud: Optional[bool] = None
    fraud_type: Optional[str] = None

    transaction_upi_fraud: int = 0
    transaction_card_fraud: int = 0
    transaction_bank_transfer: int = 0
    commerce_nondelivery: int = 0
    commerce_fake_seller: int = 0
    credential_phishing: int = 0
    social_authority_scam: int = 0
    social_urgency_scam: int = 0
    meta_victim_story: int = 0
    meta_fraud_question: int = 0

    payment_method: str = "unknown"
    fraud_channel: str = "unknown"
    victim_action: str = "unknown"
    request_type: str = "unknown"
    impersonated_entity: str = "unknown"
    amount_mentioned: str = "unknown"
    currency: str = "unknown"
    urgency_level: str = "unknown"
    amount_mentioned_value: float = 0.0

    urgency: float = 0.0
    fear: float = 0.0
    authority: float = 0.0
    reward: float = 0.0


class FeedbackInput(BaseModel):
    transaction_id: str
    is_correct: bool = True
    feedback_type: str = "correction"


class AdversarialQuery(BaseModel):
    transaction_id: str
