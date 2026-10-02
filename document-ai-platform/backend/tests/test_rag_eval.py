import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualRelevancyMetric
)

@pytest.fixture
def sample_test_case():
    return LLMTestCase(
        input="What are the payment terms under the contract?",
        actual_output="Payment must be settled within 30 days of invoice receipt as stated in Section 4.",
        expected_output="Net 30 days upon invoice receipt.",
        retrieval_context=[
            "Section 4: Payment Terms. The Client agrees to pay invoices within net thirty (30) days of receipt."
        ]
    )

def test_answer_faithfulness(sample_test_case):
    """Verifies that actual_output is factually grounded in retrieval_context (Anti-Hallucination)."""
    metric = FaithfulnessMetric(threshold=0.8, include_reason=True)
    assert_test(sample_test_case, [metric])

def test_answer_relevancy(sample_test_case):
    """Ensures the generated response directly answers the user's specific prompt."""
    metric = AnswerRelevancyMetric(threshold=0.75, include_reason=True)
    assert_test(sample_test_case, [metric])

def test_contextual_relevancy(sample_test_case):
    """Verifies that the retrieved chunks contain relevant information without excess noise."""
    metric = ContextualRelevancyMetric(threshold=0.7, include_reason=True)
    assert_test(sample_test_case, [metric])