from src.agent.utils.verification import VerificationHelper, VerificationConfig


def test_basic_verification_no_sources():
    helper = VerificationHelper()
    result = helper.basic("What is LLM?", [])
    assert "No supporting context" in result


def test_basic_verification_detects_missing_terms():
    helper = VerificationHelper()
    sources = [{"content": "This passage mentions transformers and tokens.", "metadata": {}}]
    result = helper.basic("Explain attention mechanism in LLMs", sources)
    assert "Verification" in result
    assert "attention" in result


def test_self_check_detects_unsupported_sentence():
    helper = VerificationHelper(
        VerificationConfig(stopwords={"the", "and", "in", "is"})
    )
    sources = [
        {
            "content": "Anmol works at EY and builds REST APIs.",
            "metadata": {},
        }
    ]
    answer = "Anmol works at EY. He also manages a bakery."
    result = helper.self_check(answer, sources)
    assert "Self-check" in result
    assert "bakery" in result.lower()
