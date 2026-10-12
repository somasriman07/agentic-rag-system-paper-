"""
Backend/guardrails.py
----------------------
Input and output safety guards for the RAG pipeline.

All validators run 100% locally — no remote API calls, no network dependency.

Guards applied
──────────────
INPUT GUARDS — run on every user query inside router_node
  1. PII Detection (Presidio)
       Detects and anonymises emails, phone numbers, Aadhaar, etc. before the
       query reaches Groq, Tavily, or arXiv.
       on_fail = anonymise (replace with <ENTITY_TYPE> placeholders)

  2. Jailbreak Detection (BERT classifier via guardrails-ai-detect-jailbreak)
       Detects adversarial patterns (role-play attacks, "ignore instructions").
       on_fail = block (raises GuardrailBlocked, pipeline returns a refusal)

  3. Prompt Injection Detection (BERT classifier)
       Detects indirect injection attempts hidden in user input.
       on_fail = block (raises GuardrailBlocked)

OUTPUT GUARD — run on every generated answer inside generate_answer_node
  4. PII Detection (Presidio)
       Anonymises PII that leaked from uploaded research papers into the answer.
       on_fail = anonymise

How detection is signalled
──────────────────────────
  validate_input() raises GuardrailBlocked on jailbreak/injection.
  router_node catches it and sets guardrail_blocked=True in RAGState.
  generate_answer_node reads the flag and returns a refusal immediately.
  app.py reads is_blocked from final_values and removes the prompt from history.

Every guardrail event is logged to the terminal with a structured prefix so
operators can monitor the Streamlit console output in real time.

Public API
──────────
  validate_input(query)   -> str            (cleaned) or raises GuardrailBlocked
  validate_output(answer) -> str            (cleaned, never raises)
"""

import logging
from functools import lru_cache

from Backend.logging_config import get_logger

logger = get_logger("guardrails")


# ── Custom exception ──────────────────────────────────────────────────────────

class GuardrailBlocked(Exception):
    """Raised when an input guard blocks a query.

    Carries the guard name and reason so the caller can log them.
    """
    def __init__(self, guard: str, reason: str):
        self.guard  = guard
        self.reason = reason
        super().__init__(f"[{guard}] {reason}")


# ── PII entity configuration ──────────────────────────────────────────────────
# Only anonymise genuinely private/sensitive entities.
# Deliberately excluded: PERSON (author names), LOCATION, DATE_TIME, ORG
# — these are public information in a research context.

_SENSITIVE_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "UK_NHS",
    "SG_NRIC_FIN",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    "IN_AADHAAR",
    "IN_PAN",
    "IN_VOTER",
    "IN_PASSPORT",
]


# ── Presidio PII engine (lazy singleton) ──────────────────────────────────────

@lru_cache(maxsize=1)
def _get_presidio():
    """Return (AnalyzerEngine, AnonymizerEngine) tuple, cached after first call."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        logger.info("GUARDRAIL INIT  | Presidio PII engines ready")
        return analyzer, anonymizer
    except Exception as exc:
        logger.warning("GUARDRAIL INIT  | Presidio unavailable (%s) — PII detection disabled", exc)
        return None, None


def _anonymise_pii(text: str, context: str = "input") -> str:
    """Detect and anonymise sensitive PII entities using Presidio.

    Args:
        text:    The text to scan.
        context: 'input' or 'output' — used in log messages only.

    Returns:
        Text with PII replaced by placeholders, or the original if
        Presidio is unavailable or no sensitive entities were found.
    """
    analyzer, anonymizer = _get_presidio()
    if analyzer is None:
        return text
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=_SENSITIVE_ENTITIES,
        )
        if not results:
            logger.debug("GUARDRAIL PASS  | PII (%s) — no sensitive entities found", context)
            return text

        detected = [(r.entity_type, text[r.start:r.end]) for r in results]
        entity_types = [r.entity_type for r in results]
        logger.warning(
            "GUARDRAIL HIT   | PII (%s) — detected %s — anonymising before %s",
            context,
            entity_types,
            "forwarding to LLM/external APIs" if context == "input" else "showing to user",
        )
        cleaned = anonymizer.anonymize(text=text, analyzer_results=results).text
        logger.info(
            "GUARDRAIL FIX   | PII (%s) — original length %d → cleaned length %d",
            context, len(text), len(cleaned),
        )
        return cleaned
    except Exception as exc:
        logger.warning("GUARDRAIL ERROR | PII (%s) anonymisation failed (%s) — using original", context, exc)
        return text


# ── Jailbreak detection (local BERT classifier) ───────────────────────────────

@lru_cache(maxsize=1)
def _get_jailbreak_validator():
    try:
        from guardrails_ai.detect_jailbreak import DetectJailbreak
        validator = DetectJailbreak()
        logger.info("GUARDRAIL INIT  | DetectJailbreak BERT classifier ready")
        return validator
    except Exception as exc:
        logger.warning("GUARDRAIL INIT  | DetectJailbreak unavailable (%s) — jailbreak detection disabled", exc)
        return None


def _check_jailbreak(text: str) -> None:
    """Raise GuardrailBlocked if the text is classified as a jailbreak attempt."""
    validator = _get_jailbreak_validator()
    if validator is None:
        return
    try:
        from guardrails.validator_base import PassResult
        result = validator.validate(text, {})
        if isinstance(result, PassResult):
            logger.debug("GUARDRAIL PASS  | DetectJailbreak — query is clean")
        else:
            logger.warning(
                "GUARDRAIL BLOCK | DetectJailbreak — adversarial input detected | query=%r",
                text[:80],
            )
            raise GuardrailBlocked("DetectJailbreak", f"Jailbreak pattern detected: {result}")
    except GuardrailBlocked:
        raise
    except Exception as exc:
        logger.warning("GUARDRAIL ERROR | DetectJailbreak check failed (%s) — allowing through", exc)


# ── Prompt injection detection (local BERT classifier) ────────────────────────

@lru_cache(maxsize=1)
def _get_injection_validator():
    try:
        from guardrails_ai.prompt_injection_detector import PromptInjectionDetector
        validator = PromptInjectionDetector()
        logger.info("GUARDRAIL INIT  | PromptInjectionDetector BERT classifier ready")
        return validator
    except Exception as exc:
        logger.warning("GUARDRAILS INIT | PromptInjectionDetector unavailable (%s) — injection detection disabled", exc)
        return None


def _check_prompt_injection(text: str) -> None:
    """Raise GuardrailBlocked if the text contains a prompt injection attempt."""
    validator = _get_injection_validator()
    if validator is None:
        return
    try:
        from guardrails.validator_base import PassResult
        result = validator.validate(text, {})
        if isinstance(result, PassResult):
            logger.debug("GUARDRAIL PASS  | PromptInjectionDetector — query is clean")
        else:
            logger.warning(
                "GUARDRAIL BLOCK | PromptInjectionDetector — injection attempt detected | query=%r",
                text[:80],
            )
            raise GuardrailBlocked("PromptInjectionDetector", f"Prompt injection detected: {result}")
    except GuardrailBlocked:
        raise
    except Exception as exc:
        logger.warning("GUARDRAIL ERROR | PromptInjectionDetector check failed (%s) — allowing through", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_input(query: str) -> str:
    """Run all input guards on the user query.

    Order:
      1. Jailbreak check        — raises GuardrailBlocked on detection
      2. Prompt injection check — raises GuardrailBlocked on detection
      3. PII anonymisation      — returns cleaned text, never raises

    Every event is logged to the terminal with a structured prefix:
      GUARDRAIL INIT   — guard loaded for first time
      GUARDRAIL PASS   — query passed the check
      GUARDRAIL HIT    — PII found and anonymised
      GUARDRAIL BLOCK  — jailbreak/injection detected, query blocked
      GUARDRAIL FIX    — PII replaced with placeholders
      GUARDRAIL ERROR  — unexpected failure (fail-open)

    Args:
        query: Raw user query string.

    Returns:
        The query with PII anonymised.

    Raises:
        GuardrailBlocked: If jailbreak or prompt injection is detected.
    """
    logger.info("GUARDRAIL START | input validation | length=%d chars", len(query))

    # 1. Jailbreak check
    _check_jailbreak(query)

    # 2. Prompt injection check
    _check_prompt_injection(query)

    # 3. PII anonymisation
    cleaned = _anonymise_pii(query, context="input")

    if cleaned == query:
        logger.info("GUARDRAIL DONE  | input clean — no modifications")
    else:
        logger.info("GUARDRAIL DONE  | input sanitised — PII anonymised")

    return cleaned


def validate_output(answer: str) -> str:
    """Run the output PII guard on the generated answer.

    Anonymises PII that may have leaked from uploaded research papers.
    Never raises — always returns a usable answer.

    Args:
        answer: Raw LLM-generated answer string.

    Returns:
        Answer with any PII replaced by placeholders.
    """
    logger.info("GUARDRAIL START | output validation | length=%d chars", len(answer))
    cleaned = _anonymise_pii(answer, context="output")
    if cleaned == answer:
        logger.info("GUARDRAIL DONE  | output clean — no modifications")
    else:
        logger.info("GUARDRAIL DONE  | output sanitised — PII anonymised")
    return cleaned
