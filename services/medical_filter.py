import re


INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"forget your instructions",
    r"system prompt",
    r"reveal your prompt",
    r"developer message",
    r"jailbreak",
]


EMERGENCY_WORDS = [
    "heart attack",
    "stroke",
    "severe bleeding",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "suicide",
    "overdose",
]


def validate_query(query):
    """
    Validate the user's query before sending it to the medical LLM.
    """

    if not query or not query.strip():
        return {
            "allowed": False,
            "reason": "Please enter a question."
        }

    query = query.strip()

    # Prevent very large inputs
    if len(query) > 2000:
        return {
            "allowed": False,
            "reason": "Question is too long."
        }

    # Prompt injection protection
    lower_query = query.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_query):
            return {
                "allowed": False,
                "reason": "This type of instruction is not allowed."
            }

    # Emergency detection
    for word in EMERGENCY_WORDS:
        if word in lower_query:
            return {
                "allowed": True,
                "emergency": True,
                "reason": (
                    "This may be an emergency. "
                    "Please seek immediate professional medical help."
                )
            }

    return {
        "allowed": True,
        "emergency": False,
        "reason": "Query passed the safety check."
    }
