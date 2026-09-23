import re

REFUSAL = (
    "This system is designed only for medical and clinical questions. "
    "Please ask a medical or healthcare-related question."
)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|your\s+)?(previous|prior|system)\s+instructions",
    r"forget\s+(your|the)\s+(instructions|rules)",
    r"pretend\s+(you\s+are|this\s+is)\s+(a\s+)?non[-\s]?medical",
    r"bypass\s+(the\s+|your\s+)?(medical|system)\s+restriction",
    r"jailbreak",
    r"developer\s+message",
]

EMERGENCY_TERMS = [
    "can't breathe", "cannot breathe", "difficulty breathing",
    "severe chest pain", "unconscious", "not breathing",
    "severe bleeding", "overdose", "anaphylaxis",
    "stroke symptoms", "suicide attempt", "suicidal"
]

def validate_query(text):
    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    if len(normalized) > 2000:
        return {
            "allowed": False,
            "message": "Please keep the question below 2000 characters.",
            "reason": "input-too-long",
            "emergency": False
        }

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "allowed": False,
                "message": REFUSAL,
                "reason": "prompt-injection",
                "emergency": False
            }

    emergency = any(term in normalized for term in EMERGENCY_TERMS)

    return {
        "allowed": True,
        "message": "",
        "reason": "passed-input-validation",
        "emergency": emergency
    }
