import re

# each tuple is (compiled pattern, replacement label)
REDACT_PATTERNS = [

    # email addresses — word chars, dots, +, % before @, then a domain
    (re.compile(r'[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}'), "<email>"),

    # common API / auth tokens that have a recognisable prefix
    (re.compile(r'\b(?:ghp|ghs|gho|sk|hf)_[A-Za-z0-9_-]{10,}\b'), "<token>"),
    (re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'),              "<token>"),
    (re.compile(r'\bBot\s+[A-Za-z0-9._-]{24,}\b'),                 "<token>"),

    # credit / debit card numbers
    (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), "<card>"),

    # OTP / 2FA codes — 6 consecutive digits on their own
    (re.compile(r'\b\d{6}\b'), "<otp>"),

    # phone numbers 
    (re.compile(r'\b(?:\+?\d{1,3}[\s-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b'), "<phone>"),
]


def redact(text: str) -> str:
    for pattern, label in REDACT_PATTERNS:
        text = pattern.sub(label, text)
    return text
