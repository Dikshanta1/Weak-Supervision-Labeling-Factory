"""
labeling_functions.py
----------------------
Programmatic labeling functions (LFs) for the weak supervision pipeline.

Each LF is a cheap heuristic: keyword matching, regex, or a tiny sentiment
model. None needs to be individually accurate -- Snorkel's LabelModel later
learns each LF's accuracy and correlations statistically and combines them
into a single high-quality probabilistic label.

Label schema
------------
ABSTAIN            = -1
BUG                 = 0
BILLING             = 1
FEATURE_REQUEST     = 2
UI_UX               = 3
POSITIVE            = 4
CUSTOMER_SERVICE    = 5
"""

import re
from snorkel.labeling import labeling_function
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline
    # Load pipeline globally to avoid reloading per row. 
    # DistilBERT is fast enough for 100k rows locally.
    sentiment_pipe = pipeline(
        "text-classification", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=-1
    )
except ImportError:
    sentiment_pipe = None

ABSTAIN = -1
BUG = 0
BILLING = 1
FEATURE_REQUEST = 2
UI_UX = 3
POSITIVE = 4
CUSTOMER_SERVICE = 5

LABEL_NAMES = {
    ABSTAIN: "ABSTAIN",
    BUG: "BUG",
    BILLING: "BILLING",
    FEATURE_REQUEST: "FEATURE_REQUEST",
    UI_UX: "UI_UX",
    POSITIVE: "POSITIVE",
    CUSTOMER_SERVICE: "CUSTOMER_SERVICE",
}


# ---------------------------------------------------------------------------
# 1-2. Keyword LFs: technical bugs
# ---------------------------------------------------------------------------
@labeling_function()
def lf_crash_keywords(x):
    keywords = ["crash", "freeze", "froze", "glitch", "bug", "force close", "broken", "won't open", "not working"]
    return BUG if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


@labeling_function()
def lf_error_code_regex(x):
    # Matches things like "error 500", "error code: 404", "code 403"
    pattern = r"(error\s?(code)?\s?[:#]?\s?\d{3})|(\bcode\s?[:#]?\s?\d{3}\b)"
    return BUG if re.search(pattern, str(x.text).lower()) else ABSTAIN


# ---------------------------------------------------------------------------
# 3-4. Keyword + regex LFs: billing
# ---------------------------------------------------------------------------
@labeling_function()
def lf_billing_keywords(x):
    keywords = ["refund", "charged", "billing", "subscription", "payment", "billed", "money", "pay", "expensive"]
    return BILLING if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


@labeling_function()
def lf_money_amount_regex(x):
    # Matches "$9.99", "$49", etc. co-occurring with a complaint verb
    has_amount = re.search(r"\$\d+(\.\d{2})?", str(x.text))
    complaint_words = ["charged", "billed", "refund", "deducted", "pay", "cost"]
    if has_amount and any(w in str(x.text).lower() for w in complaint_words):
        return BILLING
    return ABSTAIN


# ---------------------------------------------------------------------------
# 5. Feature requests
# ---------------------------------------------------------------------------
@labeling_function()
def lf_feature_request_keywords(x):
    keywords = ["wish", "would be nice", "please add", "feature request",
                "hoping for", "it'd be nice", "can you add", "should add", "needs"]
    return FEATURE_REQUEST if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


# ---------------------------------------------------------------------------
# 6. UI/UX complaints
# ---------------------------------------------------------------------------
@labeling_function()
def lf_ui_keywords(x):
    keywords = ["confusing", "ugly", "layout", "navigate", "interface",
                "cluttered", "unintuitive", "redesign", "hard to use", "design"]
    return UI_UX if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


# ---------------------------------------------------------------------------
# 7-8. Positive sentiment
# ---------------------------------------------------------------------------
@labeling_function()
def lf_positive_keywords(x):
    keywords = ["love", "great", "awesome", "excellent", "best app",
                "fantastic", "five stars", "highly recommend", "amazing", "good"]
    return POSITIVE if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


@labeling_function()
def lf_transformer_sentiment(x):
    """Enterprise-grade LF: Uses a HuggingFace Transformer for high-precision sentiment.
    Demonstrates blending expensive LLM inference with cheap heuristic rules."""
    if sentiment_pipe is None:
        return ABSTAIN
    
    # Truncate to keep inference blazing fast on 100k rows
    text_snippet = str(x.text)[:128]
    
    try:
        # DistilBERT returns [{'label': 'POSITIVE', 'score': 0.99}]
        result = sentiment_pipe(text_snippet)[0]
        if result['label'] == 'POSITIVE' and result['score'] > 0.95:
            return POSITIVE
    except Exception:
        pass
    return ABSTAIN


# ---------------------------------------------------------------------------
# 9. Customer service complaints
# ---------------------------------------------------------------------------
@labeling_function()
def lf_customer_service_keywords(x):
    keywords = ["customer service", "support team", "no response",
                "waited", "support chat", "unhelpful", "terrible support"]
    return CUSTOMER_SERVICE if any(k in str(x.text).lower() for k in keywords) else ABSTAIN


# ---------------------------------------------------------------------------
# 10. Length/negation heuristic (deliberately low-coverage, catches short
#     terse negative reviews that other LFs miss)
# ---------------------------------------------------------------------------
@labeling_function()
def lf_short_negative(x):
    negative_words = ["bad", "terrible", "awful", "worst", "hate", "sucks", "trash"]
    words = str(x.text).split()
    if len(words) <= 8 and any(w in str(x.text).lower() for w in negative_words):
        return BUG  # heuristic bucket: short angry review -> often a bug report
    return ABSTAIN


ALL_LFS = [
    lf_crash_keywords,
    lf_error_code_regex,
    lf_billing_keywords,
    lf_money_amount_regex,
    lf_feature_request_keywords,
    lf_ui_keywords,
    lf_positive_keywords,
    lf_transformer_sentiment,
    lf_customer_service_keywords,
    lf_short_negative,
]
