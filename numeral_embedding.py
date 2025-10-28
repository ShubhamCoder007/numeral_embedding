import re
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

# --------------------------------------
# Utilities: scaling, binning, normalize
# --------------------------------------

class ZScoreScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values: List[float]):
        arr = np.array(values, dtype=float)
        self.mean = float(arr.mean()) if arr.size else 0.0
        self.std = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
        if self.std == 0:
            self.std = 1.0

    def transform(self, values: List[float]) -> List[float]:
        return [ (v - self.mean) / self.std for v in values ]

def log10_safe(x: float) -> float:
    return math.log10(max(x, 1e-9))

def make_bins(min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    return np.linspace(min_val, max_val, num=n_bins + 1)

def soft_bin(value: float, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    hist = np.zeros(n_bins, dtype=float)
    v = min(max(value, edges[0]), edges[-1])
    bin_idx = np.searchsorted(edges, v, side='right') - 1
    if bin_idx == n_bins:
        bin_idx = n_bins - 1
    hist[bin_idx] += 0.6
    if bin_idx - 1 >= 0:
        hist[bin_idx - 1] += 0.2
    if bin_idx + 1 < n_bins:
        hist[bin_idx + 1] += 0.2
    return hist

def l1_normalize(hist: np.ndarray) -> np.ndarray:
    s = hist.sum()
    return hist if s == 0 else hist / s

# --------------------------------------
# Phase 1: Detect numerals with context
# --------------------------------------

NUM_WINDOW = 5  # words around numeral to inspect

CURRENCY_SYMS = r"[\$€£₹]"
CURRENCY_WORDS = ["usd", "eur", "gbp", "inr", "aud", "cad"]
QUANT_SUFFIX = r"(k|m|bn|b)"
QUARTER_PAT = r"(FY\d{2,4}|FY\s?\d{2,4}|Q[1-4]|\d{4}-Q[1-4]|\d{4}-\d{2})"

def tokenize(text: str) -> List[str]:
    # Simple whitespace/token split; you could swap for a better tokenizer
    return re.findall(r"\w+|%|[\$€£₹]|[.,-]", text.lower())

def find_numerals_with_context(text: str) -> List[Dict]:
    tokens = tokenize(text)
    findings = []
    for i, tok in enumerate(tokens):
        # Numeric formats
        is_number = re.match(r"^\d+(\.\d+)?$", tok) is not None
        is_percent = "%" in tok or re.match(r"^\d+(\.\d+)?\s*%$", tok) is not None
        is_currency_symbol = re.match(CURRENCY_SYMS, tok) is not None
        is_km_suffix = re.match(r"^\d+(\.\d+)?\s*(k|m|bn|b)$", tok) is not None

        # Catch adjacent currency patterns like "$" + "120,000"
        number_like = re.match(r"^\d{1,3}(?:,\d{3})*(\.\d+)?$", tok) is not None

        if is_number or is_percent or is_currency_symbol or is_km_suffix or number_like:
            start = max(0, i - NUM_WINDOW)
            end = min(len(tokens), i + NUM_WINDOW + 1)
            context = tokens[start:end]
            findings.append({"token": tok, "index": i, "context": context})
    return findings

# --------------------------------------
# Phase 2: Classify type
# --------------------------------------

TYPE_REVENUE = "revenue"
TYPE_UNITS = "units"
TYPE_MARGIN = "margin"
TYPE_DATE = "date"
TYPE_GENERIC = "generic"

CONTEXT_HINTS = {
    TYPE_REVENUE: ["revenue", "sales", "turnover", "income", "usd", "eur", "gbp"],
    TYPE_UNITS: ["units", "shipments", "quantity", "volume", "orders"],
    TYPE_MARGIN: ["margin", "gross", "net", "profit%", "gm", "gpm", "percentage", "%"],
    TYPE_DATE: ["q1", "q2", "q3", "q4", "fy", "quarter", "year", "month", "week"]
}

def classify_numeral(entry: Dict) -> str:
    tok = entry["token"]
    ctx = entry["context"]

    # Percent direct
    if "%" in tok:
        return TYPE_MARGIN

    # Quarter/date patterns in token or context
    if re.search(QUARTER_PAT, tok, re. IGNORECASE):
        return TYPE_DATE
    if any(re.search(QUARTER_PAT, c, re. IGNORECASE) for c in ctx):
        return TYPE_DATE

    # Currency patterns
    if re.search(CURRENCY_SYMS, tok) or any(c in ctx for c in CURRENCY_WORDS):
        return TYPE_REVENUE
    if re.match(r"^\d+(\.\d+)?\s*(k|m|bn|b)$", tok):
        # Often financial or quantities; use context hint to pick revenue vs units
        if any(h in ctx for h in CONTEXT_HINTS[TYPE_REVENUE]):
            return TYPE_REVENUE
        if any(h in ctx for h in CONTEXT_HINTS[TYPE_UNITS]):
            return TYPE_UNITS
        # Default to revenue if preferring currency magnitude
        return TYPE_REVENUE

    # Context hints
    if any(h in ctx for h in CONTEXT_HINTS[TYPE_MARGIN]):
        return TYPE_MARGIN
    if any(h in ctx for h in CONTEXT_HINTS[TYPE_REVENUE]):
        return TYPE_REVENUE
    if any(h in ctx for h in CONTEXT_HINTS[TYPE_UNITS]):
        return TYPE_UNITS

    # Fallback: generic numeric
    return TYPE_GENERIC

# --------------------------------------
# Parsing numeric values by type
# --------------------------------------

def parse_numeric_value(tok: str, type_hint: str, ctx: List[str]) -> Optional[float]:
    # Strip commas
    t = tok.replace(",", "")

    # Percent
    if type_hint == TYPE_MARGIN and "%" in t:
        try:
            v = float(t.replace("%", "").strip())
            return v / 100.0
        except:
            return None

    # Currency or units with suffix
    m = re.match(r"^(\d+(\.\d+)?)(\s*)(k|m|bn|b)$", t, re. IGNORECASE)
    if m:
        base = float(m.group(1))
        suf = m.group(4).lower()
        mult = {"k": 1e3, "m": 1e6, "bn": 1e9, "b": 1e9}[suf]
        return base * mult

    # Currency symbol followed by number (e.g., $ 120000 or $120,000)
    if re.match(CURRENCY_SYMS, t):
        return None  # symbol alone; next token may hold number

    # Plain number (int/float)
    try:
        return float(t)
    except:
        # Number-like formatted "120,000" handled earlier by stripping commas
        return None

# --------------------------------------
# Encoder with 16 bins per type
# --------------------------------------

class NumericHistogramEncoder:
    def __init__(self, types: List[str], bins_per_type: int = 16, z_clamp: Tuple[float, float] = (-2.5, 2.5), soft: bool = True):
        self.types = types
        self.bins_per_type = bins_per_type
        self.z_min, self.z_max = z_clamp
        self.soft = soft

        # Scalers per type (z-score space)
        self.scalers = {t: ZScoreScaler() for t in types}

        # Bin edges per type created after fit
        self.edges = {t: None for t in types}

    def fit(self, corpus_texts: List[str]):
        # Aggregate values per type from corpus
        vals_by_type = {t: [] for t in self.types}

        for text in corpus_texts:
            findings = find_numerals_with_context(text)
            tokens = tokenize(text)
            for f in findings:
                t_hint = classify_numeral(f)
                if t_hint not in self.types:
                    continue
                value = parse_numeric_value(f["token"], t_hint, f["context"])
                # Handle currency symbol followed by number token
                if value is None and re.match(CURRENCY_SYMS, f["token"]):
                    # Try next token
                    idx = f["index"]
                    if idx + 1 < len(tokens):
                        value = parse_numeric_value(tokens[idx + 1], t_hint, f["context"])
                if value is None:
                    continue
                vals_by_type[t_hint].append(value)

        # Transform and fit scalers
        for t in self.types:
            raw = vals_by_type[t]
            if t in (TYPE_REVENUE, TYPE_UNITS):
                transformed = [log10_safe(v) for v in raw]
            elif t == TYPE_MARGIN:
                transformed = raw  # already 0..1
            else:
                transformed = raw  # generic/date not used here
            self.scalers[t].fit(transformed)
            self.edges[t] = make_bins(self.z_min, self.z_max, self.bins_per_type)

    def _normalize_values(self, values: List[float], t: str) -> List[float]:
        if t in (TYPE_REVENUE, TYPE_UNITS):
            logs = [log10_safe(v) for v in values]
            return self.scalers[t].transform(logs)
        elif t == TYPE_MARGIN:
            return self.scalers[t].transform(values)
        else:
            return []  # skip unknown types in numeric histogram

    def encode_text(self, text: str) -> np.ndarray:
        findings = find_numerals_with_context(text)
        tokens = tokenize(text)

        # Collect values by type
        vals_by_type = {t: [] for t in self.types}

        for f in findings:
            t_hint = classify_numeral(f)
            if t_hint not in self.types:
                continue
            value = parse_numeric_value(f["token"], t_hint, f["context"])
            if value is None and re.match(CURRENCY_SYMS, f["token"]):
                idx = f["index"]
                if idx + 1 < len(tokens):
                    value = parse_numeric_value(tokens[idx + 1], t_hint, f["context"])
            if value is None:
                continue
            vals_by_type[t_hint].append(value)

        # Build per-type histograms
        pieces = []
        for t in self.types:
            edges = self.edges[t]
            hist = np.zeros(self.bins_per_type, dtype=float)
            z_vals = self._normalize_values(vals_by_type[t], t)
            for z in z_vals:
                contrib = soft_bin(z, edges) if self.soft else soft_bin(z, edges)
                hist += contrib
            hist = l1_normalize(hist)
            pieces.append(hist)

        return np.concatenate(pieces, axis=0)

# --------------------------------------
# Similarity functions
# --------------------------------------

def dot_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------------------------------------
# Demo
# --------------------------------------

if __name__ == "__main__":
    corpus_texts = [
        "Q3 FY24: Revenue $120,000; Units 2k; Gross margin 22%. NA region. Shipments up.",
        "FY23 summary: USD 300k revenue. Units 3.5k. Margin 30%.",
        "Q2 update: Revenue $90k and $100k; units 1500, 1.6k; margin 20% and 24%."
    ]

    # Types we will encode (16 bins each)
    types = [TYPE_REVENUE, TYPE_UNITS, TYPE_MARGIN]
    encoder = NumericHistogramEncoder(types=types, bins_per_type=16, z_clamp=(-2.5, 2.5), soft=True)
    encoder.fit(corpus_texts)

    # Encode corpus
    vecs = [encoder.encode_text(t) for t in corpus_texts]

    # Example query: "Revenue near $100k, margin below 25%"
    query_text = "Looking for revenue ~ 100k and margin 24%."
    query_vec = encoder.encode_text(query_text)

    # Compare
    for name, v in zip(["Doc A", "Doc B", "Doc C"], vecs):
        print(f"{name}: dot={dot_similarity(query_vec, v):.4f}, cosine={cosine_similarity(query_vec, v):.4f}")
