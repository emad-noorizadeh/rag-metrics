# -*- coding: utf-8 -*-
# extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union, Optional, Dict, List, Set
import regex as re

# ----- Optional libs (import guarded; none of these are required) -----
try:
    from price_parser import Price
except Exception:
    Price = None

try:
    from number_parser import parse_number
except Exception:
    parse_number = None

try:
    import dateparser
except Exception:
    dateparser = None

try:
    import phonenumbers
    from phonenumbers import PhoneNumberMatcher, PhoneNumberFormat, format_number
except Exception:
    phonenumbers = None

# =========================
# Strong, typed value types
# =========================
@dataclass(frozen=True)
class MoneyValue:
    amount: float
    currency: Optional[str] = None  # "USD"/"EUR"/...

@dataclass(frozen=True)
class NumberValue:
    value: float

@dataclass(frozen=True)
class PercentValue:
    value: float  # as written in text, e.g. "12.5%" -> 12.5

@dataclass(frozen=True)
class DateValue:
    iso: str
    resolution: Literal["date","datetime"] = "date"

@dataclass(frozen=True)
class QuantityValue:
    value: float
    unit: str  # canonical unit symbol, e.g. "kg", "m", "mi", "hr", "L", "GB", "°C"

@dataclass(frozen=True)
class PhoneValue:
    e164: str

ValueUnion = Union[MoneyValue, NumberValue, PercentValue, DateValue, QuantityValue, PhoneValue]
EntityType = Literal["MONEY","NUMBER","PERCENT","DATE","QUANTITY","PHONE"]

@dataclass(frozen=True)
class Entity:
    type: EntityType
    text: str
    span: Tuple[int,int]
    value: ValueUnion
    source: str

# Type guards
def is_money(e: Entity) -> bool: return e.type == "MONEY" and isinstance(e.value, MoneyValue)
def is_number(e: Entity) -> bool: return e.type == "NUMBER" and isinstance(e.value, NumberValue)
def is_percent(e: Entity) -> bool: return e.type == "PERCENT" and isinstance(e.value, PercentValue)
def is_date(e: Entity) -> bool: return e.type == "DATE" and isinstance(e.value, DateValue)
def is_quantity(e: Entity) -> bool: return e.type == "QUANTITY" and isinstance(e.value, QuantityValue)
def is_phone(e: Entity) -> bool: return e.type == "PHONE" and isinstance(e.value, PhoneValue)

# Preference for de-duplication (more specific outranks generic)
_PREF: Dict[EntityType, int] = {"MONEY": 3, "DATE": 2, "QUANTITY": 1, "PHONE": 1, "PERCENT": 0, "NUMBER": 0}

# =========================
# Config (toggle extractors)
# =========================
@dataclass
class ExtractorConfig:
    enable_money: bool = True
    enable_date: bool = True
    enable_quantity: bool = True   # stays True, now pure regex
    enable_phone: bool = False
    enable_number: bool = True
    enable_percent: bool = True
    timezone: str = "UTC"
    prefer_specific_over_generic: bool = True

DEFAULT_CONFIG = ExtractorConfig()

# =========================
# Regex helpers & constants
# =========================

_CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","¥":"JPY"}

# A conservative list of iso-like codes (expand if you need more)
_CUR3_ALLOWED = {
    "USD","EUR","GBP","JPY","CAD","AUD","CHF","CNY","HKD","SGD",
    "INR","NZD","SEK","NOK","DKK","RUB","BRL","MXN","ZAR","KRW",
    "TWD","AED","SAR","TRY","PLN","IDR","THB","MYR","PHP","ILS"
}

MONEY_RX = re.compile(r"""
(?:
  (?P<sym>[$€£¥])\s*
  (?P<amt1>[-+]?\(?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?|\d+(?:\.\d+)?(?:[kKmMbB])?)
 |
  (?P<cur>[A-Z]{3})\s+
  (?P<amt2>[-+]?\(?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?|\d+(?:\.\d+)?(?:[kKmMbB])?)
)
""", re.VERBOSE)

NUM_SHORTHAND_RX = re.compile(r"\b[-+]?\d+(?:\.\d+)?\s*[kKmMbB]?\b")
PARENS_NEG_RX = re.compile(r"\((\d[\d,\.]*)\)")

PERCENT_RX = re.compile(r"\b\(?[-+]?\d+(?:\.\d+)?\)?\s*%\b")

DATE_TOKEN_RX = re.compile(r"""
(
  \b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b
 | \b(?:\d{4}-\d{2}-\d{2})(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?\b
 | \b(?:today|tomorrow|yesterday|next|last|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
 | \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b \s* \d{1,2} (?:,?\s*\d{4})?
)
""", re.IGNORECASE | re.VERBOSE)

# =========================
# DATE
# =========================
def _extract_date(text: str, ref_dt=None, tz="UTC") -> List[Entity]:
    out: List[Entity] = []
    if dateparser is None:
        return out
    settings = {"TIMEZONE": tz}
    if ref_dt is not None:
        settings["RELATIVE_BASE"] = ref_dt
    for m in DATE_TOKEN_RX.finditer(text):
        frag = m.group(0)
        try:
            dt = dateparser.parse(frag, settings=settings)
        except Exception:
            dt = None
        if not dt:
            continue
        resolution: Literal["date","datetime"] = "datetime" if re.search(r"\d{1,2}:\d{2}", frag) else "date"
        iso = dt.isoformat()
        out.append(Entity("DATE", frag, m.span(), DateValue(iso=iso, resolution=resolution), "dateparser"))
    return out

# =========================
# MONEY
# =========================
def _extract_money(text: str) -> List[Entity]:
    out: List[Entity] = []

    # --- price_parser: only accept if surface shows a currency symbol or an allowed 3-letter code
    if Price is not None:
        try:
            p = Price.fromstring(text)
            if p and p.amount is not None:
                surface = (p.amount_text or str(p.amount)) or ""
                has_sym = any(s in surface for s in ("$", "€", "£", "¥"))
                has_code = any(code in surface for code in _CUR3_ALLOWED)
                if has_sym or has_code:
                    amt = float(p.amount)
                    cur = p.currency.upper() if p.currency else None
                    i = text.find(surface) if surface in text else -1
                    span = (i, i + len(surface)) if i >= 0 else (0, 0)
                    out.append(
                        Entity("MONEY", surface, span,
                               MoneyValue(amount=amt, currency=cur),
                               "price-parser")
                    )
        except Exception:
            pass

    # --- regex: symbol or ALLOWED 3-letter code only
    for m in MONEY_RX.finditer(text):
        s = m.group(0)
        cur = m.group("cur") or m.group("sym")
        amt_s = m.group("amt1") or m.group("amt2")

        # handle (1,234) negatives
        neg = False
        pm = PARENS_NEG_RX.search(s)
        if pm:
            neg = True
            amt_s = pm.group(1)

        amt = amt_s.replace(",", "").replace(" ", "")
        mult = 1.0
        if amt.lower().endswith("k"):
            mult, amt = 1e3, amt[:-1]
        elif amt.lower().endswith("m"):
            mult, amt = 1e6, amt[:-1]
        elif amt.lower().endswith("b"):
            mult, amt = 1e9, amt[:-1]

        try:
            val = float(amt) * mult
            if neg:
                val = -val

            cur3 = _CURRENCY_MAP.get(cur, cur.upper() if cur else None)

            # If matched via 3-letter code, enforce whitelist.
            if m.group("cur") and (cur3 not in _CUR3_ALLOWED):
                continue  # skip bogus codes like THE, AND, etc.

            # If there was no symbol AND no valid code, skip (avoid bare numbers)
            if (not m.group("sym")) and (cur3 is None):
                continue

            out.append(Entity("MONEY", s, m.span(), MoneyValue(amount=val, currency=cur3), "money-regex"))
        except Exception:
            pass

    return out
# =========================
# QUANTITY (pure regex + unit tables)
# =========================
# Canonical unit map (lowercased aliases -> canonical symbol)
_UNIT_ALIASES: Dict[str, str] = {
    # length
    "m":"m","meter":"m","meters":"m","metre":"m","metres":"m",
    "km":"km","kilometer":"km","kilometers":"km","kilometre":"km","kilometres":"km",
    "cm":"cm","centimeter":"cm","centimeters":"cm",
    "mm":"mm","millimeter":"mm","millimeters":"mm",
    "mi":"mi","mile":"mi","miles":"mi",
    "yd":"yd","yard":"yd","yards":"yd",
    "ft":"ft","foot":"ft","feet":"ft",
    "in":"in","inch":"in","inches":"in",

    # weight / mass
    "kg":"kg","kilogram":"kg","kilograms":"kg",
    "g":"g","gram":"g","grams":"g",
    "mg":"mg","milligram":"mg","milligrams":"mg",
    "lb":"lb","lbs":"lb","pound":"lb","pounds":"lb",
    "oz":"oz","ounce":"oz","ounces":"oz",

    # time
    "s":"s","sec":"s","secs":"s","second":"s","seconds":"s",
    "ms":"ms","millisecond":"ms","milliseconds":"ms",
    "min":"min","mins":"min","minute":"min","minutes":"min",
    "h":"h","hr":"h","hrs":"h","hour":"h","hours":"h",
    "day":"d","days":"d","d":"d","week":"wk","weeks":"wk","wk":"wk",
    "month":"mo","months":"mo","mo":"mo",
    "year":"yr","years":"yr","yr":"yr","yrs":"yr",

    # volume
    "l":"L","liter":"L","liters":"L","litre":"L","litres":"L",
    "ml":"mL","milliliter":"mL","milliliters":"mL","millilitre":"mL","millilitres":"mL",
    "gal":"gal","gallon":"gal","gallons":"gal",
    "floz":"fl_oz","fl oz":"fl_oz","fluid ounce":"fl_oz","fluid ounces":"fl_oz",

    # area
    "m2":"m^2","sqm":"m^2","square meter":"m^2","square meters":"m^2",
    "ft2":"ft^2","sqft":"ft^2","square foot":"ft^2","square feet":"ft^2",
    "km2":"km^2","square kilometer":"km^2","square kilometers":"km^2",
    "mi2":"mi^2","square mile":"mi^2","square miles":"mi^2",

    # temperature
    "°c":"°C","celsius":"°C",
    "°f":"°F","fahrenheit":"°F",

    # data size
    "kb":"KB","kilobyte":"KB","kilobytes":"KB",
    "mb":"MB","megabyte":"MB","megabytes":"MB",
    "gb":"GB","gigabyte":"GB","gigabytes":"GB",
    "tb":"TB","terabyte":"TB","terabytes":"TB",
}

# ---- Extra units: battery/energy & explicit degree aliases
_UNIT_ALIASES.update({
    "mah": "mAh",     # milliamp-hour
    "wh": "Wh",       # watt-hour
    "kwh": "kWh",
    "°c": "°C",       # ensure degree-C works through the alias map
    "°f": "°F",
})

# Build a permissive unit pattern from aliases (long names first for greediness)
_unit_keys_sorted = sorted(_UNIT_ALIASES.keys(), key=len, reverse=True)
UNIT_PATTERN = r"(?:{})(?:\.)?".format("|".join(re.escape(k) for k in _unit_keys_sorted))

# Match: number + optional space + unit (supports µ/μ, degrees, dots, plurals already covered by aliases)
# Safer construction: build the pattern and inject UNIT_PATTERN via .replace to avoid brace conflicts.
_QUANTITY_CORE = r"""
(?P<val>
    [-+]?
    (?:
        \d+(?:[,\s]\d{3})*(?:\.\d+)?   # 1,234.56
      | \d*\.\d+                       # .75
    )
)
\s*
(?P<unit>
    °[CF] | [µμ]m | {UNITS}
)
(?![A-Za-z-])      # do not allow sticking to more letters/hyphens
"""
QUANTITY_RX = re.compile(
    _QUANTITY_CORE.replace("{UNITS}", UNIT_PATTERN),
    re.IGNORECASE | re.VERBOSE
)

def _normalize_unit(u: str) -> Optional[str]:
    s = u.strip().lower().replace(".", "")
    s = s.replace("μ", "µ")  # normalize mu
    if s in ("°c", "°f"):
        return _UNIT_ALIASES.get(s, s.upper())
    if s in ("µm",):
        return "µm"
    return _UNIT_ALIASES.get(s)

def _extract_quantity(text: str) -> List[Entity]:
    out: List[Entity] = []
    for m in QUANTITY_RX.finditer(text or ""):
        raw_val = m.group("val")
        unit_raw = m.group("unit").strip()
        unit = _normalize_unit(unit_raw)
        if not unit:
            continue
        try:
            v = float(raw_val.replace(",", "").replace(" ", ""))
            span = m.span()
            surface = text[span[0]:span[1]].strip()
            out.append(
                Entity(
                    "QUANTITY",
                    surface,
                    span,
                    QuantityValue(value=v, unit=unit),
                    "quantity-regex"
                )
            )
        except Exception:
            continue
    return out

# =========================
# PHONE
# =========================
def _extract_phone(text: str, region="US") -> List[Entity]:
    out: List[Entity] = []
    if phonenumbers is None:
        return out
    try:
        for m in PhoneNumberMatcher(text, region):
            e164 = format_number(m.number, PhoneNumberFormat.E164)
            out.append(Entity("PHONE", text[m.start:m.end], (m.start, m.end), PhoneValue(e164=e164), "phonenumbers"))
    except Exception:
        pass
    return out

# =========================
# NUMBER & PERCENT
# =========================
def _extract_number_and_percent(text: str, enable_percent: bool = True) -> List[Entity]:
    out: List[Entity] = []
    if parse_number is not None:
        try:
            val = parse_number(text)
            if val is not None:
                out.append(Entity("NUMBER", text, (0, len(text)), NumberValue(value=float(val)), "number-parser"))
        except Exception:
            pass
    if enable_percent:
        for m in PERCENT_RX.finditer(text):
            s = m.group(0)
            core = s.replace("%", "").strip()
            neg = core.startswith("(") and core.endswith(")")
            core = core.strip("()")
            try:
                v = float(core)
                if neg:
                    v = -v
                out.append(Entity("PERCENT", s, m.span(), PercentValue(value=v), "percent-regex"))
            except Exception:
                pass
    for m in NUM_SHORTHAND_RX.finditer(text):
        s = m.group(0)
        core = s.strip("()").replace(",", "").strip()
        neg = s.startswith("(") and s.endswith(")")
        mult = 1.0
        if core.lower().endswith("k"): core = core[:-1]; mult = 1e3
        elif core.lower().endswith("m"): core = core[:-1]; mult = 1e6
        elif core.lower().endswith("b"): core = core[:-1]; mult = 1e9
        try:
            v = float(core) * mult
            if neg: v = -v
            out.append(Entity("NUMBER", s, m.span(), NumberValue(value=v), "number-regex"))
        except Exception:
            pass
    return out

def _extract_number(text: str) -> List[Entity]:
    return _extract_number_and_percent(text, enable_percent=False)

# =========================
# Merge/dedupe & main entry
# =========================
def _inside_any(i: int, j: int, spans: List[Tuple[int,int]]) -> bool:
    for s, e in spans:
        if i >= s and j <= e:
            return True
    return False

def extract_entities(
    text: str,
    config: ExtractorConfig = DEFAULT_CONFIG,
    ref_date=None
) -> List[Entity]:
    candidates: List[Entity] = []
    if config.enable_money:
        candidates += _extract_money(text)
    if config.enable_date:
        candidates += _extract_date(text, ref_dt=ref_date, tz=config.timezone)
    if config.enable_quantity:
        candidates += _extract_quantity(text)
    if config.enable_phone:
        candidates += _extract_phone(text)
    if config.enable_number or config.enable_percent:
        nums = _extract_number_and_percent(text, enable_percent=config.enable_percent)
        # drop numbers/percents that are entirely inside an existing entity span
        taken = [e.span for e in candidates]
        for e in nums:
            if not _inside_any(e.span[0], e.span[1], taken):
                candidates.append(e)

    if not candidates:
        return []

    by_span: Dict[Tuple[int,int], Entity] = {}
    for e in candidates:
        key = e.span
        if key not in by_span:
            by_span[key] = e
        else:
            if config.prefer_specific_over_generic and _PREF[e.type] > _PREF[by_span[key].type]:
                by_span[key] = e
    return sorted(by_span.values(), key=lambda e: (e.span[0], -_PREF[e.type]))

# =========================
# Helper APIs for metrics
# =========================
def extract_by_type(
    text: str,
    types: Optional[Set[EntityType]] = None,
    config: ExtractorConfig = DEFAULT_CONFIG,
    ref_date = None
) -> Dict[EntityType, List[Entity]]:
    ents = extract_entities(text, config=config, ref_date=ref_date)
    out: Dict[EntityType, List[Entity]] = {}
    for e in ents:
        if types is None or e.type in types:
            out.setdefault(e.type, []).append(e)
    return out

def canonicalize_date(dv: DateValue, to: Literal["date","datetime"] = "date") -> DateValue:
    if to == "datetime":
        return dv
    iso = dv.iso.split("T")[0]
    return DateValue(iso=iso, resolution="date")

# =========================
# Equality helpers
# =========================
def money_equal(a: MoneyValue, b: MoneyValue, cents_tol=0.01) -> bool:
    cur_ok = (a.currency == b.currency) or (a.currency is None) or (b.currency is None)
    return cur_ok and abs(a.amount - b.amount) <= cents_tol

def number_equal(a: NumberValue, b: NumberValue, rel=0.01, abs_tol=1e-9) -> bool:
    return abs(a.value - b.value) <= max(abs_tol, rel * max(abs(a.value), abs(b.value)))

def date_equal(a: DateValue, b: DateValue) -> bool:
    return a.iso == b.iso

def match_entity_values(a: Entity, b: Entity) -> bool:
    if a.type != b.type:
        return False
    if is_money(a) and is_money(b):
        return money_equal(a.value, b.value)
    if is_number(a) and is_number(b):
        return number_equal(a.value, b.value)
    if is_percent(a) and is_percent(b):
        return abs(a.value.value - b.value.value) <= max(1e-9, 0.0001 * max(abs(a.value.value), abs(b.value.value)))
    if is_date(a) and is_date(b):
        return a.value.iso == b.value.iso
    if is_quantity(a) and is_quantity(b):
        return (a.value.value == b.value.value) and (a.value.unit == b.value.unit)
    if is_phone(a) and is_phone(b):
        return a.value.e164 == b.value.e164
    return False

def entity_coverage(answer_ents: List[Entity], context_ents: List[Entity]) -> Dict[str, float]:
    by_type_a: Dict[EntityType, List[Entity]] = {}
    by_type_c: Dict[EntityType, List[Entity]] = {}
    for e in answer_ents:
        by_type_a.setdefault(e.type, []).append(e)
    for e in context_ents:
        by_type_c.setdefault(e.type, []).append(e)

    by_type_scores: Dict[str, float] = {}
    covered_total = 0
    total_total = 0

    for t, a_list in by_type_a.items():
        c_list = by_type_c.get(t, [])
        used = [False] * len(c_list)
        hits = 0
        for ae in a_list:
            for i, ce in enumerate(c_list):
                if not used[i] and match_entity_values(ae, ce):
                    used[i] = True
                    hits += 1
                    break
        total_total += len(a_list)
        covered_total += hits
        by_type_scores[t] = (hits / len(a_list)) if a_list else 1.0

    overall = (covered_total / total_total) if total_total else 1.0
    return {"overall": overall, "by_type": by_type_scores}