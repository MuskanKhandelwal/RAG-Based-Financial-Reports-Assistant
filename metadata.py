# Extract filing-level metadata (company, ticker, form type, fiscal year) from
# the cover pages of a financial report so it can be attached to every chunk.
#
# Filenames in data/ are not reliable indicators of the issuer -- for example
# both "10-Q4-2024-As-Filed.pdf" and "_10-K-Q4-2023-As-Filed.pdf" are Apple
# filings -- so the cover page text is the primary source and the filename is
# only a last-resort fallback.

import os
import re
from datetime import datetime

# Pages scanned for cover-page markers. SEC cover pages are page 1, but wrapped
# annual reports place the 10-K cover behind a glossy front section.
COVER_PAGE_SCAN_LIMIT = 12

UNKNOWN = "unknown"

# Registrant name as it appears on an SEC cover page:
# "Commission File Number: 001-36743 Apple Inc. (Exact name of Registrant ..."
REGISTRANT_RE = re.compile(
    r"Commission File Number[:\s]*[\d\-]+\s+(.{3,60}?)\s*\(\s*Exact name",
    re.IGNORECASE,
)

# Exchange listing line, common in press releases:
# "Meta Platforms, Inc. (Nasdaq: META) today reported ..."
LISTING_RE = re.compile(
    r"([A-Z][A-Za-z0-9.,&'\- ]{2,50}?),?\s*\(\s*(?:Nasdaq|NASDAQ|NYSE)[:\s]+([A-Z.]{1,6})\s*\)"
)

FORM_TYPE_RE = re.compile(r"FORM\s+(10-K|10-Q|8-K|20-F)", re.IGNORECASE)

# An SEC cover page always carries this header. Press releases only mention form
# types in passing ("our Quarterly Report on Form 10-Q"), so require the header
# before trusting a FORM match.
SEC_COVER_RE = re.compile(
    r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION", re.IGNORECASE
)

PRESS_RELEASE_RE = re.compile(
    r"(?:news release|reports?\s+(?:first|second|third|fourth)\s+quarter"
    r"|reports?\s+(?:fourth quarter and )?full year)",
    re.IGNORECASE,
)

# "fiscal year ended September 28, 2024" / "quarter ended March 31, 2024".
# The month is captured loosely because PDF extraction frequently splits words
# ("ENDEDS EPTEMBER 3, 2023" in Costco's wrapped annual report); internal
# spaces are removed before parsing.
PERIOD_END_RE = re.compile(
    r"(?:fiscal year|quarter|year|period)\s+ended\s*"
    r"([A-Za-z][A-Za-z ]{2,12}?)\s*(\d{1,2})\s*,\s*(\d{4})",
    re.IGNORECASE,
)

# Ticker lookup for issuers whose cover page carries no exchange listing line.
TICKER_BY_COMPANY = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "costco": "COST",
    "meta": "META",
}

# Canonical issuer names. Registrant names read off a PDF cover page are often
# broken by extraction artifacts ("Costco WholesaleC orporation"), and metadata
# used for filtering has to be consistent across every chunk of every filing.
COMPANY_BY_TICKER = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla, Inc.",
    "COST": "Costco Wholesale Corporation",
    "META": "Meta Platforms, Inc.",
}

# Last-resort mapping from filename fragments, used only when the document text
# yields no registrant name at all.
COMPANY_BY_FILENAME_HINT = {
    "cost": ("Costco Wholesale Corporation", "COST"),
    "tsla": ("Tesla, Inc.", "TSLA"),
    "aapl": ("Apple Inc.", "AAPL"),
    "meta": ("Meta Platforms, Inc.", "META"),
}


def _normalize(text):
    """Collapse whitespace so cover-page patterns survive PDF line breaks."""
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_date(month, day, year):
    """Parse a cover-page date, tolerating months split by PDF extraction."""
    cleaned = f"{month.replace(' ', '')} {day}, {year}"
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _ticker_for(company):
    lowered = company.lower()
    for key, ticker in TICKER_BY_COMPANY.items():
        if key in lowered:
            return ticker
    return UNKNOWN


def _company_from_filename(source):
    stem = os.path.basename(source or "").lower()
    for hint, (company, ticker) in COMPANY_BY_FILENAME_HINT.items():
        if hint in stem:
            return company, ticker
    return UNKNOWN, UNKNOWN


def extract_document_metadata(cover_text, source=""):
    """Derive filing-level metadata from the concatenated cover-page text.

    :param cover_text: Text of the first COVER_PAGE_SCAN_LIMIT pages.
    :param source: Path of the source file, used only for fallbacks.
    :return: Dict of scalar metadata safe to store on a Chroma chunk.
    """
    text = _normalize(cover_text)

    company, ticker = UNKNOWN, UNKNOWN

    registrant = REGISTRANT_RE.search(text)
    if registrant:
        company = registrant.group(1).strip(" .,")
        ticker = _ticker_for(company)
    else:
        listing = LISTING_RE.search(text)
        if listing:
            company = listing.group(1).strip(" .,")
            ticker = listing.group(2).strip(".")

    if company == UNKNOWN:
        company, ticker = _company_from_filename(source)
    elif ticker == UNKNOWN:
        ticker = _ticker_for(company)

    # Prefer the canonical spelling so filters match across every chunk.
    company = COMPANY_BY_TICKER.get(ticker, company)

    form_match = FORM_TYPE_RE.search(text)
    if form_match and SEC_COVER_RE.search(text):
        form_type = form_match.group(1).upper()
    elif PRESS_RELEASE_RE.search(text):
        form_type = "PRESS_RELEASE"
    elif re.search(r"annual report", text[:2000], re.IGNORECASE):
        form_type = "ANNUAL_REPORT"
    else:
        form_type = UNKNOWN

    period_end, fiscal_year = None, None
    period_match = PERIOD_END_RE.search(text)
    if period_match:
        period_end = _parse_date(*period_match.groups())
    if period_end:
        fiscal_year = int(period_end[:4])
    else:
        # Fall back to the most plausible four-digit year on the cover page.
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", text[:3000])]
        if years:
            fiscal_year = max(years)

    return {
        "company": company,
        "ticker": ticker,
        "form_type": form_type,
        "fiscal_year": fiscal_year if fiscal_year is not None else -1,
        "period_end": period_end or UNKNOWN,
        "doc_id": os.path.basename(source or UNKNOWN),
    }


def metadata_for_documents(documents):
    """Build a {source: metadata} map from LangChain page Documents.

    Pages are grouped by source so each filing's cover text is assembled once.

    :param documents: Page-level Documents as returned by a PDF loader.
    :return: Dict mapping source path to its filing metadata.
    """
    covers = {}
    for doc in documents:
        source = doc.metadata.get("source", UNKNOWN)
        page = doc.metadata.get("page")
        if page is None or page < COVER_PAGE_SCAN_LIMIT:
            covers.setdefault(source, []).append((page if page is not None else 0, doc.page_content))

    metadata_by_source = {}
    for source, pages in covers.items():
        cover_text = " ".join(content for _, content in sorted(pages, key=lambda p: p[0]))
        metadata_by_source[source] = extract_document_metadata(cover_text, source)
    return metadata_by_source
