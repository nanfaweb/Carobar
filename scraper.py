"""
PakWheels Used Car Scraper
==========================
Scrapes used car listings from PakWheels.com with anti-ban measures.
Saves results to JSON and CSV.

Usage:
    python scraper.py                        # scrape all (default 5 pages)
    python scraper.py --pages 10             # scrape 10 pages
    python scraper.py --city lahore          # filter by city
    python scraper.py --make toyota          # filter by make
    python scraper.py --max-price 2000000    # filter by max price (PKR)
    python scraper.py --debug                # dump raw HTML to debug/page_N.html
"""

import requests
from bs4 import BeautifulSoup
import json, csv
import time
import random
import logging
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Fix Windows console Unicode (cp1252 can't print arrows/emoji) ──────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL   = "https://www.pakwheels.com"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

CITIES = {
    "karachi":    "ct_75",
    "lahore":     "ct_124",
    "islamabad":  "ct_58",
    "rawalpindi": "ct_159",
    "faisalabad": "ct_42",
    "multan":     "ct_137",
    "peshawar":   "ct_152",
    "quetta":     "ct_158",
}

# All known PakWheels listing card selectors (tried in order)
CARD_SELECTORS = [
    "li.classified-listing",
    "div.classified-listing",
    ".search-li",
    "li[data-listing-id]",
    "li[data-ad-id]",
    "[data-ad-id]",
    ".car-listing",
    ".listing-item",
    ".used-car-item",
    "article.classified",
    ".col-sm-12.used-car",
    "div[class*='listing']",
    "div[class*='car-card']",
    "div[class*='CarCard']",
    "div[class*='search-result']",
]


# ── Session ────────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(get_headers())
    return session


def get_headers() -> dict:
    return {
        "User-Agent":                random.choice(USER_AGENTS),
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language":           "en-PK,en-US;q=0.9,en;q=0.8,ur;q=0.7",
        # Do NOT send Accept-Encoding — let requests handle it automatically.
        # Manually setting it can cause garbled/binary responses if the server
        # sends brotli/gzip but requests isn't given a chance to decode it.
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "same-origin",
        "Referer":                   "https://www.pakwheels.com/",
        "Cache-Control":             "max-age=0",
    }


# ── URL Builder ────────────────────────────────────────────────────────────────
def build_url(page: int = 1, city: str = None, make: str = None, max_price: int = None) -> str:
    params = []
    if make:
        params.append(f"make_{make.lower()}")
    if city and city.lower() in CITIES:
        params.append(CITIES[city.lower()])
    if max_price:
        params.append(f"price_0_{max_price}")

    filter_str = "/".join(params) + "/" if params else ""
    url = f"{BASE_URL}/used-cars/search/{filter_str}?page={page}"
    return url


# ── Fetcher ────────────────────────────────────────────────────────────────────
def fetch_page(session: requests.Session, url: str, retries: int = 3):
    """Returns (BeautifulSoup, raw_html) or (None, None) on failure."""
    for attempt in range(1, retries + 1):
        try:
            session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
            log.info(f"Fetching (attempt {attempt}): {url}")
            response = session.get(url, timeout=15)

            if response.status_code == 200:
                # Force correct encoding detection (avoids garbled/binary text)
                response.encoding = response.apparent_encoding or "utf-8"
                html = response.text
                return BeautifulSoup(html, "lxml"), html
            elif response.status_code == 403:
                log.warning("403 Forbidden - rotating headers and backing off...")
                time.sleep(random.uniform(10, 20))
            elif response.status_code == 429:
                log.warning("429 Rate limited - waiting longer...")
                time.sleep(random.uniform(20, 40))
            else:
                log.warning(f"HTTP {response.status_code} for {url}")

        except requests.RequestException as e:
            log.error(f"Request error: {e}")
            time.sleep(random.uniform(5, 10))

    log.error(f"Failed to fetch after {retries} attempts: {url}")
    return None, None


# ── Debug Helpers ──────────────────────────────────────────────────────────────
def dump_html(html: str, page_num: int, debug_dir: str = "debug"):
    Path(debug_dir).mkdir(exist_ok=True)
    path = f"{debug_dir}/page_{page_num}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"[DEBUG] Raw HTML saved -> {path} (open in browser to inspect)")


def detect_selector(soup: BeautifulSoup) -> str | None:
    for sel in CARD_SELECTORS:
        found = soup.select(sel)
        if found:
            log.info(f"Auto-detected selector: '{sel}' ({len(found)} cards found)")
            return sel
    return None


def print_html_hints(soup: BeautifulSoup):
    log.warning("=== Could not find listing cards ===")
    scripts = soup.find_all("script")
    log.info(f"  - {len(scripts)} <script> tags found (lots = JS-rendered page)")

    hints = [
        ("main",             "main"),
        ("#__next",          "#__next  (Next.js)"),
        ("#app",             "#app    (Vue/React)"),
        (".container",       ".container"),
        (".search-results",  ".search-results"),
    ]
    for sel, label in hints:
        if soup.select_one(sel):
            log.info(f"  - Found element: {label}")

    body = soup.find("body")
    if body:
        preview = body.get_text(separator=" ", strip=True)[:300]
        log.info(f"  - Body preview: {preview!r}")

    log.warning("FIX: Run with --debug, open debug/page_1.html in Chrome.")
    log.warning("FIX: Right-click a car card -> Inspect -> copy its class.")
    log.warning("FIX: Add that class to CARD_SELECTORS at the top of scraper.py.")


# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_price(price_str: str) -> int | None:
    if not price_str:
        return None
    s = price_str.strip().lower()
    try:
        number = float(re.sub(r"[^\d.]", "", s.replace(",", "")))
        if "lakh" in s:
            return int(number * 100_000)
        elif "crore" in s:
            return int(number * 10_000_000)
        return int(number)
    except (ValueError, TypeError):
        return None


def parse_mileage(mileage_str: str) -> int | None:
    if not mileage_str:
        return None
    try:
        return int(re.sub(r"[^\d]", "", mileage_str))
    except ValueError:
        return None


def parse_listing(card) -> dict | None:
    try:
        listing_id = card.get("data-listing-id", "")

        # ── PRIMARY: Schema.org JSON-LD (most reliable, structured by PakWheels) ──
        schema_tag = card.select_one('script[type="application/ld+json"]')
        schema     = {}
        if schema_tag:
            try:
                schema = json.loads(schema_tag.string)
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Title ──
        # Prefer the <a> title attribute — it includes trim (e.g. "Honda N Wgn 2016 Custom G")
        title_tag   = card.select_one("a.car-name")
        title       = title_tag.get("title", "").strip() if title_tag else ""
        if not title:
            title = schema.get("name", "N/A")

        # ── Listing URL ──
        href        = title_tag.get("href", "") if title_tag else ""
        listing_url = schema.get("offers", {}).get("url") or (
            (BASE_URL + href) if href.startswith("/") else href or "N/A"
        )

        # ── Make / Model / Year / Fuel / Transmission / Engine from schema ──
        make         = schema.get("brand", {}).get("name") or schema.get("manufacturer")
        year         = schema.get("modelDate")
        fuel_type    = schema.get("fuelType")
        transmission = schema.get("vehicleTransmission")
        engine_cc    = schema.get("vehicleEngine", {}).get("engineDisplacement")

        # Model = strip make + year from title
        model = title
        if make and model.startswith(make):
            model = model[len(make):].strip()
        if year:
            model = re.sub(rf"\b{year}\b", "", model).strip()
        model = model or None

        # ── Price ──
        price_pkr  = schema.get("offers", {}).get("price")            # clean int from schema
        price_tag  = card.select_one(".price-details")
        price_text = price_tag.get_text(strip=True) if price_tag else ""
        if not price_pkr:
            price_pkr = parse_price(price_text)

        # ── Mileage ──
        mileage_raw = schema.get("mileageFromOdometer", "")           # "138,000 km"
        mileage_km  = parse_mileage(mileage_raw)

        # ── Location — ul.search-vehicle-info first li ──
        location = None
        loc_li   = card.select_one("ul.search-vehicle-info li")
        if loc_li:
            location = loc_li.get_text(strip=True)

        # -- Images: extract photos from data-galleryinfo JSON, skipping videos --
        VIDEO_DOMAINS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")
        VIDEO_EXTS    = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".ogg")
        IMAGE_EXTS    = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif")

        def is_image_url(url: str) -> bool:
            """Return True only if the URL points to an image, not a video."""
            if not url:
                return False
            url_lower = url.lower().split("?")[0]   # strip query params before checking ext
            if any(domain in url_lower for domain in VIDEO_DOMAINS):
                return False
            if any(url_lower.endswith(ext) for ext in VIDEO_EXTS):
                return False
            # Accept if it has a known image extension OR is a pakwheels CDN URL
            has_image_ext  = any(url_lower.endswith(ext) for ext in IMAGE_EXTS)
            is_cdn         = "pakwheels.com" in url_lower
            return has_image_ext or is_cdn

        images     = []
        gallery_ul = card.select_one("ul.image-gallery")
        if gallery_ul:
            try:
                gallery_data = json.loads(gallery_ul.get("data-galleryinfo", "[]"))
                for entry in gallery_data:
                    src = entry.get("src", "")
                    if is_image_url(src):
                        # Convert .webp -> .jpg for broader compatibility
                        images.append(re.sub(r"\.webp$", ".jpg", src))
                    else:
                        log.debug(f"Skipping non-image gallery entry: {src!r}")
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: schema primary image or card thumbnail
        if not images:
            fallback = schema.get("image")
            if not fallback:
                img_tag  = card.select_one("img.pic") or card.select_one("img")
                fallback = img_tag.get("src") if img_tag else None
            if fallback:
                images = [fallback]

        hero_image = images[0] if images else None      # primary / hero image
        image_db   = images[:5]                              # first 5 images for the database

        # -- Last updated --
        dated_tag  = card.select_one(".dated")
        updated_at = dated_tag.get_text(strip=True) if dated_tag else None

        return {
            "listing_id":    listing_id,
            "title":         title,
            "make":          make,
            "model":         model,
            "year":          year,
            "engine_cc":     engine_cc,
            "price_pkr":     price_pkr,
            "price_display": price_text,
            "mileage_km":    mileage_km,
            "fuel_type":     fuel_type,
            "transmission":  transmission,
            "location":      location,
            "hero_image":    hero_image,     # primary display image
            "image_db":      image_db,       # first 5 images for the database
            "listing_url":   listing_url,
            "updated_at":    updated_at,
            "scraped_at":    datetime.now().isoformat(),
        }

    except Exception as e:
        log.warning(f"Error parsing listing {card.get('data-listing-id', '?')}: {e}")
        return None


def parse_listings_page(soup: BeautifulSoup, debug: bool = False, page_num: int = 1) -> list[dict]:
    selector = detect_selector(soup)
    if not selector:
        print_html_hints(soup)
        return []

    cards   = soup.select(selector)
    results = [l for c in cards if (l := parse_listing(c)) and l["title"] != "N/A"]
    log.info(f"Parsed {len(results)} valid listings from {len(cards)} cards")
    return results


def get_total_pages(soup: BeautifulSoup) -> int:
    try:
        nums = [int(a.get_text(strip=True)) for a in soup.select(".pagination li a")
                if a.get_text(strip=True).isdigit()]
        return max(nums) if nums else 1
    except Exception:
        return 1


# ── Savers ─────────────────────────────────────────────────────────────────────
def save_json(data: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(data)} records -> {filepath}")


def save_csv(data: list[dict], filepath: str):
    if not data:
        return
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig = Excel-friendly
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    log.info(f"Saved {len(data)} records -> {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────────
def scrape(
    pages:      int  = 5,
    city:       str  = None,
    make:       str  = None,
    max_price:  int  = None,
    output_dir: str  = "output",
    debug:      bool = False,
) -> list[dict]:

    Path(output_dir).mkdir(exist_ok=True)
    session      = make_session()
    all_listings = []

    log.info("Warming up session (visiting homepage)...")
    fetch_page(session, BASE_URL)
    time.sleep(random.uniform(2, 4))

    first_url              = build_url(page=1, city=city, make=make, max_price=max_price)
    first_soup, first_html = fetch_page(session, first_url)

    if not first_soup:
        log.error("Could not load first page. Aborting.")
        return []

    if debug and first_html:
        dump_html(first_html, page_num=1)

    detected_pages = get_total_pages(first_soup)
    total_pages    = min(pages, detected_pages)
    log.info(f"Detected {detected_pages} pages. Will scrape {total_pages}.")

    listings = parse_listings_page(first_soup, debug=debug, page_num=1)
    all_listings.extend(listings)
    log.info(f"Page 1: {len(listings)} listings (total: {len(all_listings)})")

    for page_num in range(2, total_pages + 1):
        delay = random.uniform(2, 6)
        log.info(f"Waiting {delay:.1f}s...")
        time.sleep(delay)

        url        = build_url(page=page_num, city=city, make=make, max_price=max_price)
        soup, html = fetch_page(session, url)
        if not soup:
            log.warning(f"Skipping page {page_num}")
            continue

        if debug and html:
            dump_html(html, page_num=page_num)

        listings = parse_listings_page(soup, debug=debug, page_num=page_num)
        all_listings.extend(listings)
        log.info(f"Page {page_num}: {len(listings)} listings (total: {len(all_listings)})")

        if page_num % 5 == 0:
            pause = random.uniform(10, 20)
            log.info(f"Long break: {pause:.1f}s")
            time.sleep(pause)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/pakwheels_{timestamp}.json"
    csv_path  = f"{output_dir}/pakwheels_{timestamp}.csv"
    save_json(all_listings, json_path)
    save_csv(all_listings, csv_path)

    log.info(f"Done! Scraped {len(all_listings)} listings.")
    log.info(f"JSON -> {json_path}")
    log.info(f"CSV  -> {csv_path}")

    return all_listings


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PakWheels Used Car Scraper")
    parser.add_argument("--pages",     type=int, default=5,        help="Pages to scrape (default: 5)")
    parser.add_argument("--city",      type=str,                   help=f"City: {', '.join(CITIES.keys())}")
    parser.add_argument("--make",      type=str,                   help="Make: toyota, honda, suzuki ...")
    parser.add_argument("--max-price", type=int,                   help="Max price in PKR (e.g. 2000000)")
    parser.add_argument("--output",    type=str, default="output", help="Output dir (default: output)")
    parser.add_argument("--debug",     action="store_true",        help="Save raw HTML to debug/ folder")
    args = parser.parse_args()

    scrape(
        pages=args.pages,
        city=args.city,
        make=args.make,
        max_price=args.max_price,
        output_dir=args.output,
        debug=args.debug,
    )