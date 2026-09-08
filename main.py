import os
import base64
import secrets
import hashlib
import hmac
import time
import urllib.parse
import json
import re
from difflib import SequenceMatcher
from collections import Counter

import requests
import redis

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=25, max_retries=1)

ETSY_KEYSTRING = os.getenv("ETSY_API_KEYSTRING")
ETSY_SHARED_SECRET = os.getenv("ETSY_SHARED_SECRET")
ETSY_REDIRECT_URI = "https://etsy-ai-agent.onrender.com/etsy/callback"

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL environment variable is missing.")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

TOKEN_KEY = "etsy:oauth_token"
SETTINGS_KEY = "etsy:listing_settings"
SHOP_CONTEXT_KEY = "etsy:shop_context"
SHOP_CONTEXT_TTL = 1800

DEFAULT_SHIPPING_TITLE = "Free Shipping"
DEFAULT_PROCESSING_TITLE = "2–5 days"

UNSUPPORTED_CLAIM_PATTERNS = [
    r"\bnatural\b",
    r"\bgenuine\b",
    r"\bauthentic\b",
    r"\bsolid gold\b",
    r"\b\d{2}k gold\b",
    r"\b\d{2}k\b",
    r"\b925\b",
    r"\bstirling silver\b",
    r"\bcertified\b",
    r"\bcertificate\b",
    r"\buntreated\b",
    r"\bno treatment\b",
    r"\bconflict[- ]free\b",
    r"\b\d+(?:\.\d+)?\s*(?:ct|carat|carats)\b",
]

# -------------------------------------------------------------------
# BASIC
# -------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent",
        "mode": "existing_listing_seo_optimizer",
        "optimizer_ui": "/optimizer",
        "docs": "/docs",
    }


# -------------------------------------------------------------------
# REDIS TOKEN STORAGE
# -------------------------------------------------------------------

def save_etsy_token(token_data):
    redis_client.set(TOKEN_KEY, json.dumps(token_data))


def get_etsy_token():
    data = redis_client.get(TOKEN_KEY)
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


def refresh_etsy_token():
    token_data = get_etsy_token()
    if not token_data:
        return None

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return None

    response = requests.post(
        "https://api.etsy.com/v3/public/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": ETSY_KEYSTRING,
            "refresh_token": refresh_token,
        },
        timeout=15,
    )

    if not response.ok:
        return None

    new_token = response.json()
    new_token["expires_at"] = int(time.time()) + int(
        new_token.get("expires_in", 3600)
    )
    save_etsy_token(new_token)
    return new_token


def get_valid_etsy_token():
    token_data = get_etsy_token()
    if not token_data:
        return None

    expires_at = token_data.get("expires_at")
    if expires_at and int(time.time()) >= int(expires_at) - 120:
        refreshed = refresh_etsy_token()
        if refreshed:
            token_data = refreshed

    return token_data


# -------------------------------------------------------------------
# ETSY HTTP HELPERS
# -------------------------------------------------------------------

def etsy_headers(access_token):
    return {
        "x-api-key": f"{ETSY_KEYSTRING}:{ETSY_SHARED_SECRET}",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def etsy_form_headers(access_token):
    return {
        "x-api-key": f"{ETSY_KEYSTRING}:{ETSY_SHARED_SECRET}",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def etsy_public_headers():
    return {
        "x-api-key": f"{ETSY_KEYSTRING}:{ETSY_SHARED_SECRET}",
    }


def etsy_get(url, access_token=None, params=None):
    headers = etsy_headers(access_token) if access_token else etsy_public_headers()

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=12,
    )

    if access_token and response.status_code == 401:
        refreshed = refresh_etsy_token()
        if refreshed:
            new_access_token = refreshed.get("access_token")
            response = requests.get(
                url,
                headers=etsy_headers(new_access_token),
                params=params,
                timeout=12,
            )

    return response


# -------------------------------------------------------------------
# OAUTH
# -------------------------------------------------------------------

def create_oauth_signature(state, code_verifier, timestamp):
    message = f"{state}|{code_verifier}|{timestamp}".encode("utf-8")
    return hmac.new(
        ETSY_SHARED_SECRET.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


@app.get("/etsy/connect")
def etsy_connect():
    if not ETSY_KEYSTRING or not ETSY_SHARED_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Etsy API credentials are not configured.",
        )

    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)

    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode("utf-8")).digest()
    ).decode("utf-8").rstrip("=")

    timestamp = str(int(time.time()))

    signature = create_oauth_signature(
        state,
        code_verifier,
        timestamp,
    )

    cookie_value = f"{state}|{code_verifier}|{timestamp}|{signature}"

    params = {
        "response_type": "code",
        "client_id": ETSY_KEYSTRING,
        "redirect_uri": ETSY_REDIRECT_URI,
        "scope": "listings_r listings_w shops_r",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    etsy_url = (
        "https://www.etsy.com/oauth/connect?"
        + urllib.parse.urlencode(params)
    )

    response = RedirectResponse(url=etsy_url)

    response.set_cookie(
        key="etsy_oauth",
        value=cookie_value,
        max_age=600,
        httponly=True,
        secure=True,
        samesite="lax",
    )

    return response


@app.get("/etsy/callback")
def etsy_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    error_description: str = None,
):
    if error:
        return {
            "status": "etsy_authorization_failed",
            "error": error,
            "description": error_description,
        }

    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Missing Etsy authorization code or state.",
        )

    oauth_cookie = request.cookies.get("etsy_oauth")
    if not oauth_cookie:
        raise HTTPException(
            status_code=400,
            detail="OAuth session cookie is missing. Please start again from /etsy/connect.",
        )

    try:
        cookie_state, code_verifier, timestamp, signature = (
            oauth_cookie.split("|", 3)
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth session.",
        )

    expected_signature = create_oauth_signature(
        cookie_state,
        code_verifier,
        timestamp,
    )

    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth session signature.",
        )

    if not hmac.compare_digest(cookie_state, state):
        raise HTTPException(
            status_code=400,
            detail="OAuth state mismatch.",
        )

    try:
        timestamp_int = int(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth timestamp.",
        )

    if int(time.time()) - timestamp_int > 600:
        raise HTTPException(
            status_code=400,
            detail="OAuth session expired. Please start again.",
        )

    token_response = requests.post(
        "https://api.etsy.com/v3/public/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": ETSY_KEYSTRING,
            "redirect_uri": ETSY_REDIRECT_URI,
            "code": code,
            "code_verifier": code_verifier,
        },
        timeout=30,
    )

    if not token_response.ok:
        raise HTTPException(
            status_code=token_response.status_code,
            detail=token_response.text,
        )

    token_data = token_response.json()

    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=500,
            detail="Etsy did not return an access token.",
        )

    token_data["expires_at"] = int(time.time()) + int(
        token_data.get("expires_in", 3600)
    )

    save_etsy_token(token_data)

    response = RedirectResponse(url="/etsy/status")
    response.delete_cookie(
        key="etsy_oauth",
        secure=True,
        samesite="lax",
    )

    return response


@app.get("/etsy/status")
def etsy_status():
    token_data = get_valid_etsy_token()

    if not token_data:
        return {
            "connected": False,
            "message": "Etsy account is not connected.",
        }

    access_token = token_data.get("access_token")
    if not access_token:
        return {
            "connected": False,
            "message": "Etsy access token is missing.",
        }

    user_id = access_token.split(".")[0]

    shops_url = (
        f"https://api.etsy.com/v3/application/users/"
        f"{user_id}/shops"
    )

    shop_response = etsy_get(
        shops_url,
        access_token,
    )

    if not shop_response.ok:
        return {
            "connected": True,
            "shop_loaded": False,
            "message": "Etsy authorization is stored, but shop information could not be loaded.",
            "etsy_status_code": shop_response.status_code,
        }

    shop_data = shop_response.json()

    return {
        "connected": True,
        "shop_loaded": True,
        "message": "Etsy account connected successfully.",
        "shop": shop_data,
    }


# -------------------------------------------------------------------
# SHOP CONFIG
# -------------------------------------------------------------------

def get_shop_context():
    token_data = get_valid_etsy_token()

    if not token_data:
        raise HTTPException(status_code=401, detail="Etsy account is not connected.")

    access_token = token_data.get("access_token")
    user_id = access_token.split(".")[0]

    cached = redis_client.get(SHOP_CONTEXT_KEY)
    if cached:
        try:
            cached_data = json.loads(cached)
            if cached_data.get("shop_id"):
                cached_data["access_token"] = access_token
                cached_data["user_id"] = user_id
                return cached_data
        except Exception:
            pass

    shops_url = f"https://api.etsy.com/v3/application/users/{user_id}/shops"
    shop_response = etsy_get(shops_url, access_token)
    if not shop_response.ok:
        raise HTTPException(status_code=shop_response.status_code, detail=shop_response.text)

    shop_data = shop_response.json()
    shop = shop_data.get("shop", shop_data)
    shop_id = shop.get("shop_id")

    processing_url = f"https://api.etsy.com/v3/application/shops/{shop_id}/readiness-state-definitions"
    shipping_url = f"https://api.etsy.com/v3/application/shops/{shop_id}/shipping-profiles"
    taxonomy_url = "https://api.etsy.com/v3/application/seller-taxonomy/nodes"

    processing_response = etsy_get(processing_url, access_token)
    shipping_response = etsy_get(shipping_url, access_token)
    taxonomy_response = etsy_get(taxonomy_url, access_token)

    cached_data = {
        "shop": shop,
        "shop_id": shop_id,
        "processing": processing_response.json() if processing_response.ok else {},
        "shipping": shipping_response.json() if shipping_response.ok else {},
        "taxonomy": taxonomy_response.json() if taxonomy_response.ok else {},
    }

    try:
        redis_client.setex(SHOP_CONTEXT_KEY, SHOP_CONTEXT_TTL, json.dumps(cached_data))
    except Exception:
        pass

    cached_data["access_token"] = access_token
    cached_data["user_id"] = user_id
    return cached_data


def choose_shipping_profile(shipping_data):
    results = shipping_data.get("results", [])

    for item in results:
        if str(item.get("title", "")).strip().lower() == DEFAULT_SHIPPING_TITLE.lower():
            return item

    for item in results:
        title = str(item.get("title", "")).lower()
        if "free shipping" in title:
            return item

    return results[0] if results else None


def choose_processing_profile(processing_data):
    results = processing_data.get("results", [])

    for item in results:
        min_days = item.get("min_processing_days")
        max_days = item.get("max_processing_days")

        if min_days == 2 and max_days == 5:
            return item

        title = str(item.get("title", "")).lower()
        if "2" in title and "5" in title:
            return item

    for item in results:
        title = str(item.get("title", "")).lower()
        if "2–5" in title or "2-5" in title:
            return item

    return results[0] if results else None


@app.get("/etsy/config")
def etsy_config():
    context = get_shop_context()

    return {
        "shop_id": context["shop_id"],
        "shop_name": context["shop"].get("shop_name"),
        "processing_profiles": context["processing"],
        "shipping_profiles": context["shipping"],
        "seller_taxonomy": context["taxonomy"],
        "agent_defaults": {
            "shipping_profile": DEFAULT_SHIPPING_TITLE,
            "processing_profile": DEFAULT_PROCESSING_TITLE,
            "listing_state": "draft",
            "auto_publish": False,
        },
    }


# -------------------------------------------------------------------
# TAXONOMY HELPERS
# -------------------------------------------------------------------

def taxonomy_nodes(data):
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return data["results"]

        if isinstance(data.get("nodes"), list):
            return data["nodes"]

    if isinstance(data, list):
        return data

    return []


def flatten_taxonomy(nodes, parent_path=None):
    parent_path = parent_path or []
    flattened = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = (
            node.get("id")
            or node.get("taxonomy_id")
            or node.get("node_id")
        )

        name = (
            node.get("name")
            or node.get("title")
            or node.get("display_name")
            or ""
        )

        current_path = parent_path + ([name] if name else [])

        if node_id:
            flattened.append({
                "id": node_id,
                "name": name,
                "path": current_path,
            })

        children = (
            node.get("children")
            or node.get("nodes")
            or node.get("child_nodes")
            or []
        )

        if isinstance(children, list):
            flattened.extend(
                flatten_taxonomy(children, current_path)
            )

    return flattened


def choose_taxonomy_id(taxonomy_data, product_text):
    nodes = taxonomy_nodes(taxonomy_data)
    flat = flatten_taxonomy(nodes)

    if not flat:
        return None, []

    text = product_text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))

    jewelry_words = {
        "ring", "rings", "earring", "earrings", "necklace",
        "necklaces", "bracelet", "bracelets", "pendant",
        "pendants", "jewelry", "jewellery", "brooch",
        "brooches", "gemstone", "opal", "ruby", "emerald",
        "sapphire", "tourmaline", "topaz", "coral",
    }

    scored = []

    for item in flat:
        path_text = " ".join(item["path"]).lower()
        path_tokens = set(re.findall(r"[a-z0-9]+", path_text))

        overlap = len(tokens & path_tokens)

        jewelry_bonus = 0
        if tokens & jewelry_words and (
            "jewelry" in path_text
            or "jewellery" in path_text
            or "ring" in path_text
            or "earring" in path_text
            or "necklace" in path_text
            or "bracelet" in path_text
            or "pendant" in path_text
        ):
            jewelry_bonus = 5

        leaf_bonus = 1 if item["name"] else 0

        product_type_bonus = 0
        product_type_terms = {
            "ring", "earring", "necklace", "bracelet", "pendant",
            "brooch", "cuff", "chain", "jewelry", "jewellery"
        }
        if any(term in tokens for term in product_type_terms):
            matched_types = tokens & product_type_terms & path_tokens
            product_type_bonus = 10 if matched_types else 0

        score = overlap + jewelry_bonus + product_type_bonus + leaf_bonus

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    candidates = [
        {
            "taxonomy_id": item["id"],
            "name": item["name"],
            "path": item["path"],
            "score": score,
        }
        for score, item in scored[:10]
    ]

    if not scored or scored[0][0] <= 0:
        return None, candidates

    return scored[0][1]["id"], candidates


# -------------------------------------------------------------------
# ETSY MARKET KEYWORD RESEARCH
# -------------------------------------------------------------------

def extract_keyword_signals(listings, max_terms=18):
    """Extract recurring 2-3 word search phrases from marketplace titles.
    These are used only as keyword signals; seller wording is never copied.
    """
    stop = {
        "handmade", "jewelry", "jewellery", "ring", "rings", "gift",
        "gifts", "for", "the", "and", "with", "women", "woman",
        "mens", "men", "natural", "genuine", "authentic", "beautiful",
        "elegant", "perfect", "unique", "stone", "gemstone", "silver",
        "gold", "fashion", "style", "wear", "wearable", "present",
        "anniversary", "birthday", "wedding", "statement"
    }
    counter = Counter()
    for item in listings:
        title = normalize_text(item.get("title", ""))
        w = [x for x in title.split() if len(x) >= 3]
        # Count useful 2- and 3-word phrases, while retaining product terms.
        for n in (2, 3):
            for i in range(len(w) - n + 1):
                phrase = " ".join(w[i:i+n])
                parts = phrase.split()
                if all(x in stop for x in parts):
                    continue
                if sum(x not in stop for x in parts) == 0:
                    continue
                counter[phrase] += 1
    return [p for p, _ in counter.most_common(max_terms)]


def market_keyword_research(seed_text):
    """One fast Etsy marketplace lookup used to discover buyer-language signals."""
    words_seed = [
        w for w in words(seed_text)
        if len(w) >= 3 and w not in {
            "handmade", "jewelry", "jewellery", "beautiful", "elegant",
            "natural", "genuine", "authentic", "gift", "women", "woman"
        }
    ]
    query = " ".join(words_seed[:5]).strip()
    if not query:
        query = "gemstone ring"
    try:
        results = public_competitor_search(query, limit=12)
        return {
            "query": query,
            "result_count": len(results),
            "keyword_signals": extract_keyword_signals(results),
        }
    except Exception:
        return {
            "query": query,
            "result_count": 0,
            "keyword_signals": [],
        }


# -------------------------------------------------------------------
# AI LISTING GENERATION
# -------------------------------------------------------------------

def clean_json_text(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def parse_listing_json(text):
    cleaned = clean_json_text(text)

    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("AI did not return valid JSON.")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("AI listing result is not an object.")

    return data


def generate_listing(product, details, seller_claims="", keyword_signals=None):
    keyword_signals = keyword_signals or []

    prompt = f"""
You are a senior Etsy SEO copywriter and conversion-focused listing strategist
for handmade gemstone jewelry. Your goal is to create a listing that is highly
relevant to real buyer searches while remaining clear, original, truthful, and
pleasant to read.

PRODUCT / IMAGE CONTEXT:
{product}

SELLER DETAILS:
{details}

SELLER-CONFIRMED CLAIMS:
{seller_claims}

CURRENT ETSY MARKET KEYWORD SIGNALS (use only as search-language clues):
{json.dumps(keyword_signals, ensure_ascii=False)}

CURRENT ETSY TITLE GUIDANCE:
- Clearly name the item once.
- Put the most important objective traits near the beginning: product type,
  gemstone/color, material, and another genuinely important differentiator.
- Prefer a concise title of roughly 8-15 words when possible; never exceed 140 characters.
- Do not keyword-stuff, repeat the same word, or stack synonyms unnaturally.
- Do not add "best", "perfect", "beautiful", "unique", "must have", or similar
  subjective sales language to the title.
- Do not add shipping, price, sale, or generic recipient/gift phrases unless they
  are genuinely essential to what the item is.

SEO / CONVERSION REQUIREMENTS:
- Build the title around the strongest, most specific buyer intent.
- Use the keyword signals only to understand search language. NEVER copy a
  competitor title or distinctive phrase.
- Use all 13 tags. Tags should be natural multi-word phrases, diverse, specific,
  and complementary rather than 13 near-duplicates.
- Use attributes/materials as supporting relevance rather than stuffing the title.
- The first sentence of the description must immediately identify the item and
  its strongest buyer-relevant traits.
- Description should be original and conversion-focused: what it is, design/details,
  materials/facts, who it suits or occasions when genuinely relevant, and a concise
  buyer-information section. Do not invent sizing, care, origin, certification,
  treatment, carat weight, or other facts.
- Do not make medical/healing claims.
- Do not claim natural/genuine/authentic/925/metal purity/etc. unless seller supplied it.

FACT SAFETY:
- Only use gemstone identity, metal type, purity, natural/genuine status, origin,
  treatment, certification, dimensions, carat weight and measurements when the seller
  supplied that fact.
- Never infer these claims from a photograph.

ORIGINALITY:
- Write from scratch.
- Do not imitate a competitor's sentence structure or distinctive wording.
- Common product/SEO words may naturally overlap.

Return ONLY valid JSON with exactly this structure:
{{
  "title": "string",
  "tags": ["exactly 13 tags"],
  "description": "string",
  "materials": ["seller-confirmed materials only"],
  "gift_keywords": ["short relevant phrases only"],
  "observed_facts": ["facts supplied by seller or visibly observable"],
  "verification_needed": ["claims that still need seller verification"]
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    return parse_listing_json(response.output_text)


# -------------------------------------------------------------------
# LISTING VALIDATION
# -------------------------------------------------------------------

def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def words(text):
    return normalize_text(text).split()


def phrase_ngrams(text, n=5):
    w = words(text)
    return {
        " ".join(w[i:i+n])
        for i in range(max(0, len(w) - n + 1))
    }


def similarity_score(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b),
    ).ratio()


def listing_similarity(candidate, existing):
    candidate_title = candidate.get("title", "")
    candidate_tags = " ".join(candidate.get("tags", []))
    candidate_description = candidate.get("description", "")

    existing_title = existing.get("title", "")
    existing_tags = " ".join(existing.get("tags", []))
    existing_description = existing.get("description", "")

    title_score = similarity_score(
        candidate_title,
        existing_title,
    )

    full_candidate = (
        candidate_title + " " +
        candidate_tags + " " +
        candidate_description
    )

    full_existing = (
        existing_title + " " +
        existing_tags + " " +
        existing_description
    )

    full_score = similarity_score(
        full_candidate,
        full_existing,
    )

    candidate_phrases = phrase_ngrams(full_candidate, 5)
    existing_phrases = phrase_ngrams(full_existing, 5)

    common_phrases = candidate_phrases & existing_phrases

    return {
        "title_similarity": round(title_score, 4),
        "overall_similarity": round(full_score, 4),
        "common_5_word_phrases": list(common_phrases)[:10],
        "risk": (
            title_score >= 0.82
            or full_score >= 0.78
            or len(common_phrases) >= 2
        ),
    }


def get_own_active_listings(access_token, shop_id, limit=50):
    url = (
        f"https://api.etsy.com/v3/application/shops/"
        f"{shop_id}/listings/active"
    )

    response = etsy_get(
        url,
        access_token,
        params={
            "limit": min(limit, 100),
            "offset": 0,
            "legacy": "false",
        },
    )

    if not response.ok:
        return []

    data = response.json()
    return [
        x for x in data.get("results", [])
        if x.get("state") == "active"
    ]


def public_competitor_search(keyword, limit=15):
    url = "https://api.etsy.com/v3/application/listings/active"

    response = etsy_get(
        url,
        access_token=None,
        params={
            "keywords": keyword,
            "limit": min(limit, 20),
            "offset": 0,
            "sort_on": "score",
            "sort_order": "desc",
        },
    )

    if not response.ok:
        return []

    data = response.json()
    return data.get("results", [])


def competitor_candidates(candidate):
    title_words = [
        w for w in words(candidate.get("title", ""))
        if len(w) >= 4
    ]

    # Prefer specific words instead of generic Etsy words.
    generic = {
        "handmade", "jewelry", "jewellery", "women",
        "woman", "gift", "gifts", "beautiful", "elegant",
        "natural", "stone", "gemstone",
    }

    specific = [
        w for w in title_words
        if w not in generic
    ]

    queries = []

    if len(specific) >= 2:
        queries.append(" ".join(specific[:3]))

    if len(specific) >= 1:
        queries.append(specific[0])

    if not queries:
        queries.append("gemstone jewelry")

    listings = []

    for query in queries[:1]:
        try:
            listings.extend(public_competitor_search(query, 15))
        except Exception:
            continue

    # Remove duplicates.
    unique = {}
    for listing in listings:
        listing_id = listing.get("listing_id")
        if listing_id:
            unique[listing_id] = listing

    return list(unique.values())[:5]


def originality_report(candidate, own_listings, competitor_listings):
    checks = []

    for item in own_listings:
        result = listing_similarity(candidate, item)

        if result["risk"]:
            checks.append({
                "source": "own_shop",
                "listing_id": item.get("listing_id"),
                "title": item.get("title"),
                **result,
            })

    for item in competitor_listings:
        result = listing_similarity(candidate, item)

        if result["risk"]:
            checks.append({
                "source": "etsy_marketplace",
                "listing_id": item.get("listing_id"),
                "title": item.get("title"),
                **result,
            })

    checks.sort(
        key=lambda x: max(
            x.get("title_similarity", 0),
            x.get("overall_similarity", 0),
        ),
        reverse=True,
    )

    return {
        "passed": len(checks) == 0,
        "matches": checks[:10],
    }


def has_unsupported_claims(candidate, seller_details, seller_claims):
    allowed_text = (
        str(seller_details or "") + " " +
        str(seller_claims or "")
    ).lower()

    combined = (
        str(candidate.get("title", "")) + " " +
        str(candidate.get("description", "")) + " " +
        " ".join(candidate.get("tags", []))
    ).lower()

    problems = []

    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        found = re.findall(pattern, combined, flags=re.I)
        if not found:
            continue

        # If the seller explicitly supplied the same claim, allow it.
        try:
            if re.search(pattern, allowed_text, flags=re.I):
                continue
        except re.error:
            pass

        problems.append(pattern)

    return problems


def validate_listing(candidate):
    errors = []

    title = str(candidate.get("title", "")).strip()
    tags = candidate.get("tags", [])
    description = str(candidate.get("description", "")).strip()

    if not title:
        errors.append("Title is empty.")

    if len(title) > 140:
        errors.append("Title is longer than 140 characters.")

    title_word_list = words(title)
    if len(title_word_list) > 15:
        errors.append("Title should be 15 words or fewer for buyer-friendly readability.")

    if len(title_word_list) != len(set(title_word_list)):
        errors.append("Title repeats words; rewrite for cleaner readability.")

    subjective_title_words = {
        "best", "perfect", "beautiful", "unique", "must", "amazing",
        "stunning", "gorgeous", "premium", "luxury"
    }
    if subjective_title_words.intersection(title_word_list):
        errors.append("Title contains subjective sales language; keep it factual and buyer-friendly.")

    if not isinstance(tags, list):
        errors.append("Tags must be a list.")
        tags = []

    if len(tags) != 13:
        errors.append("Exactly 13 tags are required.")

    cleaned_tags = []
    for tag in tags:
        tag = str(tag).strip()

        if not tag:
            errors.append("A tag is empty.")
            continue

        if len(tag) > 20:
            errors.append(f"Tag exceeds 20 characters: {tag}")

        cleaned_tags.append(tag)

    candidate["tags"] = cleaned_tags

    if not description:
        errors.append("Description is empty.")

    return errors


def rewrite_listing(candidate, similarity_matches, claim_problems):
    prompt = f"""
Rewrite this Etsy listing into a fresh, original version.

CURRENT LISTING:
{json.dumps(candidate, ensure_ascii=False)}

SIMILARITY WARNINGS:
{json.dumps(similarity_matches, ensure_ascii=False)}

UNSUPPORTED CLAIM WARNINGS:
{json.dumps(claim_problems, ensure_ascii=False)}

Rules:
- Preserve only facts already present in the listing.
- Do not invent facts.
- Do not reuse distinctive wording from the similarity warnings.
- Do not use unsupported gemstone/metal claims.
- Title <= 140 characters.
- Exactly 13 tags.
- Each tag <= 20 characters.
- Keep SEO useful but natural.
- Return ONLY valid JSON using:
{{
  "title": "string",
  "tags": ["13 tags"],
  "description": "string",
  "materials": ["strings"],
  "gift_keywords": ["strings"],
  "observed_facts": ["strings"],
  "verification_needed": ["strings"]
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    return parse_listing_json(response.output_text)


# -------------------------------------------------------------------
# IMAGE GENERATION / ANALYSIS
# -------------------------------------------------------------------

@app.post("/generate-listing-photo")
async def generate_listing_photo(
    image: UploadFile = File(...),
    extra_info: str = Form(""),
):
    """
    Full photo-to-listing workflow.

    This endpoint analyzes the image, generates the Etsy listing,
    validates claims/SEO, checks similarity against Etsy listings,
    and rewrites when needed. It does NOT create or publish an Etsy draft.
    """

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await image.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Please use an image under 10 MB.",
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:{image.content_type};base64,{image_base64}"

    # ---------------------------------------------------------------
    # 1. IMAGE ANALYSIS
    # ---------------------------------------------------------------

    analysis_prompt = f"""
Analyze this jewelry photograph for an Etsy listing.

Seller-confirmed information:
{extra_info}

Only identify what can reasonably be observed from the photograph.
Do not infer gemstone identity, natural/genuine status, metal purity,
origin, treatment, certification, measurements or carat weight from
appearance alone.

Return a concise analysis covering:
1. product type
2. visible design details
3. visible colors
4. visible setting details
5. facts requiring seller verification
"""

    analysis_response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": analysis_prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            }
        ],
    )

    analysis = analysis_response.output_text

    # ---------------------------------------------------------------
    # 2. GENERATE FULL ETSY LISTING
    # ---------------------------------------------------------------

    product_context = (
        "Jewelry product shown in the uploaded photograph.\n\n"
        "IMAGE OBSERVATIONS:\n" + analysis
    )

    # Use a single marketplace lookup as keyword research before copywriting.
    keyword_research = market_keyword_research(
        product_context + " " + extra_info
    )

    listing = generate_listing(
        product_context,
        extra_info,
        extra_info,
        keyword_research.get("keyword_signals", []),
    )

    # ---------------------------------------------------------------
    # 3. VALIDATION + ORIGINALITY SCREEN + AUTO REWRITE
    # ---------------------------------------------------------------

    context = get_shop_context()

    own_listings = get_own_active_listings(
        context["access_token"],
        context["shop_id"],
    )

    rewrite_count = 0
    originality = None
    claim_problems = []
    validation_errors = []
    competitors = competitor_candidates(listing)

    for attempt in range(2):
        validation_errors = validate_listing(listing)

        claim_problems = has_unsupported_claims(
            listing,
            extra_info,
            extra_info,
        )

        originality = originality_report(
            listing,
            own_listings,
            competitors,
        )

        if (
            not validation_errors
            and not claim_problems
            and originality["passed"]
        ):
            break

        if attempt == 1:
            return {
                "status": "blocked_before_draft",
                "message": (
                    "The listing did not pass validation/originality checks. "
                    "No Etsy draft was created."
                ),
                "analysis": analysis,
                "listing": listing,
                "validation_errors": validation_errors,
                "unsupported_claims": claim_problems,
                "originality": originality,
                "rewrite_attempts": rewrite_count,
                "draft_created": False,
                "published": False,
            }

        listing = rewrite_listing(
            listing,
            originality.get("matches", []),
            claim_problems,
        )
        rewrite_count += 1

    return {
        "status": "success",
        "message": (
            "Full Etsy listing generated and passed the current checks. "
            "No Etsy draft was created by this endpoint."
        ),
        "analysis": analysis,
        "keyword_research": keyword_research,
        "listing": listing,
        "validation_errors": validation_errors,
        "unsupported_claims": claim_problems,
        "originality": originality,
        "rewrite_attempts": rewrite_count,
        "draft_created": False,
        "published": False,
        "next_step": "POST /create-draft-listing",
    }


@app.post("/generate-listing")
async def generate_listing_endpoint(
    product: str = Form(...),
    details: str = Form(""),
):
    listing = generate_listing(product, details)

    errors = validate_listing(listing)

    return {
        "status": "success" if not errors else "validation_failed",
        "listing": listing,
        "validation_errors": errors,
    }


# -------------------------------------------------------------------
# ORIGINALITY CHECK ONLY
# -------------------------------------------------------------------

@app.post("/check-originality")
async def check_originality(
    title: str = Form(...),
    description: str = Form(""),
    tags: str = Form(""),
):
    context = get_shop_context()

    candidate = {
        "title": title,
        "description": description,
        "tags": [
            x.strip()
            for x in tags.split(",")
            if x.strip()
        ],
    }

    own = get_own_active_listings(
        context["access_token"],
        context["shop_id"],
    )

    competitors = competitor_candidates(candidate)

    report = originality_report(
        candidate,
        own,
        competitors,
    )

    return {
        "status": "success",
        "originality": report,
        "checked_own_shop_listings": len(own),
        "checked_marketplace_candidates": len(competitors),
        "note": (
            "This is a similarity screen, not a legal copyright guarantee. "
            "Common SEO words may overlap naturally."
        ),
    }



# -------------------------------------------------------------------
# EXISTING ETSY LISTING SEO OPTIMIZER
# -------------------------------------------------------------------

def extract_listing_id(value):
    """Accept either a numeric Etsy listing ID or an Etsy listing URL."""
    value = str(value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return value
    match = re.search(r"/listing/(\d+)", value)
    if match:
        return match.group(1)
    match = re.search(r"(?:listing_id|listingId)[=/](\d+)", value, flags=re.I)
    if match:
        return match.group(1)
    return None


def listing_market_queries(listing):
    """Build a small set of buyer-language queries from an existing listing."""
    stop = {
        "handmade", "jewelry", "jewellery", "ring", "rings", "gift",
        "gifts", "women", "woman", "men", "mens", "beautiful", "elegant",
        "natural", "genuine", "authentic", "unique", "stone", "gemstone",
        "silver", "gold", "style", "fashion", "statement", "present",
    }

    title_words = [
        w for w in words(listing.get("title", ""))
        if len(w) >= 3 and w not in stop
    ]

    existing_tags = [
        normalize_text(x)
        for x in listing.get("tags", [])
        if str(x).strip()
    ]

    queries = []

    if len(title_words) >= 3:
        queries.append(" ".join(title_words[:3]))
    elif len(title_words) >= 2:
        queries.append(" ".join(title_words[:2]))

    for tag in existing_tags[:3]:
        if tag and tag not in queries:
            queries.append(tag)

    # Keep requests deliberately small so the optimizer remains fast.
    unique = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in unique:
            unique.append(q)

    return unique[:3]


def marketplace_keyword_signals(queries, per_query=15):
    """Research ranked Etsy marketplace results and score buyer-language signals.

    This is directional marketplace evidence, not exact Etsy search volume.
    Scores combine ranked-result weight, phrase frequency, query coverage, and
    competitor-tag frequency when those fields are returned by Etsy.
    """
    candidates = []
    seen = set()
    query_result_counts = {}

    for query in queries[:3]:
        results = public_competitor_search(query, per_query)
        query_result_counts[query] = len(results)

        for rank, item in enumerate(results):
            listing_id = item.get("listing_id")
            if listing_id and listing_id in seen:
                continue
            if listing_id:
                seen.add(listing_id)

            candidates.append({
                "listing_id": listing_id,
                "rank": rank + 1,
                "query": query,
                "title": item.get("title", ""),
                "tags": item.get("tags", []) or [],
                "url": item.get("url", ""),
            })

    stop = {
        "the", "and", "for", "with", "from", "this", "that", "your",
        "handmade", "jewelry", "jewellery", "gift", "gifts", "women",
        "woman", "men", "mens", "necklace", "necklaces", "natural",
        "genuine", "authentic", "beautiful", "elegant", "fashion",
        "style", "stone", "gemstone", "piece", "pieces",
    }

    phrase_frequency = Counter()
    phrase_weight = Counter()
    phrase_queries = defaultdict(set)
    tag_frequency = Counter()
    tag_weight = Counter()
    tag_queries = defaultdict(set)

    for item in candidates:
        rank = max(int(item.get("rank", 1)), 1)
        weight = 1.0 / rank
        title_words = [w for w in words(item.get("title", "")) if len(w) >= 3]

        # Use contiguous 2- and 3-word phrases from competitor titles.
        for n in (2, 3):
            for i in range(len(title_words) - n + 1):
                phrase_words = title_words[i:i+n]
                if any(w in stop for w in phrase_words) and n == 2:
                    # Keep useful product phrases such as "ruby necklace" but
                    # discard phrases made mostly from generic terms.
                    useful = sum(w not in stop for w in phrase_words)
                    if useful < 1:
                        continue
                phrase = " ".join(phrase_words)
                if phrase in stop or len(phrase) < 5:
                    continue
                phrase_frequency[phrase] += 1
                phrase_weight[phrase] += weight
                phrase_queries[phrase].add(item.get("query", ""))

        for raw_tag in item.get("tags", []):
            tag = re.sub(r"\s+", " ", normalize_text(raw_tag)).strip()
            if not tag or len(tag) > 20:
                continue
            if tag in stop:
                continue
            tag_frequency[tag] += 1
            tag_weight[tag] += weight
            tag_queries[tag].add(item.get("query", ""))

    def ranked(counter, weighted, query_sets, limit=25):
        rows = []
        for term, freq in counter.items():
            coverage = len(query_sets[term])
            score = weighted[term] + (0.35 * freq) + (0.75 * coverage)
            rows.append({
                "keyword": term,
                "frequency": freq,
                "query_coverage": coverage,
                "signal_score": round(score, 4),
            })
        rows.sort(key=lambda x: (x["signal_score"], x["frequency"]), reverse=True)
        return rows[:limit]

    phrase_rows = ranked(phrase_frequency, phrase_weight, phrase_queries, 25)
    tag_rows = ranked(tag_frequency, tag_weight, tag_queries, 25)

    sample_listings = []
    for item in sorted(candidates, key=lambda x: x.get("rank", 999))[:10]:
        sample_listings.append({
            "query": item.get("query", ""),
            "rank": item.get("rank"),
            "title": item.get("title", ""),
            "listing_id": item.get("listing_id"),
            "url": item.get("url", ""),
        })

    return {
        "queries": queries[:3],
        "query_result_counts": query_result_counts,
        "marketplace_results_checked": len(candidates),
        "high_signal_phrases": phrase_rows,
        "high_signal_tags": tag_rows,
        "sample_listings": sample_listings,
        "method_note": (
            "Signal score uses Etsy ranked marketplace results, phrase frequency, "
            "query coverage, and rank weighting. It is not Etsy search volume."
        ),
    }


def optimize_existing_listing(listing, market_signals):
    current = {
        "title": listing.get("title", ""),
        "tags": listing.get("tags", []) or [],
        "description": listing.get("description", ""),
        "materials": listing.get("materials", []) or [],
        "taxonomy_id": listing.get("taxonomy_id"),
    }

    prompt = f"""
You are an elite Etsy SEO and conversion strategist specializing in handmade
Gemstone Jewelry. You are optimizing an EXISTING Etsy listing, not creating a
random generic listing.

CURRENT ETSY LISTING:
{json.dumps(current, ensure_ascii=False)}

MARKETPLACE RESEARCH SIGNALS:
{json.dumps(market_signals, ensure_ascii=False)}

INTERPRETATION:
- Give more weight to keywords with stronger signal_score, higher frequency, and wider query coverage.
- Prefer phrases that accurately match this listing over merely frequent generic phrases.
- Do not blindly copy competitor tags or titles; use them only as market-language evidence.
- Build a balanced tag set across core product, gemstone, material, style/use, occasion, and buyer-intent phrases where truthful.

GOAL:
Create a significantly better, buyer-focused Etsy title and 13 tags based on
what the actual product/listing says and the language appearing in ranked Etsy
marketplace results.

IMPORTANT LIMITATION:
The Etsy API does NOT provide exact keyword search-volume numbers. Therefore,
do NOT claim that a keyword is "the #1 most searched" or give fake search-volume
numbers. Treat the marketplace frequency/ranking signals above as directional
buyer-language evidence only.

TITLE RULES:
- Put the actual product name first.
- Put the strongest objective differentiators early.
- Prefer a concise, readable title, generally about 8-15 words.
- Never exceed 140 characters.
- Do not repeat the same keyword unnaturally.
- Do not use subjective fluff such as best, perfect, gorgeous, stunning, must-have.
- Do not copy a competitor title or distinctive phrase.
- Do not add facts that are absent from the current listing.

TAG RULES:
- Exactly 13 tags.
- Every tag <= 20 characters.
- Use diverse buyer intents rather than 13 variations of one phrase.
- Prefer specific multi-word phrases.
- Use marketplace signals where relevant, but only if they accurately describe
  this product.
- Never invent gemstone identity, metal purity, origin, treatment, certification,
  measurements, carat weight, or other unsupported facts.

DESCRIPTION RULES:
- Rewrite the description only when useful; preserve factual information.
- First 1-2 sentences should immediately explain what the item is and its strongest
  objective traits.
- Make it natural and conversion-focused, not keyword stuffed.
- Never make healing/medical claims.

Return ONLY valid JSON:
{{
  "current_listing_analysis": {{
    "strengths": ["..."],
    "weaknesses": ["..."],
    "seo_opportunities": ["..."]
  }},
  "recommended_title": "...",
  "recommended_tags": ["exactly 13 tags"],
  "recommended_description": "...",
  "keyword_strategy": [
    {{"keyword": "...", "reason": "..."}}
  ],
  "changes_summary": ["..."],
  "search_volume_note": "Exact Etsy search volume is not available through the API; these are marketplace ranking/frequency signals."
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    return parse_listing_json(response.output_text)


@app.get("/optimizer", response_class=HTMLResponse)
def optimizer_page():
    """Human-friendly UI for the existing Etsy listing SEO optimizer.

    This page only calls /analyze-existing-listing. It never writes to Etsy.
    """
    return HTMLResponse(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Etsy AI SEO Optimizer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background:#0b1020; color:#edf2ff; }
  .wrap { max-width:1100px; margin:0 auto; padding:32px 18px 60px; }
  .hero { margin-bottom:22px; }
  h1 { margin:0 0 8px; font-size:32px; }
  .sub { color:#aab6d3; margin:0; line-height:1.55; }
  .card { background:#121a2e; border:1px solid #263454; border-radius:16px; padding:20px; margin-top:18px; box-shadow:0 10px 30px rgba(0,0,0,.18); }
  label { display:block; font-size:14px; font-weight:700; margin-bottom:8px; }
  input { width:100%; padding:13px 14px; border-radius:10px; border:1px solid #344463; background:#0c1325; color:#fff; outline:none; }
  input:focus { border-color:#7189ff; }
  button { margin-top:14px; border:0; border-radius:10px; padding:12px 18px; font-weight:800; cursor:pointer; background:#7189ff; color:#071022; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  .status { margin-top:12px; color:#aab6d3; min-height:22px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .full { grid-column:1 / -1; }
  .section-title { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:12px; }
  h2 { font-size:19px; margin:0; }
  .copy { margin:0; padding:7px 10px; font-size:12px; border:1px solid #3b4a6c; background:#18233d; color:#dce5ff; }
  .title-box, .description { white-space:pre-wrap; line-height:1.6; background:#0b1223; border:1px solid #293957; border-radius:10px; padding:14px; }
  .title-box { font-size:18px; font-weight:750; }
  .description { min-height:170px; }
  ol, ul { margin:0; padding-left:22px; line-height:1.7; }
  .tags { display:flex; flex-wrap:wrap; gap:8px; }
  .tag { background:#1a2744; border:1px solid #34486f; border-radius:999px; padding:7px 10px; font-size:13px; }
  .muted { color:#9eabc8; font-size:13px; }
  .success { color:#87e0ad; }
  .warn { color:#ffd27a; }
  .error { color:#ff9c9c; }
  .meta { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .meta div { background:#0b1223; border:1px solid #293957; border-radius:10px; padding:12px; }
  .meta strong { display:block; font-size:12px; color:#8f9fbe; margin-bottom:4px; }
  .keyword { padding:10px 0; border-bottom:1px solid #263454; }
  .keyword:last-child { border-bottom:0; }
  .keyword b { font-size:14px; }
  .keyword span { display:block; color:#aab6d3; font-size:13px; margin-top:3px; }
  .hidden { display:none; }
  @media(max-width:760px) { .grid { grid-template-columns:1fr; } .full { grid-column:auto; } .meta { grid-template-columns:1fr; } h1 { font-size:27px; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>🚀 Etsy AI SEO Optimizer</h1>
    <p class="sub">Analyze an existing Etsy listing and get a buyer-friendly title, 13 SEO tags, optimized description and keyword strategy. <b>Nothing is edited or published.</b></p>
  </div>
  <div class="card">
    <form id="form">
      <label for="listing_url">Etsy listing URL</label>
      <input id="listing_url" name="listing_url" type="url" placeholder="https://www.etsy.com/listing/123456789/..." required>
      <button id="run" type="submit">Analyze Listing</button>
      <div id="status" class="status"></div>
    </form>
  </div>
  <div id="result" class="hidden">
    <div class="card">
      <div class="section-title"><h2>🏷️ Recommended SEO Title</h2><button type="button" class="copy" data-copy="title">Copy</button></div>
      <div id="title" class="title-box"></div><p id="titleCount" class="muted"></p>
    </div>
    <div class="grid">
      <div class="card">
        <div class="section-title"><h2>🔖 13 Etsy Tags</h2><button type="button" class="copy" data-copy="tags">Copy</button></div>
        <div id="tags" class="tags"></div><p class="muted">Each tag is checked against the 20-character limit.</p>
      </div>
      <div class="card">
        <div class="section-title"><h2>📝 Optimized Description</h2><button type="button" class="copy" data-copy="description">Copy</button></div>
        <div id="description" class="description"></div>
      </div>
      <div class="card">
        <div class="section-title"><h2>📊 Current Listing Analysis</h2></div>
        <div><b>Strengths</b><ul id="strengths"></ul></div><br>
        <div><b>Weaknesses</b><ul id="weaknesses"></ul></div><br>
        <div><b>SEO Opportunities</b><ul id="opportunities"></ul></div>
      </div>
      <div class="card">
        <div class="section-title"><h2>🔍 Keyword Strategy</h2></div><div id="keywords"></div>
      </div>
      <div class="card full">
        <div class="section-title"><h2>🔄 What Changed</h2></div><ul id="changes"></ul><p id="volumeNote" class="muted"></p>
      </div>
      <div class="card full">
        <div class="section-title"><h2>📌 Current Listing</h2></div>
        <div class="meta">
          <div><strong>CURRENT TITLE</strong><span id="currentTitle"></span></div>
          <div><strong>LISTING ID</strong><span id="listingId"></span></div>
          <div><strong>WRITE ACTION</strong><span class="success">Not performed</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let latest = { title:'', tags:[], description:'' };
function listInto(el, items) { el.innerHTML=''; (items||[]).forEach(x=>{const li=document.createElement('li');li.textContent=x;el.appendChild(li);}); }
function render(data) {
  const o=data.optimized_result||{}, a=o.current_listing_analysis||{};
  latest={title:o.recommended_title||'',tags:o.recommended_tags||[],description:o.recommended_description||''};
  $('title').textContent=latest.title;
  $('titleCount').textContent=`${latest.title.length} characters • ${latest.title.trim()?latest.title.trim().split(/\s+/).length:0} words`;
  $('tags').innerHTML=''; latest.tags.forEach((tag,i)=>{const s=document.createElement('span');s.className='tag';s.textContent=`${i+1}. ${tag}`;$('tags').appendChild(s);});
  $('description').textContent=latest.description;
  listInto($('strengths'),a.strengths); listInto($('weaknesses'),a.weaknesses); listInto($('opportunities'),a.seo_opportunities); listInto($('changes'),o.changes_summary);
  $('volumeNote').textContent=o.search_volume_note||''; $('currentTitle').textContent=data.current_listing?.title||''; $('listingId').textContent=data.listing_id||'';
  $('keywords').innerHTML=''; (o.keyword_strategy||[]).forEach(item=>{const d=document.createElement('div');d.className='keyword';const b=document.createElement('b');b.textContent=item.keyword||'';const sp=document.createElement('span');sp.textContent=item.reason||'';d.appendChild(b);d.appendChild(sp);$('keywords').appendChild(d);});
  $('result').classList.remove('hidden');
}
$('form').addEventListener('submit',async e=>{e.preventDefault();$('run').disabled=true;$('status').className='status';$('status').textContent='Analyzing Etsy listing and marketplace keyword signals…';$('result').classList.add('hidden');try{const fd=new FormData();fd.append('listing_url',$('listing_url').value.trim());const r=await fetch('/analyze-existing-listing',{method:'POST',body:fd});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Analysis failed');if(data.validation_errors?.length){$('status').className='status warn';$('status').textContent='Analysis completed, but the generated result needs validation review.';}else{$('status').className='status success';$('status').textContent='Analysis complete — nothing was changed on Etsy.';}render(data);}catch(err){$('status').className='status error';$('status').textContent=err.message||'Something went wrong.';}finally{$('run').disabled=false;}});
document.querySelectorAll('[data-copy]').forEach(btn=>btn.addEventListener('click',async()=>{const key=btn.dataset.copy;const value=key==='tags'?latest.tags.join(', '):latest[key];try{await navigator.clipboard.writeText(value||'');const old=btn.textContent;btn.textContent='Copied ✓';setTimeout(()=>btn.textContent=old,1200);}catch(_){btn.textContent='Copy failed';}}));
</script>
</body>
</html>
""")

@app.post("/analyze-existing-listing")
async def analyze_existing_listing(
    listing_id: str = Form(""),
    listing_url: str = Form(""),
):
    """Analyze an existing Etsy listing and return an improved SEO title/tags.

    This endpoint only reads/analyzes the listing. It does NOT edit, create,
    activate, or publish anything on Etsy.
    """
    resolved_id = extract_listing_id(listing_id) or extract_listing_id(listing_url)

    if not resolved_id:
        raise HTTPException(
            status_code=400,
            detail="Enter an Etsy listing ID or a full Etsy listing URL.",
        )

    context = get_shop_context()
    access_token = context["access_token"]

    url = f"https://api.etsy.com/v3/application/listings/{resolved_id}"
    response = etsy_get(
        url,
        access_token,
        params={
            "includes": "Images",
            "language": "en",
            "allow_suggested_title": "true",
            "legacy": "false",
        },
    )

    if not response.ok:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    listing = response.json()

    # Safety check: only analyze the listing; do not allow an arbitrary public
    # listing to become a write target later in this endpoint.
    if str(listing.get("shop_id")) != str(context["shop_id"]):
        raise HTTPException(
            status_code=403,
            detail="For this optimizer, use a listing from your connected Etsy shop.",
        )

    queries = listing_market_queries(listing)
    try:
        market_signals = marketplace_keyword_signals(queries, per_query=15)
    except Exception as exc:
        market_signals = {
            "queries": queries,
            "marketplace_results_checked": 0,
            "high_signal_phrases": [],
            "high_signal_tags": [],
            "sample_listings": [],
            "research_warning": str(exc),
        }

    optimized = optimize_existing_listing(listing, market_signals)

    # Validate the proposed title/tags using the same hard constraints as the
    # listing generator, without touching Etsy.
    candidate = {
        "title": optimized.get("recommended_title", ""),
        "tags": optimized.get("recommended_tags", []),
        "description": optimized.get("recommended_description", ""),
    }
    validation_errors = validate_listing(candidate)

    return {
        "status": "success" if not validation_errors else "validation_failed",
        "message": (
            "Existing Etsy listing analyzed. Nothing was edited, created, or published."
        ),
        "listing_id": resolved_id,
        "current_listing": {
            "title": listing.get("title", ""),
            "tags": listing.get("tags", []) or [],
            "description": listing.get("description", ""),
            "materials": listing.get("materials", []) or [],
            "taxonomy_id": listing.get("taxonomy_id"),
            "num_favorers": listing.get("num_favorers"),
            "url": listing.get("url", ""),
            "etsy_suggested_title": listing.get("suggested_title"),
        },
        "market_research": market_signals,
        "optimized_result": optimized,
        "validation_errors": validation_errors,
        "write_action_performed": False,
    }


# -------------------------------------------------------------------
# CREATE ETSY DRAFT
# -------------------------------------------------------------------

@app.post("/create-draft-listing")
async def create_draft_listing(
    product: str = Form(...),
    details: str = Form(""),
    seller_claims: str = Form(""),
    price: float = Form(...),
    quantity: int = Form(1),
    who_made: str = Form("i_did"),
    when_made: str = Form("made_to_order"),
    taxonomy_id: str = Form(""),
    image: UploadFile = File(None),
):
    started_at = time.monotonic()

    def time_guard(stage):
        if time.monotonic() - started_at > 90:
            raise HTTPException(
                status_code=504,
                detail=f"Draft workflow timed out during {stage}. No publish action was performed."
            )

    if price <= 0:
        raise HTTPException(
            status_code=400,
            detail="Price must be greater than 0.",
        )

    if quantity <= 0:
        raise HTTPException(
            status_code=400,
            detail="Quantity must be greater than 0.",
        )

    if who_made not in {"i_did", "someone_else", "collective"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid who_made value.",
        )

    valid_when_made = {
        "made_to_order",
        "2020_2026",
        "2010_2019",
        "2007_2009",
        "before_2007",
        "2000_2006",
        "1990s",
        "1980s",
        "1970s",
        "1960s",
        "1950s",
        "1940s",
        "1930s",
        "1920s",
        "1910s",
        "1900s",
        "1800s",
        "1700s",
        "before_1700",
    }

    if when_made not in valid_when_made:
        raise HTTPException(
            status_code=400,
            detail="Invalid when_made value.",
        )

    context = get_shop_context()
    time_guard("shop configuration")

    # ---------------------------------------------------------------
    # ONE-TIME DEFAULT PROFILES
    # ---------------------------------------------------------------

    shipping_profile = choose_shipping_profile(
        context["shipping"]
    )

    processing_profile = choose_processing_profile(
        context["processing"]
    )

    if not shipping_profile:
        raise HTTPException(
            status_code=500,
            detail="No Etsy shipping profile was found.",
        )

    if not processing_profile:
        raise HTTPException(
            status_code=500,
            detail="No Etsy processing profile was found.",
        )

    # ---------------------------------------------------------------
    # TAXONOMY
    # ---------------------------------------------------------------

    selected_taxonomy = None
    taxonomy_candidates = []

    if taxonomy_id.strip():
        selected_taxonomy = taxonomy_id.strip()
    else:
        selected_taxonomy, taxonomy_candidates = choose_taxonomy_id(
            context["taxonomy"],
            product,
        )

    if not selected_taxonomy:
        return {
            "status": "taxonomy_selection_required",
            "message": (
                "AI could not confidently select an Etsy taxonomy. "
                "Choose the correct taxonomy_id and call this endpoint again."
            ),
            "taxonomy_candidates": taxonomy_candidates,
        }

    # ---------------------------------------------------------------
    # MARKET KEYWORD RESEARCH + AI LISTING
    # ---------------------------------------------------------------

    keyword_research = market_keyword_research(
        product + " " + details
    )

    listing = generate_listing(
        product,
        details,
        seller_claims,
        keyword_research.get("keyword_signals", []),
    )
    time_guard("AI listing generation")

    # ---------------------------------------------------------------
    # VALIDATE + REWRITE IF NEEDED
    # ---------------------------------------------------------------

    rewrite_count = 0
    originality = None
    claim_problems = []

    own_listings = get_own_active_listings(
        context["access_token"],
        context["shop_id"],
    )
    time_guard("own-shop originality lookup")

    competitors = competitor_candidates(listing)
    time_guard("marketplace originality lookup")

    for attempt in range(2):
        errors = validate_listing(listing)

        claim_problems = has_unsupported_claims(
            listing,
            details,
            seller_claims,
        )

        originality = originality_report(
            listing,
            own_listings,
            competitors,
        )

        if not errors and not claim_problems and originality["passed"]:
            break

        if attempt == 1:
            return {
                "status": "blocked_before_etsy",
                "message": (
                    "The listing did not pass validation/originality checks. "
                    "No Etsy draft was created."
                ),
                "listing": listing,
                "validation_errors": errors,
                "unsupported_claims": claim_problems,
                "originality": originality,
                "rewrite_attempts": rewrite_count,
            }

        listing = rewrite_listing(
            listing,
            originality.get("matches", []),
            claim_problems,
        )

        rewrite_count += 1

    # Final hard validation.
    errors = validate_listing(listing)

    if errors:
        return {
            "status": "blocked_before_etsy",
            "message": "Final listing validation failed. No draft created.",
            "listing": listing,
            "validation_errors": errors,
        }

    time_guard("pre-draft validation")

    # ---------------------------------------------------------------
    # CREATE DRAFT
    # ---------------------------------------------------------------

    create_url = (
        f"https://api.etsy.com/v3/application/shops/"
        f"{context['shop_id']}/listings?legacy=false"
    )

    form_data = {
        "quantity": str(quantity),
        "title": listing["title"],
        "description": listing["description"],
        "price": str(price),
        "who_made": who_made,
        "when_made": when_made,
        "taxonomy_id": str(selected_taxonomy),
        "shipping_profile_id": str(
            shipping_profile.get("shipping_profile_id")
        ),
        "readiness_state_id": str(
            processing_profile.get("readiness_state_id")
            or processing_profile.get("id")
        ),
        "is_supply": "false",
        "type": "physical",
        "should_auto_renew": "true",
    }

    # Etsy's createDraftListing accepts tags as repeated form values.
    # requests handles this by passing a list of tuples.
    form_items = list(form_data.items())

    for tag in listing["tags"]:
        form_items.append(("tags", tag))

    for material in listing.get("materials", []):
        if material:
            form_items.append(("materials", str(material)))

    draft_response = requests.post(
        create_url,
        headers=etsy_form_headers(
            context["access_token"]
        ),
        data=form_items,
        timeout=60,
    )

    if not draft_response.ok:
        raise HTTPException(
            status_code=draft_response.status_code,
            detail={
                "message": "Etsy draft creation failed.",
                "etsy_response": draft_response.text,
            },
        )

    draft = draft_response.json()

    listing_id = draft.get("listing_id")

    image_upload = None

    # ---------------------------------------------------------------
    # OPTIONAL IMAGE UPLOAD
    # ---------------------------------------------------------------

    if image is not None and listing_id:
        image_bytes = await image.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The Etsy draft was created, but the uploaded image "
                    "was over 10 MB and was not uploaded."
                ),
            )

        image_upload_url = (
            f"https://api.etsy.com/v3/application/shops/"
            f"{context['shop_id']}/listings/{listing_id}/images"
        )

        image_response = requests.post(
            image_upload_url,
            headers={
                "x-api-key": (
                    f"{ETSY_KEYSTRING}:{ETSY_SHARED_SECRET}"
                ),
                "Authorization": (
                    f"Bearer {context['access_token']}"
                ),
            },
            files={
                "image": (
                    image.filename or "listing.jpg",
                    image_bytes,
                    image.content_type or "image/jpeg",
                )
            },
            timeout=15,
        )

        image_upload = {
            "uploaded": image_response.ok,
            "status_code": image_response.status_code,
            "response": (
                image_response.json()
                if image_response.ok
                else image_response.text
            ),
        }

    # IMPORTANT:
    # There is deliberately NO updateListing(state=active) call here.
    # The agent creates a draft and stops.

    return {
        "status": "draft_created",
        "message": (
            "Etsy draft listing created successfully. "
            "It was NOT published."
        ),
        "listing": listing,
        "keyword_research": keyword_research,
        "etsy_draft": draft,
        "listing_id": listing_id,
        "shipping_profile": {
            "id": shipping_profile.get("shipping_profile_id"),
            "title": shipping_profile.get("title"),
        },
        "processing_profile": {
            "id": (
                processing_profile.get("readiness_state_id")
                or processing_profile.get("id")
            ),
            "title": processing_profile.get("title"),
        },
        "taxonomy_id": selected_taxonomy,
        "originality": originality,
        "rewrite_attempts": rewrite_count,
        "image_upload": image_upload,
        "published": False,
    }
