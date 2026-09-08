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
from fastapi.responses import RedirectResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ETSY_KEYSTRING = os.getenv("ETSY_API_KEYSTRING")
ETSY_SHARED_SECRET = os.getenv("ETSY_SHARED_SECRET")
ETSY_REDIRECT_URI = "https://etsy-ai-agent.onrender.com/etsy/callback"

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL environment variable is missing.")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

TOKEN_KEY = "etsy:oauth_token"
SETTINGS_KEY = "etsy:listing_settings"

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
        "mode": "draft_only",
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
        timeout=30,
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
        timeout=30,
    )

    if access_token and response.status_code == 401:
        refreshed = refresh_etsy_token()
        if refreshed:
            new_access_token = refreshed.get("access_token")
            response = requests.get(
                url,
                headers=etsy_headers(new_access_token),
                params=params,
                timeout=30,
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
        raise HTTPException(
            status_code=401,
            detail="Etsy account is not connected.",
        )

    access_token = token_data.get("access_token")
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
        raise HTTPException(
            status_code=shop_response.status_code,
            detail=shop_response.text,
        )

    shop_data = shop_response.json()
    shop = shop_data.get("shop", shop_data)
    shop_id = shop.get("shop_id")

    processing_url = (
        f"https://api.etsy.com/v3/application/shops/"
        f"{shop_id}/readiness-state-definitions"
    )

    processing_response = etsy_get(
        processing_url,
        access_token,
    )

    processing_data = (
        processing_response.json()
        if processing_response.ok
        else {}
    )

    shipping_url = (
        f"https://api.etsy.com/v3/application/shops/"
        f"{shop_id}/shipping-profiles"
    )

    shipping_response = etsy_get(
        shipping_url,
        access_token,
    )

    shipping_data = (
        shipping_response.json()
        if shipping_response.ok
        else {}
    )

    taxonomy_url = (
        "https://api.etsy.com/v3/application/seller-taxonomy/nodes"
    )

    taxonomy_response = etsy_get(
        taxonomy_url,
        access_token,
    )

    taxonomy_data = (
        taxonomy_response.json()
        if taxonomy_response.ok
        else {}
    )

    return {
        "access_token": access_token,
        "user_id": user_id,
        "shop": shop,
        "shop_id": shop_id,
        "processing": processing_data,
        "shipping": shipping_data,
        "taxonomy": taxonomy_data,
    }


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

        score = overlap + jewelry_bonus + leaf_bonus

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


def generate_listing(product, details, seller_claims=""):
    prompt = f"""
You are an expert Etsy SEO listing agent for handmade gemstone jewelry.

Create ONE original Etsy listing from the seller information below.

PRODUCT:
{product}

SELLER DETAILS:
{details}

SELLER-CONFIRMED CLAIMS:
{seller_claims}

IMPORTANT FACT RULES:
- Only use gemstone identity, metal type, purity, natural/genuine status,
  origin, treatment, certification, dimensions, carat weight and measurements
  when the seller supplied that fact.
- Never infer "natural", "genuine", "authentic", "solid gold", "925",
  certification, origin or treatment from a photograph.
- Do not invent measurements or carat weight.
- Do not copy another seller's wording.
- Use normal buyer-friendly Etsy SEO.
- Title must be <= 140 characters.
- Produce exactly 13 tags.
- Every tag must be <= 20 characters.
- Avoid keyword stuffing.
- Description must be original and readable.

Return ONLY valid JSON with this exact structure:

{{
  "title": "string",
  "tags": ["13 tags"],
  "description": "string",
  "materials": ["seller-confirmed materials only"],
  "gift_keywords": ["short phrases"],
  "observed_facts": ["facts actually supplied or observable"],
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


def get_own_active_listings(access_token, shop_id, limit=100):
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
            "limit": min(limit, 100),
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

    for query in queries[:2]:
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

    return list(unique.values())[:30]


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

    listing = generate_listing(
        product_context,
        extra_info,
        extra_info,
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

    for attempt in range(4):
        validation_errors = validate_listing(listing)

        claim_problems = has_unsupported_claims(
            listing,
            extra_info,
            extra_info,
        )

        competitors = competitor_candidates(listing)

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

        if attempt == 3:
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
    # AI LISTING
    # ---------------------------------------------------------------

    listing = generate_listing(
        product,
        details,
        seller_claims,
    )

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

    competitors = competitor_candidates(listing)

    for attempt in range(4):
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

        if attempt == 3:
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
            timeout=60,
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
