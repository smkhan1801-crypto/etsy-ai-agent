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
from collections import Counter, defaultdict

import requests
import redis

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Always return JSON so the optimizer UI can show the real backend error
    # instead of failing with "Unexpected token I" while parsing "Internal Server Error".
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": f"{type(exc).__name__}: {str(exc) or 'Unknown server error'}",
            "path": str(request.url.path),
        },
    )

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash,gemini-3.1-flash-lite"
    ).split(",") if m.strip()
]

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Google AI Studio API key "
            "to Render Environment Variables and redeploy."
        )
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

class GeminiTextResponse:
    def __init__(self, text):
        self.output_text = text or ""

def _gemini_models_to_try():
    # Try the configured model first, then free-tier Flash fallbacks.
    seen = set()
    models = []
    for model in [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models

def _is_gemini_transient_error(exc):
    msg = str(exc).lower()
    return any(token in msg for token in [
        "503", "unavailable", "high demand", "429",
        "resource exhausted", "500", "502", "504", "temporarily"
    ])

def _gemini_generate_with_fallback(contents, config, purpose="text"):
    last_exc = None
    models = _gemini_models_to_try()
    for model_index, model in enumerate(models):
        # One short retry for a transient failure, then move to the next model.
        for attempt in range(2):
            try:
                response = get_gemini_client().models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                text = getattr(response, "text", None) or ""
                if not text:
                    raise RuntimeError(f"Gemini returned an empty {purpose} response.")
                return GeminiTextResponse(text)
            except Exception as exc:
                last_exc = exc
                if not _is_gemini_transient_error(exc):
                    raise
                # Small exponential backoff for 503/429/5xx.
                if attempt == 0:
                    time.sleep(1.5)
        # If the configured model is busy, continue to the next free-tier model.

    raise RuntimeError(
        "Gemini is temporarily unavailable on all configured models. "
        f"Tried: {', '.join(models)}. Last error: {type(last_exc).__name__}: {last_exc}"
    )

def gemini_generate_text(prompt, max_output_tokens=3000, json_mode=False):
    config_kwargs = {"max_output_tokens": max_output_tokens}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    return _gemini_generate_with_fallback(
        prompt,
        types.GenerateContentConfig(**config_kwargs),
        purpose="text",
    )

def gemini_generate_image_text(prompt, image_bytes, mime_type, max_output_tokens=1200):
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return _gemini_generate_with_fallback(
        [prompt, image_part],
        types.GenerateContentConfig(max_output_tokens=max_output_tokens),
        purpose="image-analysis",
    )

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
        "ai_backend": "Google Gemini Free Tier",
        "ai_model": GEMINI_MODEL,
        "ai_fallback_models": GEMINI_FALLBACK_MODELS,
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



def etsy_patch_form(url, access_token, data):
    response = requests.patch(url, headers=etsy_form_headers(access_token), data=data, timeout=30)
    if response.status_code == 401:
        refreshed = refresh_etsy_token()
        if refreshed:
            response = requests.patch(url, headers=etsy_form_headers(refreshed.get("access_token")), data=data, timeout=30)
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
    """Normalize common Gemini JSON wrappers without changing the payload."""
    if text is None:
        return ""
    text = str(text).replace("\ufeff", "").strip()

    # Remove Markdown fences if Gemini ignored response_mime_type.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # Remove a leading JSON label sometimes emitted by a model.
    text = re.sub(r"^\s*(?:json|JSON)\s*:\s*", "", text).strip()
    return text


def _extract_balanced_json_object(text):
    """Extract the first complete JSON object/array while respecting quoted strings."""
    if not text:
        return None

    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start_i in starts:
        opener = text[start_i]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False

        for i in range(start_i, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start_i:i + 1]
            elif opener == "{" and ch == "]":
                break
            elif opener == "[" and ch == "}":
                break

    return None


def parse_listing_json(text):
    """Parse Gemini output into the optimizer's required JSON object."""
    cleaned = clean_json_text(text)
    if not cleaned:
        raise ValueError("AI returned an empty JSON response.")

    def as_object(data):
        if isinstance(data, dict):
            return data
        # Gemini sometimes emits [{"...": ...}] even in JSON mode.
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        return None

    try:
        data = json.loads(cleaned)
        obj = as_object(data)
        if obj is not None:
            return obj
        shape_error = f"unsupported JSON shape: {type(data).__name__}"
    except json.JSONDecodeError as exc:
        shape_error = f"invalid JSON syntax: {exc}"

    candidate = _extract_balanced_json_object(cleaned)
    if candidate:
        try:
            data = json.loads(candidate)
            obj = as_object(data)
            if obj is not None:
                return obj
        except json.JSONDecodeError:
            pass

        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            data = json.loads(repaired)
            obj = as_object(data)
            if obj is not None:
                return obj
        except json.JSONDecodeError:
            pass

    preview = cleaned[:700].replace("\n", " ")
    raise ValueError(
        "AI listing result is not a JSON object. "
        f"{shape_error}. Response preview: {preview}"
    )

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

    response = gemini_generate_text(
        prompt,
        max_output_tokens=3000,
        json_mode=True,
    )

    return parse_listing_json(response.output_text)


# -------------------------------------------------------------------
# GEMINI REQUEST HELPERS
# -------------------------------------------------------------------

def ai_response_create(*, input_data, max_output_tokens=3000):
    """Generate optimizer output with Gemini Free Tier.

    Gemini is requested to return application/json. If a provider/model still
    returns malformed JSON, make one constrained repair call instead of crashing
    the whole optimizer.
    """
    response = gemini_generate_text(
        input_data,
        max_output_tokens=max_output_tokens,
        json_mode=True,
    )

    try:
        parse_listing_json(response.output_text)
        return response
    except ValueError:
        raw = response.output_text or ""
        repair_prompt = f"""
Convert the following model output into ONE valid JSON OBJECT matching the requested listing schema.
Return ONLY one JSON object, never an array, scalar, Markdown fence, or commentary.
If the input is a one-item array containing an object, unwrap that object.
Do not add, remove, or invent product facts.
Preserve the existing values exactly where possible.

MODEL OUTPUT:
{raw}
"""
        repaired = gemini_generate_text(
            repair_prompt,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        # Fail with a useful error if even the constrained repair is invalid.
        parse_listing_json(repaired.output_text)
        return repaired


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



def seo_score_report(current_listing, optimized_result, market_signals, validation_errors=None, identity=None):
    """V6 evidence-aware deterministic SEO quality score.

    The score measures the quality of the optimizer output against observable
    Etsy/listing evidence. It is not Etsy's ranking score.

    Important V6 rule:
    A keyword is scoreable only when it is supported by source product facts
    or by the optimizer's marketplace-signal set. Missing/unavailable evidence
    is not turned into a fake failure merely to lower or raise the score.
    """
    validation_errors = validation_errors or []
    identity = identity or extract_product_identity(current_listing)

    title = str(optimized_result.get("recommended_title", "") or "").strip()
    tags = [str(t).strip() for t in (optimized_result.get("recommended_tags", []) or []) if str(t).strip()]
    description = str(optimized_result.get("recommended_description", "") or "").strip()
    ki = optimized_result.get("keyword_intelligence", {}) or {}

    tw = words(title)
    tn = normalize_text(title)
    tag_norm = [normalize_text(t) for t in tags]
    dn = normalize_text(description)
    alln = " ".join([tn] + tag_norm + [dn])

    product_type = identity.get("product_type", "") or ""
    gems = identity.get("gemstones", []) or []
    primary_gem = gems[0] if gems else ""
    protected = [normalize_text(x) for x in identity.get("protected_terms", []) if x]

    def has_term(text_norm, term):
        term = normalize_text(term or "")
        return bool(term and (re.search(r"\b" + re.escape(term) + r"\b", text_norm) or term in text_norm))

    # ---------- Product identity 15 ----------
    ip, ir = 0, []
    if not product_type or has_term(tn, product_type):
        ip += 5
    else:
        ir.append("Target product type is missing from the title.")
    if not product_type or has_term(dn, product_type):
        ip += 3
    else:
        ir.append("Target product type is missing from the description.")
    if not primary_gem or has_term(tn, primary_gem):
        ip += 4
    else:
        ir.append("Primary gemstone is missing from the title.")
    if not primary_gem or has_term(dn, primary_gem):
        ip += 3
    else:
        ir.append("Primary gemstone is missing from the description.")

    # ---------- Title 15 ----------
    tp, tr = 0, []
    if title:
        tp += 3
    else:
        tr.append("Title is empty.")

    if 8 <= len(tw) <= 15:
        tp += 3
    elif 6 <= len(tw) <= 17:
        tp += 1
        tr.append(f"Title has {len(tw)} words; target 8–15.")
    else:
        tr.append(f"Title has {len(tw)} words; target 8–15.")

    if len(title) <= 140:
        tp += 2
    else:
        tr.append("Title exceeds 140 characters.")

    if tw and len(tw) == len(set(tw)):
        tp += 3
    else:
        tr.append("Title repeats one or more words.")

    subjective = {"best","perfect","beautiful","unique","amazing","stunning","gorgeous","must","premium","luxury"}
    if not subjective.intersection(set(tw)):
        tp += 2
    else:
        tr.append("Remove subjective sales language from the title.")

    if not product_type or has_term(tn[:max(1, min(len(tn), 90))], product_type):
        tp += 2
    else:
        tr.append("Place the actual product type near the beginning.")

    # ---------- Tags 20 ----------
    gp, gr = 0, []
    if len(tags) == 13:
        gp += 4
    else:
        gr.append(f"Use exactly 13 tags; current count is {len(tags)}.")

    if tags:
        valid_len = sum(len(t) <= 20 for t in tags)
        gp += round(4 * valid_len / len(tags))
        if valid_len < len(tags):
            gr.append("Every tag must be 20 characters or fewer.")

    unique_count = len(set(tag_norm))
    unique_ratio = unique_count / max(1, len(tag_norm))
    if unique_ratio == 1:
        gp += 4
    else:
        gr.append("Remove duplicate tags.")

    # Evidence-aware relevance: source facts/listing language OR marketplace
    # signals. This avoids the overly narrow "protected terms only" rule.
    signal_phrases_for_tags = set()
    def _collect_tag_signals(value):
        if isinstance(value, str):
            s = normalize_text(value)
            if 1 <= len(s.split()) <= 8 and len(s) <= 80:
                signal_phrases_for_tags.add(s)
        elif isinstance(value, list):
            for item in value:
                _collect_tag_signals(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                if str(k).lower() not in {"score", "count", "rank", "id"}:
                    _collect_tag_signals(v)
    _collect_tag_signals(market_signals or {})

    source_terms_for_tags = set(protected)
    for value in [
        product_type, primary_gem,
        current_listing.get("title", ""),
        current_listing.get("description", ""),
        *(current_listing.get("tags", []) or []),
        *(current_listing.get("materials", []) or []),
        *(current_listing.get("style", []) or []),
    ]:
        s = normalize_text(str(value))
        if s:
            source_terms_for_tags.add(s)

    identity_words_for_tags = set(re.findall(
        r"[a-z0-9]+",
        normalize_text(" ".join([product_type, primary_gem] + gems))
    ))

    def _tag_is_supported(tag):
        tag = normalize_text(tag)
        if not tag:
            return False
        # Exact/subphrase evidence from the source listing is strongest.
        if any(term and (tag in term or term in tag) for term in source_terms_for_tags):
            return True
        if tag in signal_phrases_for_tags:
            return True

        tag_words = set(re.findall(r"[a-z0-9]+", tag))
        if not tag_words:
            return False

        # A tag is also supported when every meaningful word is evidenced by the
        # source listing (title, description, existing tags, materials/style),
        # and at least one word is a product/gemstone/material identity word.
        source_word_pool = set()
        for source_value in [
            current_listing.get("title", ""),
            current_listing.get("description", ""),
            *(current_listing.get("tags", []) or []),
            *(current_listing.get("materials", []) or []),
            *(current_listing.get("style", []) or []),
        ]:
            source_word_pool.update(re.findall(r"[a-z0-9]+", normalize_text(str(source_value))))
        generic_words = {
            "natural", "genuine", "handmade", "jewelry", "jewellery", "gift",
            "for", "her", "women", "woman", "men", "mens", "dangle",
            "dangles", "drop", "earrings", "earring", "chain", "gold",
        }
        meaningful = {w for w in tag_words if w not in generic_words and len(w) > 2}
        if meaningful and meaningful.issubset(source_word_pool) and (tag_words & identity_words_for_tags):
            return True

        # Finally, accept a marketplace phrase only when it shares identity
        # language with this listing. This keeps marketplace evidence directional
        # without allowing unrelated competitor products into the tag set.
        return any(
            sig and len(tag_words & set(re.findall(r"[a-z0-9]+", sig))) >= 1
            and (tag_words & identity_words_for_tags)
            for sig in signal_phrases_for_tags
        )

    supported_flags = {t: _tag_is_supported(t) for t in tag_norm}
    relevant = sum(1 for ok in supported_flags.values() if ok)
    if tags:
        gp += round(4 * relevant / len(tags))
    unsupported_tags = [t for t, ok in supported_flags.items() if not ok]
    if unsupported_tags:
        gr.append("Replace unsupported/weak tags: " + ", ".join(unsupported_tags[:5]) + ".")
    if tags and relevant < max(7, round(len(tags) * 0.55)):
        gr.append("More tags should be directly supported by source facts or marketplace signals.")

    starts = [t.split()[0] for t in tag_norm if t.split()]
    distinct_starts = len(set(starts))
    if len(tags) == 13 and unique_ratio == 1 and distinct_starts >= 8:
        gp += 4
    else:
        gp += round(4 * min(1, distinct_starts / max(1, len(tags))))
        if distinct_starts < min(8, len(tags)):
            gr.append("Use a broader mix of search-phrase structures.")
    gp = min(20, gp)

    # ---------- V6 keyword evidence ----------
    # Gather marketplace phrases from whatever structure the existing research
    # helper returned. These are directional signals, not exact search volume.
    signal_phrases = set()

    def collect_strings(value):
        if isinstance(value, str):
            s = normalize_text(value)
            if 2 <= len(s.split()) <= 8 and len(s) <= 80:
                signal_phrases.add(s)
        elif isinstance(value, list):
            for item in value:
                collect_strings(item)
        elif isinstance(value, dict):
            for key, value2 in value.items():
                if key.lower() not in {"score", "count", "rank", "id"}:
                    collect_strings(value2)

    collect_strings(market_signals or {})

    # Source-supported terms are always eligible. Marketplace phrases are
    # eligible only if they also overlap the source identity or current listing.
    source_evidence = set(protected)
    for value in [
        product_type, primary_gem,
        current_listing.get("title", ""),
        current_listing.get("description", ""),
        *(current_listing.get("tags", []) or []),
        *(current_listing.get("materials", []) or []),
        *(current_listing.get("style", []) or []),
    ]:
        s = normalize_text(str(value))
        if s:
            source_evidence.add(s)

    def keyword_supported(k):
        k = normalize_text(k)
        if not k:
            return False
        if any(k in e or e in k for e in source_evidence if e):
            return True
        # Marketplace phrase must share meaningful identity words.
        kwords = set(re.findall(r"[a-z0-9]+", k))
        identity_words = set(re.findall(
            r"[a-z0-9]+",
            normalize_text(" ".join([product_type, primary_gem] + gems))
        ))
        return len(kwords & identity_words) >= 1 and k in signal_phrases

    # Only score the first few AI suggestions in each group. Empty/unavailable
    # groups do not create fabricated "missing keyword" penalties.
    keyword_groups = [
        ("primary_keywords", 6, 5),
        ("secondary_keywords", 4, 5),
        ("long_tail_keywords", 3, 3),
        ("buyer_intent_keywords", 2, 2),
    ]
    kp = 0
    kr = []
    for key, pts, take in keyword_groups:
        vals = list(dict.fromkeys(
            normalize_text(x) for x in (ki.get(key, []) or []) if str(x).strip()
        ))
        eligible = [k for k in vals[:take] if keyword_supported(k)]
        if not vals:
            # No source/model evidence to score for this group.
            continue
        if not eligible:
            # The AI supplied phrases, but none are evidence-supported. This is
            # a real quality problem, so it gets no credit for that group.
            kr.append(f"{key.replace('_',' ')} are not sufficiently supported by listing evidence.")
            continue

        missing = [k for k in eligible if k not in alln]
        covered = len(eligible) - len(missing)
        kp += round(pts * covered / len(eligible))
        if missing:
            kr.append(
                f"Missing supported {key.replace('_',' ')}: "
                + ", ".join(missing[:5])
            )

    # Direct identity keyword coverage gets explicit credit even when the model's
    # keyword list is sparse.
    direct_identity_terms = [x for x in [product_type, primary_gem] + gems if x]
    direct_hits = sum(1 for x in direct_identity_terms if has_term(alln, x))
    if direct_identity_terms:
        kp += min(2, direct_hits)
    kp = min(15, kp)

    # ---------- Description 15 ----------
    dp, dr = 0, []
    if description:
        dp += 3
    else:
        dr.append("Description is empty.")

    first = re.split(r"(?<=[.!?])\s+", description, maxsplit=1)[0] if description else ""
    fn = normalize_text(first)

    if not product_type or has_term(fn, product_type):
        dp += 3
    else:
        dr.append("First sentence should clearly identify the product.")

    if not primary_gem or has_term(fn, primary_gem):
        dp += 3
    else:
        dr.append("First sentence should include the primary gemstone.")

    if len(description) >= 250:
        dp += 2
    elif len(description) >= 120:
        dp += 1
        dr.append("Description could include more supported buyer information.")
    else:
        dr.append("Description is too short.")

    detail_terms = ["material","size","stone","gemstone","design","wear","care","shipping","gift"]
    detail_hits = sum(1 for x in detail_terms if re.search(r"\b" + re.escape(x) + r"\b", dn))
    dp += min(2, detail_hits // 2)

    if not subjective.intersection(set(words(description))):
        dp += 1
    else:
        dr.append("Avoid excessive subjective sales language.")

    if len(description.split()) >= 80:
        dp += 3
    else:
        dr.append("Add useful buyer information where the source listing supports it.")
    dp = min(15, dp)

    # ---------- Category / attributes 10 ----------
    attrs = current_listing.get("attributes", []) or []
    taxonomy_id = current_listing.get("taxonomy_id")
    taxonomy_properties = current_listing.get("taxonomy_properties", []) or []

    ap_earned = 0
    excluded_attribute_points = 0
    apr = []

    if taxonomy_id:
        ap_earned += 4
    else:
        apr.append("Category/taxonomy is not confirmed.")

    if isinstance(attrs, list) and attrs:
        if len(attrs) >= 4:
            ap_earned += 4
        elif len(attrs) >= 2:
            ap_earned += 2
            apr.append("Review additional relevant Etsy attributes.")
        else:
            apr.append("Only limited source attributes are available.")
    else:
        excluded_attribute_points += 4
        apr.append("Selected Etsy attribute values were not exposed in the listing payload; not scored as a failure.")

    if taxonomy_properties:
        if any(isinstance(p, dict) and p.get("supports_attributes") for p in taxonomy_properties):
            ap_earned += 2
        else:
            excluded_attribute_points += 2
            apr.append("Taxonomy schema does not expose attribute-capable properties for this category.")
    else:
        excluded_attribute_points += 2
        apr.append("Taxonomy property schema could not be read; not scored as a failure.")

    ap_max_applicable = max(1, 10 - excluded_attribute_points)
    ap = min(ap_max_applicable, ap_earned)

    # ---------- Buyer / conversion 10 ----------
    bp, br = 0, []

    if len(description) >= 250:
        bp += 3
    elif len(description) >= 120:
        bp += 2
    else:
        br.append("Description needs more useful buyer information.")

    if re.search(r"\b(size|dimension|length|mm|inch|inches)\b", dn):
        bp += 2
    else:
        br.append("Add known size/dimensions if the source listing provides them.")

    if re.search(r"\b(material|sterling|gold|silver|vermeil|gemstone|opal|spinel)\b", dn):
        bp += 2
    else:
        br.append("Make supported material/gemstone information easy to find.")

    # Gift language is optional; award this point when useful buyer context exists.
    if re.search(r"\b(gift|birthday|anniversary|wedding|holiday|present|everyday|occasion|wear)\b", dn):
        bp += 1
    else:
        bp += 1

    if not re.search(r"\b(heal|healing|cure|treat|medical)\b", alln):
        bp += 1
    else:
        br.append("Remove medical/healing claims.")

    if product_type and primary_gem and has_term(fn, product_type) and has_term(fn, primary_gem):
        bp += 1
    else:
        br.append("Keep the opening sentence specific about the product and gemstone.")
    bp = min(10, bp)

    sections = [
        ("Product identity", ip, 15, ir),
        ("Title", tp, 15, tr),
        ("13 tags", gp, 20, gr),
        ("Keyword coverage", kp, 15, kr),
        ("Description", dp, 15, dr),
        ("Category & attributes", ap, ap_max_applicable, apr),
        ("Buyer/conversion quality", bp, 10, br),
    ]

    raw_earned = sum(s for _, s, _, _ in sections)
    raw_max = sum(m for _, _, m, _ in sections)
    score = round((raw_earned / raw_max) * 100) if raw_max else 0
    score = max(0, min(100, score))

    gaps = [
        {
            "section": name,
            "score": earned,
            "max": maximum,
            "points_lost": maximum - earned,
            "reasons": reasons[:5],
        }
        for name, earned, maximum, reasons in sections
        if earned < maximum
    ]

    return {
        "current_score": None,
        "optimized_score": score,
        "raw_points": raw_earned,
        "raw_max": raw_max,
        "applicable_points": raw_max,
        "excluded_points": 10 - ap_max_applicable,
        "grade": (
            "A+" if score >= 95 else
            "A" if score >= 90 else
            "B" if score >= 80 else
            "C" if score >= 70 else
            "D" if score >= 60 else "E"
        ),
        "is_genuine_100": score == 100 and not validation_errors,
        "hard_validation_errors": validation_errors[:10],
        "breakdown": {
            "product_identity": {"optimized": ip, "max": 15},
            "title": {"optimized": tp, "max": 15},
            "tags": {"optimized": gp, "max": 20},
            "keyword_coverage": {"optimized": kp, "max": 15},
            "description": {"optimized": dp, "max": 15},
            "attributes": {
                "optimized": ap,
                "max": ap_max_applicable,
                "original_max": 10,
                "excluded": 10 - ap_max_applicable,
            },
            "buyer_quality": {"optimized": bp, "max": 10},
        },
        "gaps": gaps,
        "note": (
            "Genuine internal SEO-quality score based on deterministic checks. "
            "It is not Etsy's ranking score. V11 scores keyword coverage only "
            "against source/listing evidence or marketplace signals and never "
            "treats unavailable Etsy metadata as a failure."
        ),
    }

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

    response = gemini_generate_text(
        prompt,
        max_output_tokens=3000,
        json_mode=True,
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

    analysis_response = gemini_generate_image_text(
        analysis_prompt,
        image_bytes,
        image.content_type,
        max_output_tokens=1200,
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


def extract_product_identity(listing):
    """Build a deterministic identity lock from the target listing.

    This is the source-of-truth layer for the optimizer. Marketplace competitors
    may influence keyword language, but they are never allowed to redefine what
    the target product is.
    """
    title = normalize_text(listing.get("title", ""))
    tags = [normalize_text(x) for x in (listing.get("tags", []) or []) if str(x).strip()]
    description = normalize_text(listing.get("description", ""))
    materials = [normalize_text(x) for x in (listing.get("materials", []) or []) if str(x).strip()]
    source_text = " ".join([title] + tags + [description] + materials).lower()

    product_types = [
        ("necklace", ["necklace", "necklaces", "choker"]),
        ("bracelet", ["bracelet", "bracelets", "bangle"]),
        ("ring", ["ring", "rings"]),
        ("earring", ["earring", "earrings", "stud", "hoop earrings"]),
        ("pendant", ["pendant", "pendants"]),
        ("anklet", ["anklet", "anklets"]),
    ]
    product_type = ""
    for canonical, variants in product_types:
        if any(re.search(r"\b" + re.escape(v) + r"\b", source_text) for v in variants):
            product_type = canonical
            break

    gemstones = [
        "black spinel", "fire opal", "white opal", "rainbow opal", "ethiopian opal",
        "blue sapphire", "yellow sapphire", "pink sapphire", "green sapphire",
        "blue topaz", "smoky quartz", "rose quartz", "moonstone", "tourmaline",
        "amethyst", "aquamarine", "garnet", "ruby", "emerald", "opal", "coral",
        "onyx", "peridot", "citrine", "topaz", "lapis lazuli", "turquoise",
        "peridot", "spinel", "sapphire", "diamond", "pearl",
    ]
    gemstone_terms = []
    for gem in gemstones:
        if re.search(r"\b" + re.escape(gem) + r"\b", source_text):
            gemstone_terms.append(gem)
    # Preserve the most specific gemstone phrases, but determine the primary
    # gemstone by the order it appears in the source title/description rather
    # than alphabetical/length sorting. This prevents a secondary stone such as
    # "black spinel" from accidentally becoming the primary stone simply because
    # it sorts ahead of "ethiopian opal".
    unique_gemstones = list(dict.fromkeys(gemstone_terms))
    def first_position(term):
        m = re.search(r"\b" + re.escape(term) + r"\b", title.lower())
        if m:
            return (0, m.start())
        m = re.search(r"\b" + re.escape(term) + r"\b", description.lower())
        if m:
            return (1, m.start())
        return (2, len(source_text))
    unique_gemstones.sort(key=lambda x: (first_position(x), -len(x.split()), -len(x)))
    filtered_gems = []
    for gem in unique_gemstones:
        if not any(gem != other and gem in other for other in filtered_gems):
            filtered_gems.append(gem)

    protected_terms = []
    for term in [product_type] + filtered_gems[:3]:
        if term:
            protected_terms.append(term)

    # Add explicit objective material/purity terms only when they are actually
    # present in the source listing.
    objective_patterns = [
        r"\b925 sterling silver\b", r"\bsterling silver\b", r"\b18k gold vermeil\b",
        r"\b14k gold vermeil\b", r"\b10k gold\b", r"\b14k gold\b", r"\b18k gold\b",
        r"\b24k gold\b", r"\bgold vermeil\b", r"\bgold plated\b",
    ]
    for pattern in objective_patterns:
        m = re.search(pattern, source_text)
        if m:
            term = m.group(0)
            if term not in protected_terms:
                protected_terms.append(term)

    return {
        "product_type": product_type,
        "gemstones": filtered_gems[:5],
        "protected_terms": protected_terms,
        "source_title": listing.get("title", ""),
        "source_tags": tags,
    }


def validate_product_identity(optimized, identity):
    """Reject an AI result that drifts into another jewelry product."""
    errors = []
    title = normalize_text(optimized.get("recommended_title", ""))
    tags = [normalize_text(x) for x in (optimized.get("recommended_tags", []) or [])]
    description = normalize_text(optimized.get("recommended_description", ""))
    all_text = " ".join([title] + tags + [description]).lower()

    product_type = identity.get("product_type", "")
    gemstones = identity.get("gemstones", []) or []

    if product_type:
        if not re.search(r"\b" + re.escape(product_type) + r"s?\b", title.lower()):
            errors.append(f"Product type drift: optimized title must contain the target product type '{product_type}'.")
        if not re.search(r"\b" + re.escape(product_type) + r"s?\b", description.lower()):
            errors.append(f"Product type drift: optimized description must describe the target product type '{product_type}'.")

        conflicting = {
            "necklace": ["ring", "bracelet", "earring", "pendant", "anklet"],
            "bracelet": ["ring", "necklace", "earring", "pendant", "anklet"],
            "ring": ["necklace", "bracelet", "earring", "pendant", "anklet"],
            "earring": ["ring", "necklace", "bracelet", "pendant", "anklet"],
            "pendant": ["ring", "necklace", "bracelet", "earring", "anklet"],
            "anklet": ["ring", "necklace", "bracelet", "earring", "pendant"],
        }.get(product_type, [])
        for conflict in conflicting:
            if re.search(r"\b" + re.escape(conflict) + r"s?\b", title.lower()):
                errors.append(f"Product type drift: title contains conflicting product type '{conflict}'.")
            # A conflicting type appearing repeatedly in the complete output is
            # also treated as drift, while a single incidental word is tolerated.
            count = len(re.findall(r"\b" + re.escape(conflict) + r"s?\b", all_text))
            if count >= 2:
                errors.append(f"Product type drift: output repeatedly uses conflicting product type '{conflict}'.")

    if gemstones:
        primary_gem = gemstones[0]

        # Multi-word gemstone names can be expressed with harmless intervening
        # words (for example, "Ethiopian natural opal"). Treat the gemstone as
        # preserved when all meaningful words are present, while still rejecting
        # a true substitution such as opal -> emerald.
        def gemstone_present(text, phrase):
            text_l = text.lower()
            if re.search(r"\b" + re.escape(phrase) + r"\b", text_l):
                return True
            words = [w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if len(w) > 2]
            return len(words) > 1 and all(re.search(r"\b" + re.escape(w) + r"\b", text_l) for w in words)

        if not gemstone_present(title, primary_gem):
            errors.append(f"Gemstone drift: optimized title must retain the target gemstone '{primary_gem}'.")
        if not gemstone_present(description, primary_gem):
            errors.append(f"Gemstone drift: optimized description must retain the target gemstone '{primary_gem}'.")

        # If the source clearly identifies a second gemstone (e.g. black spinel),
        # it is useful but not mandatory in the title. It must not be replaced by
        # an unrelated gemstone.
        source_gems = set(gemstones)
        unrelated_gems = [
            "coral", "ruby", "emerald", "sapphire", "opal", "spinel", "tourmaline",
            "topaz", "amethyst", "garnet", "pearl", "onyx", "moonstone", "turquoise",
        ]
        for other in unrelated_gems:
            # Allow generic gemstone words that are already part of a more
            # specific source gemstone phrase. For example, a listing whose
            # source contains "black spinel" and "fire opal" may legitimately
            # use both "spinel" and "opal". The old validator incorrectly
            # flagged those base words as unrelated gemstones.
            if other in source_gems or any(
                other == source_gem
                or re.search(r"\b" + re.escape(other) + r"\b", source_gem)
                or re.search(r"\b" + re.escape(source_gem) + r"\b", other)
                for source_gem in source_gems
            ):
                continue
            count = len(re.findall(r"\b" + re.escape(other) + r"\b", all_text))
            if count >= 2:
                errors.append(f"Gemstone drift: output repeatedly introduces unrelated gemstone '{other}'.")

    return list(dict.fromkeys(errors))


def repair_optimized_tags(optimized, listing, market_signals, identity):
    """Deterministically repair only unsupported/duplicate tags.

    AI remains responsible for the SEO strategy. This safety pass only replaces
    tags that cannot be grounded in the source listing or marketplace evidence,
    so the final 13-tag set does not lose a point merely because a single vague
    phrase slipped through generation.
    """
    raw_tags = [str(x).strip() for x in (optimized.get("recommended_tags", []) or []) if str(x).strip()]
    raw_tags = raw_tags[:13]
    source_texts = [
        listing.get("title", ""), listing.get("description", ""),
        *(listing.get("tags", []) or []), *(listing.get("materials", []) or []),
        *(listing.get("style", []) or []),
    ]
    source_pool = []
    seen_pool = set()
    for value in source_texts:
        txt = normalize_text(str(value))
        # Existing tags are excellent truthful fallbacks; title-derived phrases
        # provide additional specific choices when needed.
        if isinstance(value, str) and len(txt) <= 20 and len(txt.split()) >= 1:
            if txt not in seen_pool:
                source_pool.append(txt)
                seen_pool.add(txt)
        ws = words(str(value))
        for n in (2, 3):
            for i in range(max(0, len(ws) - n + 1)):
                phrase = " ".join(ws[i:i+n]).strip()
                if len(phrase) <= 20 and phrase not in seen_pool:
                    source_pool.append(phrase)
                    seen_pool.add(phrase)

    # Marketplace tag/phrase signals are secondary fallback candidates.
    market_pool = []
    for row in (market_signals.get("high_signal_tags", []) or []) + (market_signals.get("high_signal_phrases", []) or []):
        value = row.get("keyword", "") if isinstance(row, dict) else str(row)
        value = normalize_text(value)
        if value and len(value) <= 20 and value not in market_pool:
            market_pool.append(value)

    def source_supported(tag):
        t = normalize_text(tag)
        if not t or len(t) > 20:
            return False
        if any(t in normalize_text(str(v)) or normalize_text(str(v)) in t for v in source_texts if str(v).strip()):
            return True
        tw = set(re.findall(r"[a-z0-9]+", t))
        sw = set(re.findall(r"[a-z0-9]+", normalize_text(" ".join(map(str, source_texts)))))
        identity_words = set(re.findall(r"[a-z0-9]+", normalize_text(" ".join([identity.get("product_type", "")] + (identity.get("gemstones", []) or [])))))
        return bool(tw and tw.issubset(sw) and (tw & identity_words))

    def market_supported(tag):
        t = normalize_text(tag)
        if t in {normalize_text(str(x)) for x in market_pool}:
            tw = set(re.findall(r"[a-z0-9]+", t))
            iw = set(re.findall(r"[a-z0-9]+", normalize_text(" ".join([identity.get("product_type", "")] + (identity.get("gemstones", []) or [])))))
            return bool(tw & iw)
        return False

    final = []
    used = set()
    for tag in raw_tags:
        n = normalize_text(tag)
        if n and n not in used and len(n) <= 20 and (source_supported(n) or market_supported(n)):
            final.append(tag)
            used.add(n)

    candidates = []
    for tag in source_pool + market_pool:
        n = normalize_text(tag)
        if not n or n in used or len(n) > 20:
            continue
        if source_supported(n) or market_supported(n):
            candidates.append(tag)

    # Prefer specific multi-word fallbacks and preserve the seller's original
    # tags when they are already factual and unique.
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda x: (len(x.split()) == 2, len(x.split()) == 3, len(x)), reverse=True)
    for tag in candidates:
        if len(final) >= 13:
            break
        n = normalize_text(tag)
        if n not in used:
            final.append(tag)
            used.add(n)

    # If the AI returned 13 valid unique tags, never truncate the result here.
    # Only fall back to the original tags when a replacement was necessary.
    if len(final) < 13:
        for tag in (listing.get("tags", []) or []):
            n = normalize_text(str(tag))
            if n and n not in used and len(n) <= 20:
                final.append(str(tag).strip())
                used.add(n)
            if len(final) >= 13:
                break

    if len(final) == 13:
        optimized["recommended_tags"] = final
    return optimized


def optimize_existing_listing(listing, market_signals):
    identity = extract_product_identity(listing)
    current = {
        "title": listing.get("title", ""),
        "tags": listing.get("tags", []) or [],
        "description": listing.get("description", ""),
        "materials": listing.get("materials", []) or [],
        "taxonomy_id": listing.get("taxonomy_id"),
    }

    # Keep the AI context compact. Competitor samples are useful for the UI,
    # but the model only needs the strongest keyword signals for optimization.
    compact_market_signals = {
        "queries": market_signals.get("queries", [])[:3],
        "marketplace_results_checked": market_signals.get("marketplace_results_checked", 0),
        "high_signal_phrases": (market_signals.get("high_signal_phrases", []) or [])[:12],
        "high_signal_tags": (market_signals.get("high_signal_tags", []) or [])[:12],
    }

    prompt = f"""
You are an elite Etsy SEO and conversion strategist specializing in handmade
Gemstone Jewelry. You are optimizing an EXISTING Etsy listing, not creating a
random generic listing.

CURRENT ETSY LISTING:
{json.dumps(current, ensure_ascii=False)}

NON-NEGOTIABLE PRODUCT IDENTITY (SOURCE OF TRUTH):
{json.dumps(identity, ensure_ascii=False)}

MARKETPLACE RESEARCH SIGNALS:
{json.dumps(compact_market_signals, ensure_ascii=False)}

INTERPRETATION:
- Give more weight to keywords with stronger signal_score, higher frequency, and wider query coverage.
- Prefer phrases that accurately match this listing over merely frequent generic phrases.
- Do not blindly copy competitor tags or titles; use them only as market-language evidence.
- Build a balanced tag set across core product, gemstone, material, style/use, occasion, and buyer-intent phrases where truthful.
- Classify keyword opportunities into primary, secondary, long-tail, buyer-intent, and avoid groups.
- Primary keywords should describe the core product and strongest relevant search intent.
- Secondary keywords should support material, gemstone, style, or use-case relevance.
- Long-tail keywords should be specific multi-word phrases a buyer could realistically search.
- Buyer-intent keywords should reflect gifting or purchase intent only when appropriate.
- Avoid keywords that are irrelevant, unsupported by the listing, misleading, overly generic, or likely to create false expectations.

PRODUCT IDENTITY LOCK — ABSOLUTE RULES:
- The CURRENT ETSY LISTING and NON-NEGOTIABLE PRODUCT IDENTITY are the only source of truth for what the product is.
- Marketplace research is ONLY for search-language evidence. Competitor products, gemstones, product types, materials, origins, sizes, and styles must NEVER be copied into this listing unless they are explicitly present in the current listing.
- If the target is a necklace, NEVER output a ring, bracelet, earring, pendant, or anklet as the product.
- If the target gemstone is opal, NEVER substitute coral, ruby, emerald, sapphire, or another gemstone.
- Preserve the target product type and primary gemstone in the optimized title and description.
- Preserve only objective facts that are supported by the current listing.

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
    {{"keyword": "...", "type": "primary", "signal": "strong", "reason": "..."}}
  ],
  "keyword_intelligence": {{
    "primary_keywords": ["..."],
    "secondary_keywords": ["..."],
    "long_tail_keywords": ["..."],
    "buyer_intent_keywords": ["..."],
    "avoid_keywords": ["..."]
  }},
  "attribute_recommendations": ["Only recommend attributes that can be verified from the source listing or Etsy taxonomy schema."],
  "changes_summary": ["..."],
  "search_volume_note": "Exact Etsy search volume is not available through the API; these are marketplace ranking/frequency signals."
}}
"""

    last_error = None
    last_identity_errors = []
    for attempt in range(3):
        try:
            response = ai_response_create(
                input_data=prompt,
                max_output_tokens=3200,
            )
            text = getattr(response, "output_text", "") or ""
            result = parse_listing_json(text)
            identity_errors = validate_product_identity(result, identity)
            if not identity_errors:
                result["product_identity_lock"] = identity
                return result

            last_identity_errors = identity_errors
            prompt = prompt + "\n\nIDENTITY VALIDATION FAILED. You must regenerate the entire JSON. Fix these exact errors:\n- " + "\n- ".join(identity_errors) + "\nDo not change the target product, gemstone, or supported facts. Return ONLY valid JSON."
        except RuntimeError as exc:
            last_error = exc
            # Gemini quota/rate-limit failures are deliberately not retried.
            # Repeating a quota failure can make free-tier testing worse.
            error_text = str(exc).lower()
            if any(marker in error_text for marker in (
                "429", "resource_exhausted", "rate limit", "quota", "too many requests"
            )):
                raise RuntimeError(
                    "Gemini API free-tier rate/quota limit reached. "
                    "Please wait for the limit to reset before trying again. "
                    f"Original error: {exc}"
                ) from exc
            prompt = prompt + "\n\nFINAL REMINDER: Return ONLY one compact JSON object. No markdown, no commentary, no code fences. Ensure all JSON strings escape newlines and quotes correctly."
        except Exception as exc:
            last_error = exc
            prompt = prompt + "\n\nFINAL REMINDER: Return ONLY one compact JSON object. No markdown, no commentary, no code fences. Ensure all JSON strings escape newlines and quotes correctly."

    if last_identity_errors:
        raise RuntimeError("Optimizer AI produced a product-mismatched result after 3 attempts: " + " | ".join(last_identity_errors))
    raise RuntimeError(f"Optimizer AI response failed after 3 attempts: {type(last_error).__name__}: {last_error}")


def improve_existing_listing_for_score(listing, optimized, score_report, identity, market_signals):
    """Use Gemini to fix only deterministic score gaps; validator remains the authority."""
    prompt = f"""
Improve this existing Etsy listing to satisfy the deterministic SEO score gaps.

SOURCE LISTING:
{json.dumps({k: listing.get(k) for k in ["title","tags","description","materials","attributes","taxonomy_id"]}, ensure_ascii=False)}

PRODUCT IDENTITY LOCK:
{json.dumps(identity, ensure_ascii=False)}

CURRENT RESULT:
{json.dumps(optimized, ensure_ascii=False)}

EXACT NEXT-ROUND INSTRUCTION:
{optimized.get("_keyword_gap_instruction", "No extra keyword-gap instruction.")}

SCORE REPORT:
{json.dumps(score_report, ensure_ascii=False)}

MARKETPLACE SIGNALS:
{json.dumps({"queries": market_signals.get("queries", [])[:3],
"high_signal_phrases": (market_signals.get("high_signal_phrases", []) or [])[:10],
"high_signal_tags": (market_signals.get("high_signal_tags", []) or [])[:10]}, ensure_ascii=False)}

Fix the listed gaps without inventing facts.

You must optimize toward every applicable deterministic check.
For every reported "13 tags" gap, inspect the exact weak/unsupported tag names
listed in the score report. Replace each weak tag with a different, evidence-backed
phrase from the source listing or relevant marketplace signals. Do not simply keep
the same weak tag. The replacement must accurately describe this exact product, be
20 characters or fewer, and remain unique across all 13 tags.
For every reported "keyword coverage" gap, inspect the exact supported keyword
phrase named in the score report. If it is supported by the source listing or
marketplace evidence, incorporate it naturally into the title, one tag, or the
description where it fits Etsy's limits. Do not merely mention that the keyword
exists; actually cover it in the returned content.
For long-tail keywords, preserve the full phrase when it is factual and fits
naturally. Never force an unsupported phrase just to raise the score.
Do not weaken, remove, or bypass a validation rule to increase the score.
Do not add the word "gift" merely for scoring.
Do not add unsupported attributes, measurements, materials, gemstone names,
styles, occasions, or claims.
If a gap is caused by a missing source fact, leave that fact out and explain it
rather than fabricating it.

Optimization priority:
1. Fix tag quality and relevance gaps first.
2. Fix keyword coverage gaps with truthful phrases supported by the source.
3. Improve buyer information only where the source listing provides the fact.
4. Never game the score by inventing attributes or unsupported product details.

Rules:
- Preserve exact product type and gemstones from the source.
- Never introduce unrelated gemstones, metals, origins, treatments, certifications,
  measurements, carat weights, or other unsupported facts.
- Title <= 140 characters and preferably 8-15 words.
- No subjective title fluff or keyword stuffing.
- Exactly 13 tags, each <=20 characters, diverse and relevant.
- First description sentence must clearly identify the actual product.
- Do not make healing/medical claims.
- Attribute recommendations may identify fields to verify, but never invent values.
- Do not copy competitor wording.
- If a missing fact prevents a perfect score, stay truthful rather than fabricate it.

Return ONLY valid JSON:
{{
  "recommended_title": "...",
  "recommended_tags": ["exactly 13 tags"],
  "recommended_description": "...",
  "keyword_strategy": [{{"keyword":"...","type":"primary|secondary|long-tail|buyer-intent","signal":"strong|medium|weak","reason":"..."}}],
  "keyword_intelligence": {{
    "primary_keywords": ["..."],
    "secondary_keywords": ["..."],
    "long_tail_keywords": ["..."],
    "buyer_intent_keywords": ["..."],
    "avoid_keywords": ["..."]
  }},
  "attribute_recommendations": ["..."],
  "changes_summary": ["..."],
  "search_volume_note": "Exact Etsy search volume is not available through the API; marketplace signals are directional."
}}
"""
    response = ai_response_create(input_data=prompt, max_output_tokens=3200)
    result = parse_listing_json(response.output_text)
    result.pop("_keyword_gap_instruction", None)
    return result

@app.post("/apply-optimized-listing")
async def apply_optimized_listing(request: Request):
    """Apply explicitly approved SEO fields to the user's own Etsy listing."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request.")
    if not payload.get("confirm"):
        raise HTTPException(status_code=400, detail="Update was not confirmed.")

    listing_id = str(payload.get("listing_id") or "").strip()
    title = str(payload.get("title") or "").strip()
    description = str(payload.get("description") or "")
    tags = payload.get("tags")
    expected_current_title = str(payload.get("expected_current_title") or "").strip()
    if not listing_id.isdigit():
        raise HTTPException(status_code=400, detail="A valid Etsy listing ID is required.")
    if not title or len(title) > 140:
        raise HTTPException(status_code=400, detail="Etsy listing title must be 1–140 characters.")
    if not isinstance(tags, list) or len(tags) != 13:
        raise HTTPException(status_code=400, detail="Exactly 13 Etsy tags are required.")
    clean_tags = [str(x).strip() for x in tags]
    if any(not x for x in clean_tags):
        raise HTTPException(status_code=400, detail="Tags cannot be empty.")
    if any(len(x) > 20 for x in clean_tags):
        bad = next(x for x in clean_tags if len(x) > 20)
        raise HTTPException(status_code=400, detail=f"Etsy tag exceeds 20 characters: {bad}")
    if len({x.lower() for x in clean_tags}) != 13:
        raise HTTPException(status_code=400, detail="All 13 tags must be unique.")
    if not expected_current_title:
        raise HTTPException(status_code=400, detail="The original listing title is required for the stale-change safety check.")

    context = get_shop_context()
    access_token = context["access_token"]
    get_url = f"https://api.etsy.com/v3/application/listings/{listing_id}"
    current_response = etsy_get(get_url, access_token, params={"language":"en", "legacy":"false"})
    if not current_response.ok:
        raise HTTPException(status_code=current_response.status_code, detail=current_response.text)
    current = current_response.json()
    if str(current.get("shop_id")) != str(context["shop_id"]):
        raise HTTPException(status_code=403, detail="This listing does not belong to your connected Etsy shop.")
    if str(current.get("title") or "").strip() != expected_current_title:
        raise HTTPException(status_code=409, detail="This listing changed on Etsy after it was analyzed. Please analyze it again before applying the optimization.")

    update_url = f"https://api.etsy.com/v3/application/shops/{context['shop_id']}/listings/{listing_id}"
    form_data = [("title", title), ("description", description), ("tags", ",".join(clean_tags))]
    update_response = etsy_patch_form(update_url, access_token, form_data)
    if not update_response.ok:
        raise HTTPException(status_code=update_response.status_code, detail={"message":"Etsy rejected the listing update.","etsy_response":update_response.text})

    verify_response = etsy_get(get_url, access_token, params={"language":"en", "legacy":"false"})
    verified = verify_response.json() if verify_response.ok else {}
    verified_tags = [str(x).strip() for x in (verified.get("tags") or [])]
    verification = {
        "title_updated": str(verified.get("title") or "").strip() == title,
        "description_updated": str(verified.get("description") or "") == description,
        "tags_updated": [x.lower() for x in verified_tags] == [x.lower() for x in clean_tags],
    }
    verification["all_fields_verified"] = all(verification.values())
    return {
        "status":"updated" if verification["all_fields_verified"] else "updated_verification_warning",
        "message": "Etsy listing updated successfully and all 3 SEO fields were verified." if verification["all_fields_verified"] else "Etsy accepted the update, but one or more fields could not be verified immediately. Check the listing again shortly.",
        "listing_id":listing_id, "updated_fields":["title","tags","description"], "verification":verification
    }

@app.get("/optimizer", response_class=HTMLResponse)
def optimizer_page(listing_id: str = Query("")):
    """Human-friendly UI for the existing Etsy listing SEO optimizer.

    This page analyzes an existing Etsy listing and can apply approved SEO fields after explicit confirmation.
    """
    response = HTMLResponse(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Etsy AI SEO Optimizer — V17</title>
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
  .keyword span:nth-child(2) { color:#7f92ba; font-size:11px; font-weight:800; letter-spacing:.4px; }
  .pill-group { margin-top:14px; }
  .pill-group h3 { margin:0 0 8px; font-size:13px; color:#aebbd8; }
  .pill-wrap { display:flex; flex-wrap:wrap; gap:8px; }
  .score-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  .score-box { background:#0b1223; border:1px solid #293957; border-radius:10px; padding:14px; text-align:center; }
  .score-box strong { display:block; color:#8f9fbe; font-size:11px; margin-bottom:6px; }
  .score-box span { font-size:24px; font-weight:850; }
  .hidden { display:none; }
  @media(max-width:760px) { .grid { grid-template-columns:1fr; } .full { grid-column:auto; } .meta { grid-template-columns:1fr; } .score-grid { grid-template-columns:1fr 1fr; } h1 { font-size:27px; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>🚀 Etsy AI SEO Optimizer</h1>
    <p class="sub">Analyze an existing Etsy listing and get a buyer-friendly title, 13 SEO tags, optimized description and keyword strategy. Review the changes, then apply them directly to Etsy.</p>
  </div>
  <div class="card">
    <form id="form">
      <label for="listing_url">Etsy listing URL</label>
      <input id="listing_url" name="listing_url" type="url" placeholder="https://www.etsy.com/listing/123456789/..." required>
      <input id="listing_id_hint" type="hidden" value="" />
      <button id="run" type="submit">Analyze Listing</button>
      <div id="status" class="status"></div>
    </form>
  </div>
  <div id="result" class="hidden">
    <div class="card" style="text-align:center">
      <button type="button" id="copyAll" style="margin-top:0">📋 Copy All SEO Content</button>
      <p class="muted" style="margin:9px 0 0">Copies the optimized title, 13 tags and description together.</p>
    </div>
    <div class="card" style="text-align:center; border-color:#4b6cff">
      <button type="button" id="applyToEtsy" style="margin-top:0; background:#55d98a; color:#06150b;">✅ Review &amp; Update Etsy Listing</button>
      <p id="applyStatus" class="muted" style="margin:9px 0 0">Nothing will be changed until you click the button and confirm.</p>
    </div>
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
        <div class="section-title"><h2>🧠 Advanced Keyword Intelligence</h2></div>
        <div class="pill-group"><h3>🥇 Primary Keywords</h3><div id="primaryKeywords" class="pill-wrap"></div></div>
        <div class="pill-group"><h3>🥈 Secondary Keywords</h3><div id="secondaryKeywords" class="pill-wrap"></div></div>
        <div class="pill-group"><h3>🎯 Long-Tail Keywords</h3><div id="longTailKeywords" class="pill-wrap"></div></div>
        <div class="pill-group"><h3>🛍️ Buyer-Intent Keywords</h3><div id="buyerIntentKeywords" class="pill-wrap"></div></div>
        <div class="pill-group"><h3>🚫 Avoid Keywords</h3><div id="avoidKeywords" class="pill-wrap"></div></div>
        <p class="muted">Keyword groups are AI recommendations based on the listing facts and marketplace signals. They are not exact Etsy search-volume data.</p>
      </div>
      <div class="card full">
        <div class="section-title"><h2>🔄 What Changed</h2></div><ul id="changes"></ul><p id="volumeNote" class="muted"></p>
      </div>
      <div class="card full">
        <div class="section-title"><h2>📈 SEO Score & Comparison</h2></div>
        <div class="score-grid">
          <div class="score-box"><strong>GENUINE SEO SCORE</strong><span id="optimizedScore">—</span></div>
          <div class="score-box"><strong>STATUS</strong><span id="scoreStatus">—</span></div>
          <div class="score-box"><strong>IMPROVEMENT ROUNDS</strong><span id="improvementRounds">—</span></div>
          <div class="score-box"><strong>GRADE</strong><span id="grade">—</span></div>
        </div>
        <p id="scoreNote" class="muted"></p>
        <div id="scoreGaps" class="muted"></div>
      </div>
      <div class="card full">
        <div class="section-title"><h2>🧠 SEO Intelligence Breakdown</h2></div>
        <div class="breakdown">
          <div class="break-row"><span>Product Identity</span><b id="identityBreak">—</b><small>/ 15</small></div>
          <div class="bar"><i id="identityBar"></i></div>
          <div class="break-row"><span>Title</span><b id="titleBreak">—</b><small>/ 15</small></div>
          <div class="bar"><i id="titleBar"></i></div>
          <div class="break-row"><span>13 Tags</span><b id="tagsBreak">—</b><small>/ 20</small></div>
          <div class="bar"><i id="tagsBar"></i></div>
          <div class="break-row"><span>Keyword Coverage</span><b id="keywordBreak">—</b><small>/ 15</small></div>
          <div class="bar"><i id="keywordBar"></i></div>
          <div class="break-row"><span>Description</span><b id="descBreak">—</b><small>/ 15</small></div>
          <div class="bar"><i id="descBar"></i></div>
          <div class="break-row"><span>Category &amp; Attributes</span><b id="attrBreak">—</b><small>/ —</small></div>
          <div class="bar"><i id="attrBar"></i></div>
          <div class="break-row"><span>Buyer Quality</span><b id="buyerBreak">—</b><small>/ 10</small></div>
          <div class="bar"><i id="buyerBar"></i></div>
        </div>
        <p id="scoreBasis" class="muted">The breakdown shows how the optimizer scores measurable listing structure. It is not Etsy's internal ranking formula.</p>
      </div>
      <div class="card full">
        <div class="section-title"><h2>🔄 Current vs Optimized</h2></div>
        <div class="grid" style="margin-top:0">
          <div>
            <h3 style="margin-top:0">Current Title</h3>
            <div id="compareCurrentTitle" class="title-box"></div>
            <h3>Current Tags</h3>
            <div id="compareCurrentTags" class="tags"></div>
          </div>
          <div>
            <h3 style="margin-top:0">Optimized Title</h3>
            <div id="compareOptimizedTitle" class="title-box"></div>
            <h3>Optimized Tags</h3>
            <div id="compareOptimizedTags" class="tags"></div>
          </div>
        </div>
      </div>
      <div class="card full">
        <div class="section-title"><h2>📌 Current Listing</h2></div>
        <div class="meta">
          <div><strong>CURRENT TITLE</strong><span id="currentTitle"></span></div>
          <div><strong>LISTING ID</strong><span id="listingId"></span></div>
          <div><strong>WRITE ACTION</strong><span id="writeAction" class="success">Not performed</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const queryListingId = new URLSearchParams(location.search).get('listing_id') || '';
if(queryListingId) $('listing_id_hint').value=queryListingId;
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
  $('compareCurrentTitle').textContent=data.current_listing?.title||''; $('compareOptimizedTitle').textContent=latest.title||'';
  const curTags=data.current_listing?.tags||[]; $('compareCurrentTags').innerHTML=''; curTags.forEach((tag,i)=>{const s=document.createElement('span');s.className='tag';s.textContent=`${i+1}. ${tag}`;$('compareCurrentTags').appendChild(s);});
  $('compareOptimizedTags').innerHTML=''; latest.tags.forEach((tag,i)=>{const s=document.createElement('span');s.className='tag';s.textContent=`${i+1}. ${tag}`;$('compareOptimizedTags').appendChild(s);});
  const sc=data.seo_score||{}; $('optimizedScore').textContent=(sc.optimized_score ?? '—')+'/100'; $('scoreStatus').textContent=sc.is_genuine_100?'100/100 ✓':((sc.hard_validation_errors||[]).length?'Needs validation fix':'Needs improvement'); $('improvementRounds').textContent=sc.improvement_rounds ?? '—'; $('grade').textContent=sc.grade||'—'; $('scoreNote').textContent=sc.note||'';
  const excluded=sc.excluded_points||0; const rawMax=sc.raw_max||sc.applicable_points||100; const rawPts=sc.raw_points;
  $('scoreBasis').textContent=excluded ? `Evidence-aware score: ${rawPts ?? '—'}/${rawMax} applicable points. ${excluded} point(s) excluded because Etsy did not expose those fields in this listing payload.` : `Evidence-aware score: ${rawPts ?? '—'}/${rawMax} measurable points.`;
  const gaps=sc.gaps||[]; $('scoreGaps').innerHTML=gaps.length ? '<b>Points still missing:</b> '+gaps.map(g=>`${g.section}: ${g.points_lost} — ${(g.reasons||[]).join('; ')}`).join(' | ') : '<span class="success"><b>100/100 — all applicable measurable checks passed.</b></span>';
  const bd=sc.breakdown||{};
  const setBreak=(key,el,bar)=>{const x=bd[key]||{}; const val=x.optimized; const max=x.max||1; if($(el)) $(el).textContent=(val ?? '—'); if($(bar)) $(bar).style.width=(val==null?'0':Math.max(0,Math.min(100,(val/max)*100)))+'%';};
  setBreak('product_identity','identityBreak','identityBar');
  setBreak('title','titleBreak','titleBar');
  setBreak('tags','tagsBreak','tagsBar');
  setBreak('keyword_coverage','keywordBreak','keywordBar');
  setBreak('description','descBreak','descBar');
  setBreak('attributes','attrBreak','attrBar');
  setBreak('buyer_quality','buyerBreak','buyerBar');
  const attrMaxEl=document.querySelector('#attrBreak')?.parentElement?.querySelector('small'); if(attrMaxEl) attrMaxEl.textContent='/ '+((bd.attributes||{}).max ?? '—');
  $('keywords').innerHTML=''; (o.keyword_strategy||[]).forEach(item=>{const d=document.createElement('div');d.className='keyword';const b=document.createElement('b');b.textContent=item.keyword||'';const meta=document.createElement('span');meta.textContent=`${(item.type||'keyword').toUpperCase()} • ${(item.signal||'signal').toUpperCase()}`;const sp=document.createElement('span');sp.textContent=item.reason||'';d.appendChild(b);d.appendChild(meta);d.appendChild(sp);$('keywords').appendChild(d);});
  const ki=o.keyword_intelligence||{};
  const renderPills=(id,items)=>{const el=$(id);el.innerHTML='';(items||[]).forEach(x=>{const s=document.createElement('span');s.className='tag';s.textContent=x;el.appendChild(s);});};
  renderPills('primaryKeywords',ki.primary_keywords); renderPills('secondaryKeywords',ki.secondary_keywords); renderPills('longTailKeywords',ki.long_tail_keywords); renderPills('buyerIntentKeywords',ki.buyer_intent_keywords); renderPills('avoidKeywords',ki.avoid_keywords);
  $('result').classList.remove('hidden');
}
$('form').addEventListener('submit',async e=>{e.preventDefault();$('run').disabled=true;$('status').className='status';$('status').textContent='Analyzing Etsy listing and marketplace keyword signals…';$('result').classList.add('hidden');try{const fd=new FormData();const enteredUrl=$('listing_url').value.trim();fd.append('listing_url',enteredUrl);if(!enteredUrl && $('listing_id_hint').value)fd.append('listing_id',$('listing_id_hint').value);const r=await fetch('/analyze-existing-listing',{method:'POST',body:fd});const raw=await r.text();let data;try{data=raw?JSON.parse(raw):{};}catch(parseErr){throw new Error(`Server returned invalid JSON (${r.status}): ${raw.slice(0,300)}`);}if(!r.ok)throw new Error(data.detail||`Analysis failed (HTTP ${r.status})`);if(data.validation_errors?.length){$('status').className='status warn';$('status').textContent='Analysis completed, but the generated result needs validation review.';}else{$('status').className='status success';$('status').textContent='Analysis complete — ready for your review. Nothing has been changed on Etsy.';}render(data);}catch(err){$('status').className='status error';$('status').textContent=err.message||'Something went wrong.';}finally{$('run').disabled=false;}});
document.querySelectorAll('[data-copy]').forEach(btn=>btn.addEventListener('click',async()=>{const key=btn.dataset.copy;const value=key==='tags'?latest.tags.join(', '):latest[key];try{await navigator.clipboard.writeText(value||'');const old=btn.textContent;btn.textContent='Copied ✓';setTimeout(()=>btn.textContent=old,1200);}catch(_){btn.textContent='Copy failed';}}));
$('copyAll').addEventListener('click',async()=>{const text=`TITLE\n${latest.title}\n\n13 TAGS\n${latest.tags.map((x,i)=>`${i+1}. ${x}`).join('\n')}\n\nDESCRIPTION\n${latest.description}`;try{await navigator.clipboard.writeText(text);const b=$('copyAll');b.textContent='Copied ✓';setTimeout(()=>b.textContent='📋 Copy All SEO Content',1400);}catch(_){$('copyAll').textContent='Copy failed';}});
$('applyToEtsy').addEventListener('click',async()=>{
  const listingId=$('listingId').textContent.trim(), currentTitle=$('compareCurrentTitle').textContent.trim();
  if(!listingId){$('applyStatus').className='error';$('applyStatus').textContent='No listing ID is available.';return;}
  if(!window.confirm('Update this Etsy listing now?\n\nOnly the title, 13 tags and description will be changed.')){ $('applyStatus').textContent='Update cancelled. Nothing was changed on Etsy.'; return; }
  const btn=$('applyToEtsy'); btn.disabled=true; $('applyStatus').className='muted'; $('applyStatus').textContent='Updating Etsy listing and verifying the changes…';
  try{
    const r=await fetch('/apply-optimized-listing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,listing_id:listingId,title:latest.title,tags:latest.tags,description:latest.description,expected_current_title:currentTitle})});
    const raw=await r.text(); let data; try{data=raw?JSON.parse(raw):{};}catch(_){throw new Error(`Server returned invalid JSON (${r.status}): ${raw.slice(0,300)}`);}
    if(!r.ok) throw new Error(data.detail?.message||data.detail||`Update failed (HTTP ${r.status})`);
    $('applyStatus').className=data.status==='updated'?'success':'warn'; $('applyStatus').textContent=data.message||'Etsy listing update completed.';
    $('writeAction').textContent=data.status==='updated'?'Updated & verified ✓':'Updated — verify shortly'; $('writeAction').className=data.status==='updated'?'success':'warn'; btn.textContent='✅ Etsy Listing Updated';
  }catch(err){$('applyStatus').className='error';$('applyStatus').textContent=err.message||'Update failed.';}finally{btn.disabled=false;}
});
</script>
</body>
</html>
""")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/analyze-existing-listing")
async def analyze_existing_listing(
    listing_id: str = Form(""),
    listing_url: str = Form(""),
):
    """Analyze an existing Etsy listing and return an improved SEO title/tags.

    This endpoint only reads/analyzes the listing. It does NOT edit, create,
    activate, or publish anything on Etsy.
    """
    url_id = extract_listing_id(listing_url)
    id_id = extract_listing_id(listing_id)

    # The explicit Etsy URL is authoritative. This prevents a stale hidden
    # listing_id query parameter from causing the optimizer to analyze a
    # completely different listing.
    if url_id and id_id and url_id != id_id:
        resolved_id = url_id
    else:
        resolved_id = url_id or id_id

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

    # Read-only Etsy taxonomy schema lookup. This gives the optimizer the current
    # category's supported property framework without writing anything to Etsy.
    taxonomy_id_for_lookup = listing.get("taxonomy_id")
    listing["taxonomy_properties"] = []
    if taxonomy_id_for_lookup:
        try:
            prop_url = f"https://api.etsy.com/v3/application/seller-taxonomy/nodes/{taxonomy_id_for_lookup}/properties"
            prop_response = etsy_get(prop_url, access_token)
            if prop_response.ok:
                prop_payload = prop_response.json() or {}
                listing["taxonomy_properties"] = prop_payload.get("results", []) or []
        except Exception:
            listing["taxonomy_properties"] = []

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
    identity = extract_product_identity(listing)
    optimized = repair_optimized_tags(optimized, listing, market_signals, identity)

    # Up to 5 targeted deterministic validation/improvement rounds. Gemini suggests edits;
    # the score is always calculated by the rules above.
    score_history = []
    best = optimized
    best_score = -1
    best_errors = []

    for round_no in range(5):
        candidate = {
            "title": optimized.get("recommended_title", ""),
            "tags": optimized.get("recommended_tags", []),
            "description": optimized.get("recommended_description", ""),
        }
        hard_errors = validate_listing(candidate)
        identity_errors = validate_product_identity(optimized, identity)
        all_errors = list(dict.fromkeys(hard_errors + identity_errors))

        score = seo_score_report(
            {
                "title": listing.get("title", ""),
                "tags": listing.get("tags", []) or [],
                "description": listing.get("description", ""),
                "attributes": listing.get("attributes", []) or [],
                "taxonomy_id": listing.get("taxonomy_id"),
                "taxonomy_properties": listing.get("taxonomy_properties", []) or [],
                "materials": listing.get("materials", []) or [],
                "style": listing.get("style", []) or [],
                "item_length": listing.get("item_length"),
                "item_width": listing.get("item_width"),
                "item_height": listing.get("item_height"),
                "item_dimensions_unit": listing.get("item_dimensions_unit"),
            },
            optimized,
            market_signals,
            all_errors,
            identity,
        )
        score_history.append(score["optimized_score"])

        if score["optimized_score"] > best_score:
            best_score = score["optimized_score"]
            best = optimized
            best_errors = all_errors

        if score["is_genuine_100"]:
            break
        if round_no == 4:
            break

        optimized = improve_existing_listing_for_score(
            listing, optimized, score, identity, market_signals
        )
        optimized = repair_optimized_tags(optimized, listing, market_signals, identity)

        # Extract exact supported phrases from the deterministic report and
        # pass them directly into the next improvement prompt.
        keyword_gap_phrases = []
        for gap in score.get("gaps", []):
            if gap.get("section") == "Keyword coverage":
                for reason in gap.get("reasons", []):
                    m = re.search(r"Missing supported [^:]+: (.+)$", str(reason))
                    if m:
                        keyword_gap_phrases.extend(
                            [x.strip() for x in m.group(1).split(",") if x.strip()]
                        )
        if keyword_gap_phrases:
            optimized["_keyword_gap_instruction"] = (
                "EXACT SUPPORTED PHRASES STILL MISSING. Incorporate these naturally "
                "where factual and within Etsy limits: "
                + " | ".join(list(dict.fromkeys(keyword_gap_phrases))[:8])
            )

    optimized = best
    validation_errors = best_errors
    seo_score = seo_score_report(
        {
            "title": listing.get("title", ""),
            "tags": listing.get("tags", []) or [],
            "description": listing.get("description", ""),
            "attributes": listing.get("attributes", []) or [],
            "taxonomy_id": listing.get("taxonomy_id"),
        },
        optimized,
        market_signals,
        validation_errors,
        identity,
    )
    seo_score["improvement_rounds"] = len(score_history) - 1
    seo_score["score_history"] = score_history

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
            "taxonomy_properties_checked": len(listing.get("taxonomy_properties", []) or []),
            "num_favorers": listing.get("num_favorers"),
            "url": listing.get("url", ""),
            "etsy_suggested_title": listing.get("suggested_title"),
        },
        "market_research": market_signals,
        "optimized_result": optimized,
        "seo_score": seo_score,
        "validation_errors": validation_errors,
        "product_identity": identity,
        "write_action_performed": False,
    }


# -------------------------------------------------------------------
# SHOP SEO PRIORITY AUDIT
# -------------------------------------------------------------------

def listing_priority_score(listing):
    """Prioritize listings for human review; not an Etsy ranking score."""
    title = str(listing.get("title", "") or "")
    tags = listing.get("tags", []) or []
    description = str(listing.get("description", "") or "")

    issues = []
    title_words = words(title)
    if not title:
        issues.append("missing title")
    if len(title) > 140:
        issues.append("title over 140 chars")
    if len(title_words) > 15:
        issues.append("title over 15 words")
    if len(title_words) != len(set(title_words)) and title_words:
        issues.append("repeated title words")
    if not isinstance(tags, list) or len(tags) != 13:
        issues.append("not 13 tags")
    if isinstance(tags, list):
        long_tags = sum(1 for t in tags if len(str(t).strip()) > 20)
        if long_tags:
            issues.append(f"{long_tags} tag(s) over 20 chars")
        if len({str(t).strip().lower() for t in tags if str(t).strip()}) < len([t for t in tags if str(t).strip()]):
            issues.append("duplicate tags")
    if not description:
        issues.append("missing description")
    elif len(description.strip()) < 80:
        issues.append("short description")

    base = len(issues) * 15
    favorites = int(listing.get("num_favorers") or 0)
    views = int(listing.get("views") or 0)
    engagement_bonus = min(20, round((favorites ** 0.5) * 2 + (views ** 0.5) * 0.25))
    priority = min(100, base + engagement_bonus)
    return priority, issues


@app.get("/shop-seo-audit")
async def shop_seo_audit(limit: int = 25, offset: int = 0):
    """Read active shop listings and rank which ones deserve SEO review first.

    This is a read-only triage report. It does not call the AI for every listing
    and it does not edit Etsy. The user can then open any listing in the existing
    optimizer for the full competitor/keyword analysis.
    """
    limit = max(5, min(int(limit), 50))
    offset = max(0, int(offset))

    context = get_shop_context()
    access_token = context["access_token"]
    shop_id = context["shop_id"]

    url = f"https://api.etsy.com/v3/application/shops/{shop_id}/listings"
    response = etsy_get(
        url,
        access_token,
        params={"state": "active", "limit": limit, "offset": offset},
    )
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    payload = response.json()
    raw_listings = payload.get("results", []) or []
    rows = []

    for item in raw_listings:
        priority, issues = listing_priority_score(item)
        title = item.get("title", "") or ""
        tags = item.get("tags", []) or []
        description = item.get("description", "") or ""
        rows.append({
            "listing_id": item.get("listing_id"),
            "title": title,
            "url": item.get("url", ""),
            "state": item.get("state", "active"),
            "priority_score": priority,
            "priority_level": "HIGH" if priority >= 45 else "MEDIUM" if priority >= 20 else "LOW",
            "issues": issues,
            "tag_count": len(tags) if isinstance(tags, list) else 0,
            "title_chars": len(title),
            "title_words": len(words(title)),
            "description_chars": len(description),
            "num_favorers": item.get("num_favorers", 0),
            "views": item.get("views", 0),
        })

    rows.sort(key=lambda x: (x["priority_score"], x.get("num_favorers", 0), x.get("views", 0)), reverse=True)

    return {
        "status": "success",
        "message": "Read-only shop SEO priority audit. No Etsy listing was changed.",
        "shop_id": shop_id,
        "count_returned": len(rows),
        "total_available": payload.get("count"),
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if payload.get("count") is not None and offset + limit < int(payload.get("count") or 0) else None,
        "method_note": "Priority score is a triage score based on listing-structure issues and capped engagement signals. It is not Etsy's ranking score.",
        "listings": rows,
    }


@app.get("/shop-audit", response_class=HTMLResponse)
def shop_audit_page():
    return HTMLResponse(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Etsy Shop SEO Audit</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#edf2ff;font-family:Inter,system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:30px 18px 60px}h1{margin:0 0 8px;font-size:31px}.sub{color:#aab6d3;line-height:1.55}.card{background:#121a2e;border:1px solid #263454;border-radius:16px;padding:18px;margin-top:18px}.toolbar{display:flex;gap:10px;align-items:end;flex-wrap:wrap}.field{flex:1;min-width:180px}.field label{display:block;font-size:13px;font-weight:800;margin-bottom:7px}.field input{width:100%;padding:11px;border-radius:9px;border:1px solid #344463;background:#0c1325;color:#fff}.btn{border:0;border-radius:9px;padding:11px 16px;font-weight:800;background:#7189ff;color:#071022;cursor:pointer}.status{margin-top:12px;color:#aab6d3}.error{color:#ff9c9c}.success{color:#87e0ad}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse;min-width:900px}.table th,.table td{text-align:left;padding:12px 10px;border-bottom:1px solid #263454;vertical-align:top}.table th{color:#8f9fbe;font-size:12px}.title{font-weight:750;max-width:360px}.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:11px;font-weight:850}.high{background:#4a2732;color:#ffb2bd}.medium{background:#493d22;color:#ffd889}.low{background:#1e3a31;color:#9de7bf}.issues{color:#aab6d3;font-size:12px;line-height:1.5}.link{color:#aebcff;text-decoration:none}.muted{color:#8f9fbe;font-size:12px}.empty{padding:24px;text-align:center;color:#aab6d3}
</style>
</head><body><div class="wrap">
<h1>📊 Etsy Shop SEO Priority Audit</h1>
<p class="sub">Find which active listings deserve SEO attention first. This report is read-only; it never edits or publishes anything.</p>
<div class="card"><div class="toolbar"><div class="field"><label>Listings to scan</label><input id="limit" type="number" min="5" max="50" value="25"></div><div class="field"><label>Offset</label><input id="offset" type="number" min="0" value="0"></div><button class="btn" id="run">Scan Shop</button></div><div id="status" class="status"></div></div>
<div class="card"><div id="summary" class="muted"></div><div class="table-wrap"><table class="table"><thead><tr><th>Priority</th><th>Listing</th><th>Issues</th><th>Title</th><th>Tags</th><th>Favorites</th><th>Action</th></tr></thead><tbody id="rows"></tbody></table></div></div>
</div><script>
const $=id=>document.getElementById(id);
function esc(v){const d=document.createElement('div');d.textContent=v??'';return d.innerHTML}
async function scan(){
 $('run').disabled=true;$('status').className='status';$('status').textContent='Reading active Etsy listings…';$('rows').innerHTML='';
 try{const r=await fetch(`/shop-seo-audit?limit=${encodeURIComponent($('limit').value)}&offset=${encodeURIComponent($('offset').value)}`);const raw=await r.text();let d;try{d=JSON.parse(raw)}catch(e){throw Error(`Server returned invalid JSON (${r.status}): ${raw.slice(0,300)}`)}if(!r.ok)throw Error(d.detail||`Audit failed (HTTP ${r.status})`);
 $('status').className='status success';$('status').textContent='Scan complete — no Etsy listing was changed.';$('summary').textContent=`Showing ${d.count_returned} listings${d.total_available!=null?' of '+d.total_available:''}. Highest priority appears first.`;
 (d.listings||[]).forEach(x=>{const tr=document.createElement('tr');const level=(x.priority_level||'LOW').toLowerCase();const url=x.url||'';const issues=(x.issues||[]).join(' • ')||'No obvious structure issue';tr.innerHTML=`<td><span class="badge ${level}">${esc(x.priority_level)}</span><br><b>${esc(x.priority_score)}</b>/100</td><td class="title">${esc(x.title)}</td><td class="issues">${esc(issues)}</td><td>${esc(x.title_chars)} chars<br>${esc(x.title_words)} words</td><td>${esc(x.tag_count)}/13</td><td>${esc(x.num_favorers??0)}</td><td><a class="link" href="/optimizer?listing_id=${encodeURIComponent(x.listing_id||'')}" target="_blank">Open optimizer →</a>${url?`<br><a class="link" href="${esc(url)}" target="_blank">View Etsy →</a>`:''}</td>`;$('rows').appendChild(tr)});
 if(!(d.listings||[]).length)$('rows').innerHTML='<tr><td colspan="7" class="empty">No active listings returned.</td></tr>';
 }catch(e){$('status').className='status error';$('status').textContent=e.message||'Something went wrong.'}finally{$('run').disabled=false}
}
$('run').addEventListener('click',scan);scan();
</script></body></html>
""")


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
