import os
import base64
import secrets
import hashlib
import hmac
import time
import urllib.parse
import json

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

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)

TOKEN_KEY = "etsy:oauth_token"


@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent"
    }


# ---------------------------------------------------------
# REDIS
# ---------------------------------------------------------

def save_etsy_token(token_data):
    redis_client.set(
        TOKEN_KEY,
        json.dumps(token_data)
    )


def get_etsy_token():
    data = redis_client.get(TOKEN_KEY)

    if not data:
        return None

    try:
        return json.loads(data)
    except Exception:
        return None


# ---------------------------------------------------------
# ETSY TOKEN REFRESH
# ---------------------------------------------------------

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

    save_etsy_token(new_token)

    return new_token


def get_valid_etsy_token():
    token_data = get_etsy_token()

    if not token_data:
        return None

    expires_at = token_data.get("expires_at")

    if expires_at:
        # Refresh slightly before expiry
        if int(time.time()) >= int(expires_at) - 120:
            refreshed = refresh_etsy_token()

            if refreshed:
                token_data = refreshed

    return token_data


# ---------------------------------------------------------
# ETSY API
# ---------------------------------------------------------

def etsy_headers(access_token):
    return {
        "x-api-key": f"{ETSY_KEYSTRING}:{ETSY_SHARED_SECRET}",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def etsy_get(url, access_token):
    response = requests.get(
        url,
        headers=etsy_headers(access_token),
        timeout=30,
    )

    if response.status_code == 401:
        refreshed = refresh_etsy_token()

        if refreshed:
            access_token = refreshed.get("access_token")

            response = requests.get(
                url,
                headers=etsy_headers(access_token),
                timeout=30,
            )

    return response


# ---------------------------------------------------------
# OAUTH SECURITY
# ---------------------------------------------------------

def create_oauth_signature(state, code_verifier, timestamp):
    message = f"{state}|{code_verifier}|{timestamp}".encode("utf-8")

    return hmac.new(
        ETSY_SHARED_SECRET.encode("utf-8"),
        message,
        hashlib.sha256
    ).hexdigest()


# ---------------------------------------------------------
# ETSY CONNECT
# ---------------------------------------------------------

@app.get("/etsy/connect")
def etsy_connect():

    if not ETSY_KEYSTRING or not ETSY_SHARED_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Etsy API credentials are not configured."
        )

    state = secrets.token_urlsafe(32)

    code_verifier = secrets.token_urlsafe(64)

    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(
            code_verifier.encode("utf-8")
        ).digest()
    ).decode("utf-8").rstrip("=")

    timestamp = str(int(time.time()))

    signature = create_oauth_signature(
        state,
        code_verifier,
        timestamp
    )

    cookie_value = (
        f"{state}|{code_verifier}|{timestamp}|{signature}"
    )

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

    response = RedirectResponse(
        url=etsy_url
    )

    response.set_cookie(
        key="etsy_oauth",
        value=cookie_value,
        max_age=600,
        httponly=True,
        secure=True,
        samesite="lax"
    )

    return response


# ---------------------------------------------------------
# OAUTH CALLBACK
# ---------------------------------------------------------

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
            detail="Missing Etsy authorization code or state."
        )

    oauth_cookie = request.cookies.get("etsy_oauth")

    if not oauth_cookie:
        raise HTTPException(
            status_code=400,
            detail="OAuth session cookie is missing. Please start again from /etsy/connect."
        )

    try:
        cookie_state, code_verifier, timestamp, signature = (
            oauth_cookie.split("|", 3)
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth session."
        )

    expected_signature = create_oauth_signature(
        cookie_state,
        code_verifier,
        timestamp
    )

    if not hmac.compare_digest(
        signature,
        expected_signature
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth session signature."
        )

    if not hmac.compare_digest(
        cookie_state,
        state
    ):
        raise HTTPException(
            status_code=400,
            detail="OAuth state mismatch."
        )

    try:
        timestamp_int = int(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth timestamp."
        )

    if int(time.time()) - timestamp_int > 600:
        raise HTTPException(
            status_code=400,
            detail="OAuth session expired. Please start again."
        )

    # Exchange authorization code for Etsy token
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
            detail="Etsy did not return an access token."
        )

    # Etsy access tokens expire.
    # Store an approximate expiration timestamp.
    expires_in = int(
        token_data.get(
            "expires_in",
            3600
        )
    )

    token_data["expires_at"] = int(
        time.time()
    ) + expires_in

    # SAVE TOKEN TO REDIS
    save_etsy_token(token_data)

    user_id = access_token.split(".")[0]

    response = RedirectResponse(
        url="/etsy/status"
    )

    response.delete_cookie(
        key="etsy_oauth",
        secure=True,
        samesite="lax"
    )

    return response


# ---------------------------------------------------------
# ETSY STATUS
# ---------------------------------------------------------

@app.get("/etsy/status")
def etsy_status():

    token_data = get_valid_etsy_token()

    if not token_data:
        return {
            "connected": False,
            "message": "Etsy account is not connected."
        }

    access_token = token_data.get("access_token")

    if not access_token:
        return {
            "connected": False,
            "message": "Etsy access token is missing."
        }

    user_id = access_token.split(".")[0]

    # Try to retrieve the seller's shop
    shops_url = (
        f"https://api.etsy.com/v3/application/users/"
        f"{user_id}/shops"
    )

    shop_response = etsy_get(
        shops_url,
        access_token
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


# ---------------------------------------------------------
# PHOTO LISTING GENERATOR
# ---------------------------------------------------------

@app.post("/generate-listing-photo")
async def generate_listing_photo(
    image: UploadFile = File(...),
    extra_info: str = Form("")
):

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_bytes = await image.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Please use an image under 10 MB."
        )

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:{image.content_type};base64,{image_base64}"
    )

    prompt = f"""
You are an expert Etsy SEO listing agent specializing in
handmade gemstone jewelry.

Analyze the uploaded product photograph carefully.

Identify ONLY what can reasonably be observed from the image.

Do NOT invent:
- gemstone identity
- natural/genuine status
- gemstone origin
- treatment
- certification
- metal purity
- measurements
- carat weight
- authenticity

Additional seller information:

{extra_info}

Create an Etsy-ready listing.

Return:

1. Product observations
2. SEO title under 140 characters
3. Exactly 13 Etsy tags, each 20 characters or fewer
4. Natural Etsy description
5. Materials
6. Suggested Etsy attributes
7. Gift/occasion keywords
8. Claims that require seller verification

Important:

- Do not claim "natural", "genuine", "solid gold",
  "925", etc. unless supplied by the seller or clearly verifiable.
- Keep keywords natural.
- Avoid keyword stuffing.
- Make the listing suitable for handmade gemstone jewelry.
- Make the title readable and buyer-focused.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    }
                ]
            }
        ]
    )

    return {
        "status": "success",
        "listing": response.output_text
    }


# ---------------------------------------------------------
# SIMPLE TEXT LISTING GENERATOR
# ---------------------------------------------------------

@app.post("/generate-listing")
async def generate_listing(
    product: str = Form(...),
    details: str = Form("")
):

    prompt = f"""
Create a professional Etsy listing for this handmade jewelry product.

Product:
{product}

Seller details:
{details}

Return:

1. SEO title under 140 characters
2. Exactly 13 Etsy tags
3. Product description
4. Materials
5. Suggested attributes
6. Gift keywords
7. Claims requiring seller verification

Never invent gemstone authenticity, natural status,
metal purity, origin, treatment or measurements.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return {
        "status": "success",
        "listing": response.output_text
    }
# ---------------------------------------------------------
# ETSY SHOP CONFIG
# ---------------------------------------------------------

@app.get("/etsy/config")
def etsy_config():

    token_data = get_valid_etsy_token()

    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Etsy account is not connected."
        )

    access_token = token_data.get("access_token")

    user_id = access_token.split(".")[0]

    # Get shop
    shops_url = (
        f"https://api.etsy.com/v3/application/users/"
        f"{user_id}/shops"
    )

    shop_response = etsy_get(
        shops_url,
        access_token
    )

    if not shop_response.ok:
        raise HTTPException(
            status_code=shop_response.status_code,
            detail=shop_response.text
        )

    shop_data = shop_response.json()

    shop = shop_data.get("shop", shop_data)

    shop_id = shop.get("shop_id")

    # Get processing profiles
    processing_url = (
        f"https://api.etsy.com/v3/application/shops/"
        f"{shop_id}/readiness-state-definitions"
    )

    processing_response = etsy_get(
        processing_url,
        access_token
    )

    processing_data = {}

    if processing_response.ok:
        processing_data = processing_response.json()

    # Get seller taxonomy
    taxonomy_url = (
        "https://api.etsy.com/v3/application/seller-taxonomy/nodes"
    )

    taxonomy_response = etsy_get(
        taxonomy_url,
        access_token
    )

    taxonomy_data = {}

    if taxonomy_response.ok:
        taxonomy_data = taxonomy_response.json()

    return {
        "shop_id": shop_id,
        "shop_name": shop.get("shop_name"),
        "processing_profiles": processing_data,
        "seller_taxonomy": taxonomy_data,
    }
/etsy/config
