import os
import base64
import secrets
import hashlib
import hmac
import time
import urllib.parse
import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ETSY_KEYSTRING = os.getenv("ETSY_API_KEYSTRING")
ETSY_SHARED_SECRET = os.getenv("ETSY_SHARED_SECRET")

ETSY_REDIRECT_URI = "https://etsy-ai-agent.onrender.com/etsy/callback"


@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent"
    }


def create_oauth_signature(state, code_verifier, timestamp):
    message = f"{state}|{code_verifier}|{timestamp}".encode("utf-8")

    return hmac.new(
        ETSY_SHARED_SECRET.encode("utf-8"),
        message,
        hashlib.sha256
    ).hexdigest()


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

    response = RedirectResponse(url=etsy_url)

    response.set_cookie(
        key="etsy_oauth",
        value=cookie_value,
        max_age=600,
        httponly=True,
        secure=True,
        samesite="lax"
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

    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(
            status_code=400,
            detail="Invalid OAuth session signature."
        )

    if not hmac.compare_digest(cookie_state, state):
        raise HTTPException(
            status_code=400,
            detail="OAuth state mismatch."
        )

    if int(time.time()) - int(timestamp) > 600:
        raise HTTPException(
            status_code=400,
            detail="OAuth session expired. Please start again."
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
            detail="Etsy did not return an access token."
        )

    user_id = access_token.split(".")[0]

    response = RedirectResponse(
        url="/etsy/status?connected=1"
    )

    response.delete_cookie(
        key="etsy_oauth",
        secure=True,
        samesite="lax"
    )

    return response


@app.get("/etsy/status")
def etsy_status(connected: int = 0):
    if connected == 1:
        return {
            "connected": True,
            "message": "Etsy authorization completed successfully."
        }

    return {
        "connected": False,
        "message": "Etsy account is not connected yet."
    }


@app.post("/generate-listing-photo")
async def generate_listing_photo(
    image: UploadFile = File(...),
    extra_info: str = Form("")
):
    if not image.content_type or not image.content_type.startswith("image/"):
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

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = (
        f"data:{image.content_type};base64,{image_base64}"
    )

    prompt = f"""
You are an expert Etsy SEO listing agent specializing in handmade
gemstone jewelry.

Analyze the uploaded product photograph carefully.

Identify ONLY what can reasonably be observed from the image.
Do NOT invent gemstone identity, natural/genuine status, origin,
treatment, certification, metal purity, or measurements.

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
- Do not claim "natural", "genuine", "solid gold", "925", etc.
  unless supplied by the seller or clearly verifiable.
- Keep keywords natural.
- Avoid keyword stuffing.
- Make the listing suitable for handmade gemstone jewelry.
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
