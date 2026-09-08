import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent"
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
