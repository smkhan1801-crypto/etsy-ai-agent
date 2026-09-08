import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ListingRequest(BaseModel):
    product_name: str
    gemstone: str = ""
    metal: str = ""
    color: str = ""
    style: str = ""


@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent"
    }


@app.post("/generate-listing")
def generate_listing(data: ListingRequest):

    prompt = f"""
You are an expert Etsy SEO listing agent for handmade gemstone jewelry.

Create an Etsy-ready listing for this product.

Product name: {data.product_name}
Gemstone: {data.gemstone}
Metal: {data.metal}
Color: {data.color}
Style: {data.style}

Return:
1. SEO title under 140 characters
2. Exactly 13 Etsy tags, each under 20 characters
3. Detailed natural Etsy description
4. Materials
5. Suggested Etsy attributes
6. Gift/occasion keywords

Do not invent gemstone type, metal purity, origin, certification,
or "natural/genuine" claims unless supplied in the product information.

Return the result in clean JSON.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return {
        "listing": response.output_text
    }
