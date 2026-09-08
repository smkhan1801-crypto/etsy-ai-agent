from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "running",
        "agent": "Etsy AI Listing Agent"
    }

@app.get("/generate")
def generate_listing(product_name: str):
    return {
        "title": f"SEO Optimized {product_name}",
        "description": f"Premium handmade {product_name} for Etsy",
        "tags": [
            product_name,
            "etsy jewelry",
            "gift for her",
            "handmade"
        ]
    }
