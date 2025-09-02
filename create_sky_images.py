import os
import requests
from PIL import Image
from io import BytesIO

API_KEY = "52079194-786d4b722b22e071f49ad22b4"
OUTPUT_DIR = "data/sky_img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_sky_images(count=50, size=(256, 256)):
    url = "https://pixabay.com/api/"
    params = {
        "key": API_KEY,
        "q": "sky",
        "image_type": "photo",
        "safesearch": "true",
        "per_page": count,
        "page": 1
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    print("Total hits available:", data["totalHits"])
    print("Number of images returned:", len(data["hits"]))
    print("Saving to:", os.path.abspath(OUTPUT_DIR))

    hits = data["hits"]

    for idx, hit in enumerate(hits, start=1):
        img_url = hit.get("largeImageURL")
        if not img_url:
            continue

        try:
            img_resp = requests.get(img_url, timeout=10)
            img_resp.raise_for_status()

            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img = img.resize(size, Image.LANCZOS)

            fname = os.path.join(OUTPUT_DIR, f"sky_{idx:03d}.jpg")
            img.save(fname, "JPEG")
            print(f"✅ Saved {fname}")
        except Exception as e:
            print(f"❌ Failed to save image {idx}: {e}")

download_sky_images()
