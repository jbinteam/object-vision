"""Download a reproducible random selection of background photos from Lorem Picsum."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parent


def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "objectSegmentation-background-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download(photo, destination, width, height):
    url = f"https://picsum.photos/id/{photo['id']}/{width}/{height}"
    filename = f"picsum_{photo['id']}_{width}x{height}.jpg"
    path = destination / filename
    payload = path.read_bytes() if path.is_file() else fetch(url)
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.size != (width, height) or image.format != "JPEG":
            raise ValueError(f"Unexpected image returned for {url}")
    if not path.is_file():
        temporary = path.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {"file": filename, "provider_id": photo["id"], "author": photo["author"],
            "source_url": photo["url"], "download_url": url,
            "width": width, "height": height, "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "backgrounds/picsum")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if min(args.count, args.width, args.height) < 1:
        parser.error("count, width, and height must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = []
    for page in range(1, 6):
        catalog.extend(json.loads(fetch(f"https://picsum.photos/v2/list?page={page}&limit=100")))
    photos = list({p["id"]: p for p in catalog if p["width"] >= 1280 and p["height"] >= 720
                   and 1.2 <= p["width"] / p["height"] <= 2.5}.values())
    random.Random(args.seed).shuffle(photos)
    if len(photos) < args.count:
        parser.error(f"Only {len(photos)} suitable backgrounds in this catalog; lower --count")
    records, failures = [], []
    while photos and len(records) < args.count:
        batch, photos = photos[:args.count-len(records)], photos[args.count-len(records):]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download, photo, args.output, args.width, args.height): photo for photo in batch}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                    print(f"[{len(records)}/{args.count}] downloaded {records[-1]['file']}", flush=True)
                except (OSError, ValueError) as error:
                    failures.append({"id": futures[future]["id"], "error": str(error)})
    manifest = {"provider": "https://picsum.photos/", "seed": args.seed,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "images": sorted(records, key=lambda record: int(record["provider_id"])), "failures": failures}
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(records) != args.count:
        raise SystemExit(f"Downloaded {len(records)}/{args.count}; see sources.json")
    print(f"Ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
