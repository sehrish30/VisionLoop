import hashlib
import io
import uuid
from PIL import Image, ImageOps, UnidentifiedImageError

from . import store

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000


def save_image(content, filename, source="upload", label=None, split="train"):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a JPG, PNG, or WebP image under 8 MB.")
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Supported image formats are JPG, PNG, and WebP.")
            if original.width * original.height > MAX_PIXELS:
                raise ValueError("Choose an image smaller than 20 megapixels.")
            image = ImageOps.exif_transpose(original).convert("RGB")
            digest = hashlib.sha256(f"{image.size}".encode() + image.tobytes()).hexdigest()
            image.thumbnail((1200, 1200))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file could not be read as an image.") from exc
    with store.connect() as db:
        existing = db.execute("SELECT * FROM images WHERE hash=? OR stored_hash=?", (digest, hashlib.sha256(content).hexdigest())).fetchone()
        if existing:
            return dict(existing)
        image_id = uuid.uuid4().hex
        path = f"images/{image_id}.jpg"
        (store.DATA / path).write_bytes(output.getvalue())
        db.execute("INSERT INTO images (id,path,hash,filename,source,label,split,created_at,stored_hash) VALUES (?,?,?,?,?,?,?,?,?)", (image_id, path, digest, filename[:200], source, label, split, store.now(), hashlib.sha256(output.getvalue()).hexdigest()))
        return dict(db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())
