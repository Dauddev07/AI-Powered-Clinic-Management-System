"""One-time setup — generates a VAPID keypair for Web Push (see
app.services.push_notifications). Run once, then copy the printed values into
Render's environment variables (VAPID_PRIVATE_KEY_B64, VAPID_PUBLIC_KEY,
VAPID_CONTACT_EMAIL) and your local .env — never commit them to the repo.

Re-running this generates a DIFFERENT keypair, which invalidates every push
subscription already stored in push_subscriptions (a browser subscribes against a
specific public key) — existing patients would need to re-enable notifications.
Only re-run if you actually intend to rotate the keys, not routinely.

Usage (from backend/, with the venv activated):
    python -m app.scripts.generate_vapid_keys
"""
import base64

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from py_vapid import Vapid


def main() -> None:
    vapid = Vapid()
    vapid.generate_keys()

    private_pem = vapid.private_pem()
    private_b64 = base64.b64encode(private_pem).decode()

    public_raw = vapid.public_key.public_bytes(
        encoding=Encoding.X962, format=PublicFormat.UncompressedPoint
    )
    public_b64 = base64.urlsafe_b64encode(public_raw).rstrip(b"=").decode()

    print("Add these to your environment (Render + local .env):\n")
    print(f"VAPID_PRIVATE_KEY_B64={private_b64}")
    print(f"VAPID_PUBLIC_KEY={public_b64}")
    print("VAPID_CONTACT_EMAIL=<a real contact email/URL for abuse reports>")


if __name__ == "__main__":
    main()
