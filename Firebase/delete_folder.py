import os
import shutil
from firebase_admin import credentials, storage
import firebase_admin

FIREBASE_CERT   = "/home/pi/Public/WildLife-Detection/Firebase/real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json"
FIREBASE_BUCKET = "real-time-wildlife-detector.firebasestorage.app"
LOCAL_IMAGES_DIR = "/home/pi/Public/WildLife-Detection/Firebase/Images"

cred = credentials.Certificate(FIREBASE_CERT)
firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
bucket = storage.bucket()

# Delete Firebase
for prefix in ["detected/", "empty/"]:
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Deleting {len(blobs)} files from Firebase '{prefix}'...")
    for blob in blobs:
        blob.delete()
        print(f"  Deleted: {blob.name}")

# Delete local
if os.path.exists(LOCAL_IMAGES_DIR):
    entries = os.listdir(LOCAL_IMAGES_DIR)
    print(f"\nDeleting {len(entries)} local folders from {LOCAL_IMAGES_DIR}...")
    for entry in entries:
        full_path = os.path.join(LOCAL_IMAGES_DIR, entry)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"  Deleted folder: {entry}")
        else:
            os.remove(full_path)
            print(f"  Deleted file: {entry}")
else:
    print(f"Local directory not found: {LOCAL_IMAGES_DIR}")

print("\nDone")