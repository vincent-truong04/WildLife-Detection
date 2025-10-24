import firebase_admin
from firebase_admin import credentials, storage

# 1️⃣ Point to your downloaded JSON key file
cred = credentials.Certificate("real-time-wildlife-detector-firebase-adminsdk-fbsvc-8cd04e9916.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "real-time-wildlife-detector.firebasestorage.app"
})

# 2️⃣ Choose an image on your computer to upload
file_path = "test_image.jpg"  # <-- put the image name here
bucket = storage.bucket()
blob = bucket.blob("uploads/test_image.jpg")  # path inside Firebase Storage

# 3️⃣ Upload the image
blob.upload_from_filename(file_path)
