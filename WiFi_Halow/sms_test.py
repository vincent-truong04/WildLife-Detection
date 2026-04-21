import smtplib
from email.mime.text import MIMEText

GMAIL_ADDRESS      = "vincenttruong.usa@gmail.com"
GMAIL_APP_PASSWORD = "ncvvcnqyxvdkstlm"
YOUR_EMAIL         = "vincenttruong.usa@gmail.com"  # just send to yourself first to confirm SMTP works

body = "Wildlife detected!\nCam: test\nObjects: white-tailed deer\nTime: 04_21_2026_120000"

msg = MIMEText(body)
msg["From"]    = GMAIL_ADDRESS
msg["To"]      = YOUR_EMAIL
msg["Subject"] = "Wildlife Alert"

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    server.sendmail(GMAIL_ADDRESS, YOUR_EMAIL, msg.as_string())

print("Email sent!")