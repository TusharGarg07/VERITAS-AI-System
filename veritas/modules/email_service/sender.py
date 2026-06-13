import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from veritas.config.settings import settings
from veritas.utils.logger import logger

class EmailSender:
    def send_email_task(self, to_email: str, subject: str, html_content: str, pdf_path: str):
        try:
            print("EMAIL FUNCTION CALLED")
            print("--- SMTP DEBUG ---")
            print("LOGIN USER:", settings.SMTP_USER)
            print("EMAIL FROM:", settings.EMAIL_FROM)
            print("TO EMAIL:", to_email)
            print("------------------")

            msg = MIMEMultipart()
            msg['From'] = settings.EMAIL_FROM
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(html_content, 'html'))

            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(pdf_path)}",
                    )
                    msg.attach(part)
            else:
                logger.error(f"Attachment file not found: {pdf_path}")
                return False

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASS)
                server.send_message(msg)
            
            logger.info(f"Email successfully sent to {to_email}")
            return True
        except Exception as e:
            print("EMAIL ERROR:", str(e))
            logger.error(f"Failed to send email: {str(e)}")
            return False

email_sender = EmailSender()
