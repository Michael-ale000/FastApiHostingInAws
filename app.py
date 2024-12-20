import os
import imaplib
import email
from email.header import decode_header
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#import streamlit as st  # For debugging logs
from groq import Groq  # Groq client
from supabase import Client,create_client
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification,AutoModelForSequenceClassification,AutoTokenizer

# Initialize environment variables
load_dotenv()
app = FastAPI()

# User credentials from .env file
EMAIL_USER = os.getenv("EMAIL_USER") or "valorantphp06@gmail.com"
EMAIL_PASS = os.getenv("EMAIL_PASS") or "zelb oqav jcrw mruq"
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
port = int(os.getenv("PORT", 8000))

model_path = "Michael444/DistilbertModel"  # Path to the saved model
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Pydantic model for Email
class Email(BaseModel):
    email_id: str
    sender: str
    subject: str
    content: str
    arrived_at: str
    date: str
    is_important: int

# Function to classify email importance (dummy implementation)
def is_important(message_text):
    inputs = tokenizer(message_text, return_tensors="tf", truncation=True, padding=True)
    outputs = loaded_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    prediction = tf.argmax(outputs.logits, axis=-1).numpy().item()
    return prediction == 1

# Save email to database
def save_email_to_db(email_data):
    email = {
        'email_id': email_data['email_id'],
        'user_email': EMAIL_USER,
        'sender': email_data['sender'],
        'subject': email_data['subject'],
        'content': email_data['content'],
        'arrivedat': email_data['arrived_at'],
        'date': email_data['date'],
        'is_important': int(email_data['is_important']),
        'status': 0,
    }

    # Insert data into the Supabase table
    response = supabase.table('emails3').insert(email).execute()
    
    # Check if the data was successfully inserted
    if response.data:
        print(f"Email with ID {email['email_id']} saved successfully.")
    elif response.error:
        print(f"Failed to save email with ID {email['email_id']}: {response.error.message}")



def is_email_already_stored(email_id: str) -> bool:
    """
    Check if an email with the given email_id is already stored in the database.

    Args:
        email_id (str): The unique ID of the email.

    Returns:
        bool: True if the email is already stored, False otherwise.
    """
    try:
        response = supabase.table("emails3").select("email_id").eq("email_id", email_id).execute()
        return bool(response.data)  # Return True if email exists, False otherwise
    except Exception as e:
        print(f"Error checking email in database: {str(e)}")
        return False


def fetch_Inbox_and_save_emails():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("INBOX")

    today = datetime.today().strftime("%d-%b-%Y")
    _, data = mail.search(None, f'SINCE {today}')
    email_ids = data[0].split()

    for email_id in email_ids:
        email_id_decoded = email_id.decode()

        # Check if the email is already stored
        if is_email_already_stored(email_id_decoded):
            print(f"Email with ID {email_id_decoded} is already stored. Skipping...")
            continue

        # Fetch and process the email
        _, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Decode the subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")

                # Extract sender, date, and content
                from_ = msg.get("From")
                date = msg.get("Date")
                content = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            content = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    content = msg.get_payload(decode=True).decode(errors="ignore")

                email_data = {
                    "email_id": email_id_decoded,
                    "sender": from_,
                    "subject": subject,
                    "content": content,
                    "arrived_at": date,
                    "date": today,
                    "is_important": is_important(content),
                    "status": 0
                }

                # Save email data to Supabase
                save_email_to_db(email_data)

    mail.logout()
    return {"message": "Emails fetched and saved successfully."}


#groq api llm response
def get_llama_response(email_body):
    """Fetch a human-like automated response from the llama3-8b-8192 model using the Groq API."""
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("Groq API key is missing from environment variables.")

    prompt = f"""You are a human replying to an email. Your response should feel warm, genuine, and tailored to the specific message below:
    
    Email Content:
    {email_body}

    Compose a friendly, clear, and concise reply."""

    client = Groq(api_key=groq_api_key)

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        return f"Error: {str(e)}"


def send_automated_replies():
    today = datetime.today().strftime("%d-%b-%Y")
    # Query Supabase to get today's important emails with status = 0
    response = supabase.table("emails3") \
        .select("email_id, sender, subject, content") \
        .filter("is_important", "eq", 1) \
        .filter("status", "eq", 0) \
        .filter("date", "eq", today) \
        .execute()

    emails = response.data

    if not emails:
        raise HTTPException(status_code=404, detail="No important emails found for today that need replies.")

    # Process each email
    replies = []
    for email in emails:
        email_id = email["email_id"]
        sender = email["sender"]
        subject = email["subject"]
        content = email["content"]

        # Generate response using LLaMA (or Claude)
        generated_response = get_llama_response(content)
        if generated_response.startswith("Error"):
            replies.append({"email_id": email_id, "sender": sender, "status": "Failed to generate response"})
            continue

        # Send the email
        sent = send_email(
            from_email=EMAIL_USER,  # Replace with your email
            app_password=EMAIL_PASS,  # Replace with your app password
            to_email=sender,
            subject=f"Re: {subject}",
            body=generated_response
        )

        if sent:
            # Update the status of the email in Supabase to 1 (reply sent)
            update_response = supabase.table("emails3") \
                .update({"status": 1}) \
                .eq("email_id", email_id) \
                .execute()

            replies.append({"email_id": email_id, "sender": sender, "status": "Reply sent and status updated"})
        else:
            replies.append({"email_id": email_id, "sender": sender, "status": "Failed to send reply"})

    return {"replies": replies}


def send_email(from_email, app_password, to_email, subject, body):
    """Sends an email via Gmail's SMTP server."""
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, app_password)

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


# Routes
@app.get("/fetchandsaveinbox")
async def fetch_and_save_endpoint():
    try:
        result = fetch_Inbox_and_save_emails()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/displayallemailsinbox")
async def display_all_emails():
    """
    Fetch all emails from Supabase for the logged-in user and today's date.
    """
    try:
        today = datetime.today().strftime("%d-%b-%Y")  # Ensure date format matches Supabase storage
        response = supabase.table("emails3") \
                            .select("*") \
                            .filter("user_email", "eq", EMAIL_USER) \
                            .filter("date", "eq", today) \
                            .order("email_id", desc=True) \
                            .execute()
        
        if response.data:
            return {"emails": response.data}
        else:
            return {"emails": [], "message": "No emails found for today."}
    
    except Exception as e:
        return {"error": str(e), "message": "Failed to fetch emails from Supabase."}



@app.get("/displayimportantemailsinbox")
async def display_important_emails():
    """
    Fetch all emails from Supabase for the logged-in user and today's date.
    """
    try:
        today = datetime.today().strftime("%d-%b-%Y")  # Ensure date format matches Supabase storage
        response = supabase.table("emails3") \
                            .select("*") \
                            .filter("user_email", "eq", EMAIL_USER) \
                            .filter("date", "eq", today) \
                            .filter("is_important","eq",1) \
                            .order("email_id", desc=True) \
                            .execute()
        
        if response.data:
            return {"emails": response.data}
        else:
            return {"emails": [], "message": "No emails found for today."}
    
    except Exception as e:
        return {"error": str(e), "message": "Failed to fetch emails from Supabase."}
 
@app.get("/sendautomaticreplies")
async def send_replies_endpoint():
    try:
        result = send_automated_replies()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=port)