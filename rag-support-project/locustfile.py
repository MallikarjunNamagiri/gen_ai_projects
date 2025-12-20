from locust import HttpUser, task, between
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

 
class ChatUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def chat(self):
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.client.post("/api/chat",
            json={"query": "How do I reset password?"},
            headers={"Authorization": f"Bearer {bool(supabase_key)}"})