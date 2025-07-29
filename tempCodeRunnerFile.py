from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env
print(os.getenv("GOOGLE_API_KEY"))  # This should print your key