import os
import psycopg2
from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv(override=True)

# # Get database credentials from environment variables
# host = os.getenv("DB_HOST")
# port = os.getenv("DB_PORT")
# username = os.getenv("DB_USER")
# password = os.getenv("DB_PASSWORD")

try:
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host="automation-challenge.clflrheh2sz7.eu-west-1.rds.amazonaws.com",
        port="5432",
        user="adludiochallenge",
        password="abelchallenge"
    )

    print("Connected to the PostgreSQL database!")

    # Close the connection
    conn.close()
except psycopg2.Error as e:
    # Error occurred while connecting
    print("Error connecting to the PostgreSQL database:", e)
