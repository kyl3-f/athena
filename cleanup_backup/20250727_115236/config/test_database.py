# Save as test_database.py
from settings import setup_database, db_config

print("🗄️ Testing database setup...")
print(f"Database type: {db_config.database_type}")
print(f"Database path: {db_config.sqlite_db_path}")

try:
    engine = setup_database()
    print("✅ Database setup successful!")
    print(f"\nDBeaver connection info:")
    print(f"Type: SQLite")
    print(f"Path: {db_config.sqlite_db_path}")
except Exception as e:
    print(f"❌ Database setup failed: {e}")