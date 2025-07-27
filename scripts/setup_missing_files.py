# scripts/setup_missing_files.py
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_missing_essential_files():
    """Create missing essential files for production"""
    
    project_root = Path(__file__).parent.parent
    
    # Create .env.example
    env_example = project_root / "config" / ".env.example"
    if not env_example.exists():
        env_content = """# Athena Trading System Environment Variables

# Polygon.io API Configuration
POLYGON_API_KEY=your_polygon_api_key_here

# Database Configuration
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=data/athena.db

# PostgreSQL Configuration (for production)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=athena
POSTGRES_USER=athena
POSTGRES_PASSWORD=your_password_here

# Email Alerts Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_RECIPIENTS=alert1@email.com,alert2@email.com

# Trading Configuration
ATHENA_ENV=local
"""
        with open(env_example, 'w') as f:
            f.write(env_content)
        logger.info(f"Created {env_example}")
    
    # Create LICENSE
    license_file = project_root / "LICENSE"
    if not license_file.exists():
        license_content = """MIT License

Copyright (c) 2025 Athena Trading System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        with open(license_file, 'w') as f:
            f.write(license_content)
        logger.info(f"Created {license_file}")
    
    # Ensure data directories exist
    directories = [
        "data/silver",
        "data/bronze",
        "data/finviz",
        "logs",
        "src",
        "src/ingestion", 
        "src/processing"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {full_path}")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/ingestion/__init__.py",
        "src/processing/__init__.py",
        "config/__init__.py"
    ]
    
    for init_file in init_files:
        full_path = project_root / init_file
        if not full_path.exists():
            with open(full_path, 'w') as f:
                f.write('"""Athena Trading System"""\n')
            logger.info(f"Created {full_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_missing_essential_files()
    print("✅ Missing essential files created!")