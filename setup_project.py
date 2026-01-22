#!/usr/bin/env python3
"""
Netflix Stock Volatility Forecasting - Project Setup Script
Sets up the environment and runs initial validation
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'logs',
        'artifacts',
        'docs/reports',
        'models/backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def run_initial_validation():
    """Run project validation"""
    print("\n🔍 Running initial validation...")
    try:
        result = subprocess.run([sys.executable, 'validate_project.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Project validation passed!")
            return True
        else:
            print("❌ Project validation failed!")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def display_next_steps():
    """Display what to do next"""
    print("\n🚀 SETUP COMPLETE!")
    print("=" * 50)
    print("📋 NEXT STEPS:")
    print("1. Run the complete pipeline:")
    print("   python scripts/run_pipeline.py")
    print("\n2. Run EDA analysis:")
    print("   python analysis/run_eda_analysis.py")
    print("\n3. Run modeling analysis:")
    print("   python analysis/run_modeling_analysis.py")
    print("\n4. Explore notebooks:")
    print("   jupyter notebook notebooks/eda/Data_Understanding_2.ipynb")
    print("\n5. Validate project anytime:")
    print("   python validate_project.py")
    print("\n📊 Generated outputs will be in analysis/outputs/")
    print("🤖 Trained models will be in models/")
    print("📋 Metadata will be in data/processed/")

def main():
    """Main setup function"""
    print("🎬 Netflix Stock Volatility Forecasting - Project Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Run validation
    if not run_initial_validation():
        return False
    
    # Display next steps
    display_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
