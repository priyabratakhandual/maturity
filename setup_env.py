#!/usr/bin/env python3
"""
Helper script to create .env file from env.example
"""
import os
import shutil

def setup_env():
    """Create .env file from env.example if it doesn't exist"""
    env_example = 'env.example'
    env_file = '.env'
    
    if os.path.exists(env_file):
        print(f"⚠ {env_file} already exists. Skipping creation.")
        print(f"   Please edit {env_file} manually to add your OpenAI API key.")
        return
    
    if not os.path.exists(env_example):
        print(f"❌ {env_example} not found. Cannot create .env file.")
        return
    
    try:
        shutil.copy(env_example, env_file)
        print(f"✅ Created {env_file} from {env_example}")
        print(f"\n📝 Next steps:")
        print(f"   1. Edit {env_file}")
        print(f"   2. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
        print(f"   3. Get your API key from: https://platform.openai.com/api-keys")
    except Exception as e:
        print(f"❌ Error creating {env_file}: {e}")

if __name__ == '__main__':
    setup_env()

