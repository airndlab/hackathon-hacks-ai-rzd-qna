import os

import yaml

from app.main import Profile

PROFILES_FILE_PATH = os.getenv('PROFILES_FILE_PATH')
with open(PROFILES_FILE_PATH, 'r', encoding='utf-8') as file:
    PROFILES = yaml.safe_load(file)


async def get_profiles():
    return [
        Profile(id=key, text=", ".join(value.values()))
        for key, value in PROFILES.items()
    ]
