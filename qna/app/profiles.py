import json
import os
from typing import List

from app.db import Profile

PROFILES_FILE_PATH = os.getenv('PROFILES_FILE_PATH')
with open(PROFILES_FILE_PATH, 'r', encoding='utf-8') as file:
    profiles_json = json.load(file)
    profiles = {key: Profile(id=key, **value) for key, value in profiles_json.items()}

PROFILES_DEFAULT_ID = os.getenv('PROFILES_DEFAULT_ID', 'young')


async def get_profiles() -> List[Profile]:
    return list(profiles.values())


async def get_profile(profile_id: str) -> Profile:
    return profiles.get(profile_id)


async def get_default_profile() -> Profile:
    return await get_profile(PROFILES_DEFAULT_ID)
