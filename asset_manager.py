import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

ASSET_DIR = "assets"
if not os.path.exists(ASSET_DIR):
    os.makedirs(ASSET_DIR)

def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Downloaded asset: {local_filename}")
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None

def get_audio_asset(api_key, query, media_type):
    if not api_key:
        logging.error(f"Pixabay API key not found for {media_type}.")
        return None
    try:
        params = {
            "key": api_key,
            "q": query,
            "media_type": media_type,
            "per_page": 3
        }
        url = "https://pixabay.com/api/"
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if not results or not results.get("hits"):
            logging.warning(f"No {media_type} found for query: '{query}'")
            return None
        
        item = results["hits"][0]
        download_url = item.get('url') # For audio, the URL is often at the top level
        if not download_url:
            download_url = item.get('downloads', {}).get('mp3', {}).get('url')

        if not download_url:
            logging.error(f"Could not find download URL for {media_type} ID {item['id']}")
            return None

        file_name = f"{media_type}_{query.replace(' ', '_')}_{item['id']}.mp3"
        local_path = os.path.join(ASSET_DIR, file_name)
        return download_file(download_url, local_path)

    except Exception as e:
        logging.error(f"Failed to get {media_type} from Pixabay: {e}")
        return None

def get_sfx(effect_name):
    return get_audio_asset(PIXABAY_API_KEY, f"{effect_name} sound", "audio")

def get_music(mood):
    return get_audio_asset(PIXABAY_API_KEY, mood, "music")

def get_b_roll(keyword, orientation="portrait"):
    # Pexels for video has a different API structure
    if not PEXELS_API_KEY:
        logging.error("Pexels API key not found.")
        return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": keyword, "per_page": 1, "orientation": orientation}
        url = "https://api.pexels.com/videos/search"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        videos = response.json().get('videos', [])
        if not videos:
            logging.warning(f"No B-roll found for keyword: '{keyword}'")
            return None
        video = videos[0]
        video_link = next((vf['link'] for vf in video['video_files'] if vf.get('quality') == 'hd'), video['video_files'][0]['link'])
        file_name = f"broll_{keyword.replace(' ', '_')}_{video['id']}.mp4"
        local_path = os.path.join(ASSET_DIR, file_name)
        return download_file(video_link, local_path)
    except Exception as e:
        logging.error(f"Failed to get B-roll from Pexels: {e}")
        return None