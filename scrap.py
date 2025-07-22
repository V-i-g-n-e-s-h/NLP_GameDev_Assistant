import json
import re
import sys
import time
from datetime import datetime
import uuid
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm


OUTPUT_FILE = Path("unity_docs.jsonl")
HEADLESS = True 
TIMEOUT = 15


def init_driver(headless: bool = True) -> webdriver.Chrome:
    """Return a ready-to-use Chrome WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    return webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=chrome_options,
    )


def clean_text(raw: str) -> str:
    """Collapse whitespace but keep paragraph breaks."""
    cleaned = re.sub(r"\n{2,}", "\n", raw)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def extract_main_text(driver: webdriver.Chrome) -> str:
    selectors = ["main", "article", ".container", "body"]
    for css in selectors:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, css)
            if elem and elem.text.strip():
                return clean_text(elem.text)
        except Exception:
            continue
    return ""


def scrape_one(url: str, driver: webdriver.Chrome) -> dict:
    driver.get(url)
    WebDriverWait(driver, TIMEOUT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    return {
        "id": str(uuid.uuid4()),
        "url": url,
        "title": (driver.title or "").strip(),
        "content": extract_main_text(driver),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    urls = []
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    driver = init_driver(HEADLESS)
    
    driver.get(url="https://docs.unity3d.com/Manual/index.html")
    time.sleep(3)
    plus_xpath = '//*[@id="customScrollbar"]/ul/li/div[@class="arrow collapsed"]/following-sibling::a' # top level links
    for ele in driver.find_elements(By.XPATH, plus_xpath):
        urls.append(ele.get_attribute("href"))
        time.sleep(0.3)

    written = 0
    try:
        with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
            for url in tqdm(urls, desc="Scraping", unit="page"):
                try:
                    doc = scrape_one(url, driver)
                    if doc["content"]:
                        json.dump(doc, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        written += 1
                except Exception as e:
                    print(f"[WARN] Failed to scrape {url}: {e}", file=sys.stderr)
    finally:
        driver.quit()

    print(f"Saved {written} / {len(urls)} documents to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
