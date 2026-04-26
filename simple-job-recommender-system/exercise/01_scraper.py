from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import time

# YANG DI DALAM KURUNG ISI SENDIRI

file_path = "./data/jobstreet_data_{TEMPAT}_{NAMA_JOB}.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(file_path):
    pd.DataFrame(columns=["role","company","description","link"]).to_csv(file_path, index=False)
    
options = Options()
options.binary_location = "/usr/bin/google-chrome"
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)
 
search_url = "https://id.jobstreet.com/id/{NAMA_JOB}-jobs/in-{TEMPAT}"
driver.get(search_url)

time.sleep(5)

job_links = set()

while len(job_links) < 100:
    print(f"Collected links: {len(job_links)}")

    jobs = driver.find_elements(By.XPATH, "//a[contains(@href, '/job/')]")

    for job in jobs:
        link = job.get_attribute("href")
        if link and "/job/" in link:
            job_links.add(link)

    if len(job_links) >= 100:
        break

    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        next_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//span[text()='Selanjutnya']/ancestor::a")
            )
        )
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(5)

    except:
        print("No more pages found")
        break

print(f"Total job links collected: {len(job_links)}")

job_links = list(job_links)[:100]

data = []

for idx, link in enumerate(job_links, start=1):
    try:
        driver.get(link)

        wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )

        role = driver.find_element(By.TAG_NAME, "h1").text

        try:
            company = driver.find_element(
                By.CSS_SELECTOR,
                "span[data-automation='advertiser-name']"
            ).text
        except:
            company = "Unknown Company"

        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[data-automation='jobAdDetails']")
            )
        )

        description = driver.find_element(
            By.CSS_SELECTOR,
            "div[data-automation='jobAdDetails']"
        ).text

        data.append({
            "role": role,
            "company": company,
            "description": description,
            "link": link
        })
        
        pd.DataFrame([{
            "role": role,
            "company": company,
            "description": description,
            "link": link
        }]).to_csv(file_path, mode="a", header=False, index=False)
        
        print("="*60)
        print(f"SUCCESS SCRAPE JOB #{idx}")
        print("ROLE:", role)
        print("COMPANY:", company)
        print("DESCRIPTION PREVIEW:", description[:100])
        print("="*60)

        print(f"SUCCESS #{idx}")

        time.sleep(3)

    except Exception as e:
        print(f"FAILED #{idx} | {link}")
        print(e)
        continue

driver.quit()

print("\nSaved to:", file_path)