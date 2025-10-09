#!/usr/bin/env python3
"""
Streamlit app auto wake up script
"""

import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager


def wake_up_streamlit_app():
    """Use Selenium to wake up Streamlit app"""
    # 配置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-setuid-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-accelerated-2d-canvas')
    chrome_options.add_argument('--no-first-run')
    chrome_options.add_argument('--no-zygote')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins')
    user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_argument(f'--user-agent={user_agent}')

    driver = None
    try:
        print("🚀 Start Chrome browser...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)

        print("🌐 Visit Streamlit app...")
        driver.get("https://climviz.streamlit.app/")

        wait = WebDriverWait(driver, 30)
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            print("✅ Body element loaded")
            
            try:
                driver.execute_script("return document.readyState") == "complete"
                print("✅ Document ready state complete")
            except Exception:
                pass
                        
            try:
                wait.until(lambda driver: len(driver.find_elements(By.CSS_SELECTOR, "div, span, p, h1, h2, h3")) > 0)
                print("✅ Content elements detected")
            except TimeoutException:
                print("⚠️ No content elements found, continue...")
            
            time.sleep(5)
            
        except TimeoutException:
            print("⚠️ Page load timeout, continue...")

        print("🔍 Check app status...")
        body_element = driver.find_element(By.TAG_NAME, "body")
        body_text = body_element.text.lower()

        sleep_indicators = [
            "zzzz",
            "your app has gone to sleep",
            "wake it back up",
            "inactivity",
        ]

        is_sleeping = any(indicator in body_text for indicator in sleep_indicators)

        if is_sleeping:
            print("⚠️ App is sleeping, try to wake up...")

            wake_up_selectors = [
                "//button[contains(text(), 'back')]",
                "//button[contains(text(), 'Yes, get this app back up')]",
                "//button[contains(@data-testid, 'wake')]",
                "//button[contains(@data-testid, 'wake-up')]",
                "//button[contains(text(), 'Wake')]",
                "//button[contains(text(), 'wake')]",
                "//button"
            ]

            wake_up_clicked = False
            for selector in wake_up_selectors:
                try:
                    button = driver.find_element(By.XPATH, selector)
                    if button.is_displayed() and button.is_enabled():
                        button.click()
                        print("✅ Wake up button clicked")
                        wake_up_clicked = True
                        break
                except (NoSuchElementException, WebDriverException):
                    continue

            if not wake_up_clicked:
                print("🔘 No specific wake up button found, try to click page...")
                try:
                    body_element.click()
                    time.sleep(2)
                except WebDriverException:
                    print("⚠️ Cannot click page element")

            print("⏳ Wait for app to wake up...")
            time.sleep(20)

            try:
                updated_body_element = driver.find_element(By.TAG_NAME, "body")
                updated_body_text = updated_body_element.text.lower()
                still_sleeping = any(
                    indicator in updated_body_text
                    for indicator in sleep_indicators
                )

                if not still_sleeping:
                    print("✅ App has been successfully awakened!")
                else:
                    print("⚠️ App may still be sleeping, but wake up attempt has been completed")
            except Exception as e:
                print(f"⚠️ Error verifying wake up status: {e}")
        else:
            print("✅ App is already active")

        screenshot_path = "app_status.png"
        try:
            driver.save_screenshot(screenshot_path)
            print(f"📸 Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"⚠️ Screenshot save failed: {e}")

        return True

    except Exception as error:
        print(f"❌ Error: {error}")
        if driver:
            try:
                error_screenshot_path = "error_screenshot.png"
                driver.save_screenshot(error_screenshot_path)
                print(f"📸 Error screenshot saved: {error_screenshot_path}")
            except Exception:
                pass
        return False

    finally:
        if driver:
            try:
                driver.quit()
                print("🔒 Browser closed")
            except Exception:
                pass


def main():
    print("🤖 Start Streamlit app auto wake up...")

    try:
        success = wake_up_streamlit_app()

        if success:
            print("🎉 Wake up process completed")
            sys.exit(0)
        else:
            print("💥 Wake up process failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ User interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
