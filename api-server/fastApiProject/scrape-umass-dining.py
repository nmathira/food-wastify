from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import requests


def main():
    items = {

    }
    html = requests.get("https://umassdining.com/locations-menus/worcester/menu")

    # Check if the request was successful (status code 200)
    if html.status_code == 200:
        # Get the HTML content of the page
        html_content = html.text

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        lunch_div = soup.find("div", id="lunch_menu")

        list_items = lunch_div.find_all('li')

        # print(lunch_div.prettify())
        for item in list_items:
            a_tag = item.findNext("a")
            items[a_tag.getText()] = a_tag["data-carbon-list"]
    else:
        print(f"Failed to retrieve page. Status code: {html.status_code}")

    print(items)


if __name__ == '__main__':
    main()
