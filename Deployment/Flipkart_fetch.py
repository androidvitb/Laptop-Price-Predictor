import requests
from bs4 import BeautifulSoup
import pandas as pd
from itertools import zip_longest


def fetch_link(prediction):
    min_range = prediction-5000
    max_range = prediction+5000

    url = f"https://www.flipkart.com/search?q=Laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&p%5B%5D=facets.price_range.from%3D{min_range}&p%5B%5D=facets.price_range.to%3D{max_range}"
    return url

def get_flipkart_laptop(prediction):
    url=fetch_link(prediction)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "lxml")
        laptops_list = []
        price_list=[]
        Description_list=[]
        reviews_list=[]
        links_list = []
    
        products = soup.find_all("a", class_="CGtC98")
        name=soup.find_all("div",class_="KzDlHZ")
        price=soup.find_all("div",class_="Nx9bqj _4b5DiR")
        Description=soup.find_all("div",class_="_6NESgJ")
        reviews=soup.find_all("div",class_="_5OesEi")
        
        base_url = "https://www.flipkart.com"
        for product in products:
            laptops_list.append(product.get_text(strip=True))
            product_link = base_url + product.get("href")  # Extract relative link and append base URL
            links_list.append(product_link)
        for i in name:
            name=i.text
            laptops_list.append(name)
        for i in price:
            name=i.text
            price_list.append(name)
        for i in Description:
            name=i.text
            Description_list.append(name)
        for i in reviews:
            name=i.text
            reviews_list.append(name)

        df = pd.DataFrame(list(zip_longest(laptops_list,price_list,Description_list,reviews_list,links_list)), columns=["laptops_list", "price_list", "Description_list","reviews_list","links_list"])
        return df
    else:
        print("Failed to retrieve data. Status Code:", response.status_code)
    














