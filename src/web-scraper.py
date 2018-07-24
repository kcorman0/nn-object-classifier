import cv2
import os
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from os.path import basename

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error creating folder")

download_directory = "/Users/kippc/Downloads/NN_Dataset/"
queries = ["Test", "Superhot"]
num_images = 10
start_image = 1

start_time = datetime.now()
for i in queries:
    url = "https://www.bing.com/images/async?q=" + i + "&first=3&count=100&mmasync=1"
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html5lib")
    soup = soup.select("img[src^=http]")

    for i in soup:
        print(i.get("src"))

        img_url = i.get("src")
        # with open(basename(img_url), "wb") as f:
            # print("W")
            # f.write(requests.get(img_url).content)

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
print("Total time:", total_time)