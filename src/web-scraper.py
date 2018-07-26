import cv2
import os
from bs4 import BeautifulSoup
import requests
from datetime import datetime

# Method to create a folder at the given directory (the folder name is in the directory)
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error creating folder")

# Where the images will be stored
download_directory = "/Users/kippc/Downloads/NN_Dataset/"
# What it will scrape from Google images
queries = ["Credit card", "Apple"]
# Number of images to scrape for each query
num_images = 100

start_time = datetime.now()
for q in queries:
    counter = 0
    print("Created folder " + q + " in " + download_directory)
    folder_directory = os.path.join(download_directory, q)
    create_folder(folder_directory)
    while counter < num_images:
        # url = "https://www.bing.com/images/async?q=" + str(q) + "&first=" + str(counter) + "&count=10&mmasync=1"
        url = "https://www.google.com/search?q=" + str(q) + "&safe=active&sout=1&tbm=isch&start=" + str(counter) + "&sa=N"
        html = requests.get(url)
        soup = BeautifulSoup(html.text, "html5lib")
        soup = soup.select("img[src^=http]")

        for i in soup:
            counter += 1
            # print(i.get("src"))
            img_url = i.get("src")
            full_directory = os.path.join(folder_directory, "image" + str(counter) + ".png") 
            with open(full_directory, "wb") as f:
                f.write(requests.get(img_url).content)
        print(counter)

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
print("Total time:", total_time)