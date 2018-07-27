import cv2
import os
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import time
import numpy as np

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
queries = ["City", "Ocean", "Desert", "Forest", "Grassland"]
# Number of images to scrape for each query
num_images = 5000

# Go through each query, make a folder in the directory and fill it with the first (num_images) results from Google Images
start_time = datetime.now()
for q in queries:
    counter = 0
    bad_images = 0
    print("Created folder " + q + " in " + download_directory)
    folder_directory = os.path.join(download_directory, q)
    create_folder(folder_directory)
    while counter < num_images:
        if counter % 260 == 0 and counter != 0:
            print("Sleep for 1 minute")
            time.sleep(60)
        url = "https://www.google.com/search?q=" + str(q) + "&safe=active&sout=1&tbm=isch&start=" + str(counter) + "&sa=N"
        html = requests.get(url)
        soup = BeautifulSoup(html.text, "html5lib")
        soup = soup.select("img[src^=http]")

        for i in soup:
            counter += 1
            img_url = i.get("src")
            full_directory = os.path.join(folder_directory, "image" + str(counter) + ".png")
            with open(full_directory, "wb") as f:
                f.write(requests.get(img_url).content)

            image = cv2.cv2.imread(full_directory)
            height = np.size(image, 0)
            width = np.size(image, 1)
            if height < 50 or width < 50:
                bad_images += bad_images
                os.remove(full_directory)
                print("Dimensions too small - deleted")
        print(counter)
    print(bad_images, "images removed")

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
print("Scraped {0} images in {1:.2f} seconds".format(len(queries)*num_images, total_time))