import cv2
import os
import requests
from datetime import datetime
import urllib.request, json

# Creates a folder at the given directory (the folder name is specified in directory)
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error creating folder")

# Where the images will be stored
download_directory = "/Users/kipp/Downloads/NN_Dataset2/"
# The search queries that will be scraped
queries = ["Urban", "Nature"]
# The starting image for each query
starting_points = {"Urban": 0, "Nature": 0}
# Number of images to scrape for each query
num_images = 12000

# Go through each query, make a folder in the directory and fill it with the first (num_images) results from Unsplash
# The Unsplash json pages only display information for 30 images at a time, so the url has to be reloaded every 30 images
start_time = datetime.now()
for q in queries:  
    counter = starting_points.get(q)
    print("Created folder " + q + " in " + download_directory)
    folder_directory = os.path.join(download_directory, q)
    create_folder(folder_directory)
    print(starting_points.get(q))

    while counter < num_images + int(starting_points.get(q)):
        unsplash = "https://unsplash.com/napi/search/photos?query=" + str(q) + "&xp=&per_page=30&page=" + str(1 + (counter / 30))
        with urllib.request.urlopen(unsplash) as url:
            data = json.loads(url.read().decode())

        for i in range(30):
            img_url = data['results'][i]['urls'].get('small') # Download the small verion of each image
            full_directory = os.path.join(folder_directory, "image" + str(counter) + ".png")
            with open(full_directory, "wb") as f:
                f.write(requests.get(img_url).content)
            counter += 1
            print(counter)
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
print("Scraped {0} images in {1:.2f} seconds".format(len(queries)*num_images, total_time))
