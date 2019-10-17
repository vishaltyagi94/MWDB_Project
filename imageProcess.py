"""
Authors: Kovid, Tharun, Vishal, Anh, Dhriti, Rinku
Last Edited By: Vishal
Last Edited On: 10/14/2019
Class Description: Class to Extract Features from images
"""
import csv
import glob
import os

import cv2
import matplotlib.image as mpimg
# Import statements
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import skew
from skimage import feature
from skimage.transform import downscale_local_mean
import csv

class imageProcess:
    def __init__(self, dirpath='C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\', ext='*.jpg'):

        self.dirpath = dirpath
        self.ext = ext

    # Method to fetch images as pixels
    def fetchImagesAsPix(self, filename):
        image = cv2.imread(filename)
        size = image.shape
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return img_yuv, size

    # Method to calculate the moments
    def calMommets(self, calc):
        calc = np.array([x for y in calc for x in y])
        mean = np.mean(calc, axis=0)
        sd = np.std(calc, axis=0)
        skw = skew(calc, axis=0)
        mom = [mean.tolist(), sd.tolist(), skw.tolist()]
        mom = [x for y in mom for x in y]
        return mom

    # Method to split image into 100*100 blocks
    def imageMoments(self, image, size, x=100, y=100):
        momments = []
        for idx1 in range(0, size[0], x):
            for idx2 in range(0, size[1], y):
                window = image[idx1:idx1 + x, idx2:idx2 + y]
                momments.append(self.calMommets(window.tolist()))
        return momments

    # Function to calculate the N SIFT feature vectors for each image
    def sift_features(self, filepath):
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return des

    # Function to Calculate the HOG of an image
    def hog_process(self, filename):
        image = cv2.imread(filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dsimg = downscale_local_mean(img, (10, 10))
        (H, hogImage) = feature.hog(dsimg, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), block_norm="L2-Hys",
                                    visualize=True)
        return H

    # Function to calculate the local binary pattern of the window
    def calculate_lbp(self, curr_window):
        eps = 1e-7
        hist = []
        # Initializing LBP settings - radius and number of points
        radius = 3
        num_of_points = 8 * radius
        # Checking for uniform patterns
        window_lbp = feature.local_binary_pattern(curr_window, num_of_points, radius, method='uniform')
        # Generating the histogram
        (histogram, temp) = np.histogram(window_lbp.ravel(),
                                         bins=np.arange(0, num_of_points + 3),
                                         range=(0, num_of_points + 2))
        # Typecasting histogram type to float
        histogram = histogram.astype('float')
        # Normalizing the histogram such that sum = 1
        histogram /= (histogram.sum() + eps)
        hist.append(histogram)
        return hist

    # Function to pre-process images into grayscale and form windows of 100X100 to be fed to calculate_lbp
    def lbp_preprocess(self, filename):
        local_binary_pattern = []
        # Converting the BGR image to Grayscale
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
        for i in range(0, gray.shape[0], 100):
            j = 0
            while j < gray.shape[1]:
                current_window = gray[i:i + 99, j:j + 99]
                temp_lbp = self.calculate_lbp(current_window)
                local_binary_pattern.extend(temp_lbp)
                j = j + 100

        local_binary_pattern = [x for y in local_binary_pattern for x in y]
        local_binary_pattern = np.asarray(local_binary_pattern, dtype=float).tolist()

        return local_binary_pattern

    """
    Method to Save feature data to Postgres Database
    1. Sift: imagedata_sift(imageid, data)
    2. Moments: imagedata_moments(imageid, data)
    3. Hog: imagedata_hog(imageid, data)
    4. LBP: imagedata_lbp(imageid, data)    
    """
    def dbSave(self, conn, model):
        # Count the number of files in the directory
        filecnt = len(glob.glob(self.dirpath + self.ext))
        pbar = tqdm.tqdm(total=filecnt)
        # Read images from the directory
        for filename in glob.glob(self.dirpath + self.ext):
            if model == 'm':
                pixels, size = self.fetchImagesAsPix(filename)
                momments = self.imageMoments(pixels, size)
                # Convert to string to insert into DB as an array
                values_st = str(momments).replace('[', '{')
                values_st = values_st.replace(']', '}')
                dbname = 'imagedata_moments'
            elif model == 's':
                des = self.sift_features(filename)
                values_st = str(des.tolist()).replace('[', '{')
                values_st = values_st.replace(']', '}')
                dbname = 'imagedata_sift'
            elif model == 'h':
                h_val = self.hog_process(filename)
                values_st = str(h_val.tolist()).replace('[', '{')
                values_st = values_st.replace(']', '}')
                dbname = 'imagedata_hog'
            elif model == 'l':
                lbp_val = self.lbp_preprocess(filename)
                values_st = str(lbp_val).replace('[', '{')
                values_st = values_st.replace(']', '}')
                dbname = 'imagedata_lbp'
            else:
                print('Incorrect value for Model provided')
                exit()
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            # create a cursor
            cur = conn.cursor()
            sql = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=str(name),
                                                                  y=values_st,
                                                                  db=dbname)
            cur.execute(sql)
            conn.commit()
            # close cursor
            cur.close
            pbar.update(1)

    # Method to fetch data from Database
    def dbFetch(self, conn, model, db=''):
        # Create cursor
        cur = conn.cursor()
        if model == 's':
            dbname = 'imagedata_sift'
        elif model == 'm':
            dbname = 'imagedata_moments'
        elif model == 'h':
            dbname = 'imagedata_hog'
        elif model == 'l':
            dbname = 'imagedata_lbp'
        elif db != '':
            dbname = db

        sql = "SELECT * FROM {db}".format(db=dbname)
        cur.execute(sql)
        recs = cur.fetchall()
        return recs

    # Method to access the database
    def dbProcess(self, password, process='s', model='s', host='localhost',
                  database='mwdb_project', user='postgres', port=5432):
        # Connect to the database
        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        # Save Features
        if process == 's':
            self.dbSave(conn, model)
            print('Data saved successfully to the Database!')

        elif process == 'f':
            recs = self.dbFetch(conn,model)
            recs_flt = []
            # Flatten the data structure and 
            if model == 'm':
                for rec in recs:
                    recs_flt.append((rec[0], [float(x) for y in rec[1] for x in y]))
            elif model == 's':
                for rec in recs:
                    recs_flt.append((rec[0], [[float(x) for x in y] for y in rec[1]]))
            elif model == 'l' or model == 'h':
                for rec in recs:
                    recs_flt.append((rec[0], [float(x) for x in rec[1]]))
            return recs_flt

    # Method to calculate the Cosine Similarity
    def cosine_sim(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        cos = 1 - dot_product / (norm_a * norm_b)
        return cos
        # return 1 - spatial.distance.cosine(vec1, vec2)

    # method to calculate Manhattan distance
    def man_dist(self, vec1, vec2):
        dist = [abs(x - y) for x,y in zip(vec1, vec2)]
        return sum(dist)

    # Calculate the L2 distance
    def l2Dist(self, d1, d2):
        d1 = np.array(d1, dtype=np.float32)
        d2 = np.array(d2, dtype=np.float32)
        dist = cv2.norm(d1, d2, cv2.NORM_L2)
        return dist

    # Calculate the vector matches
    def knnMatch(self, d1, d2, k=2):
        distances = []
        for d in d1:
            dis = sorted([self.l2Dist(d, x) for x in d2])
            distances.append(dis[0:k])
        return distances

    # Method to calculate Similarity for SIFT vectors
    def sift_sim(self, d1, d2):
        matches = self.knnMatch(d1, d2, k=2)
        good = []
        all = []
        d1 = np.array(d1, dtype=np.float32)
        for m, n in matches:
            all.append(m)
            if m < 0.8 * n:
                good.append(m)
        return len(good) / d1.shape[0]

    # Method to calculate Similarity
    def SimCalc(self, img, recs, imgmodel='m', k=5):
        # Calculate the Similarity matrix for Moments model
        rec_dict = dict((x, y) for x, y in recs)
        img_vec = rec_dict[img]
        if imgmodel == 'm':
            sim_matrix = sorted([(rec[0], self.cosine_sim(img_vec, rec[1])) for rec in recs
                                if rec[0] != img], key=lambda x: x[1])
        if imgmodel == 's':
            sim_matrix = sorted([(rec[0], self.sift_sim(img_vec, rec[1])) for rec in recs
                                if rec[0] != img], key=lambda x: x[1], reverse=True)
        return sim_matrix[0:k]

    # Method to display images
    def dispImages(self, images, savepath):
        no_images = len(images)
        columns = 3
        rows = no_images // columns
        if no_images % columns != 0:
            rows += 1
        ax = []
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Similar Images for Image {x}'.format(x=images[0]))
        plt.axis('off')
        for idx, i in enumerate(images):
            img = mpimg.imread(self.dirpath + self.ext.replace('*', i))
            ax.append(fig.add_subplot(rows, columns, idx + 1))
            if idx == 0:
                ax[-1].set_title("Original Image: " + i)  # set title
            else:
                ax[-1].set_title("Similar Image " + str(idx) + ":" + i)  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig(savepath)
        plt.show()

    # Method to write to a file
    def writeFile(self, content, path):
        with open(path, 'w') as file:
            file.write(str(content))

    # Method to read the Metadata
    def readMetaData(self, filepath='C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\HandInfo.csv'):
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            meta_file = []
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                sub_id = row[0]
                id = row[7].split('.')[0]
                gender = row[2]
                orientation = row[6].split(' ')
                accessories = row[4]
                meta_file.append([sub_id, id, gender, orientation[0], orientation[1], accessories])
            return meta_file

    # Convert list to string
    def list2string(self, lst):
        values_st = str(lst).replace('[[', '(')
        values_st = values_st.replace('[', '(')
        values_st = values_st.replace(']]', ']')
        values_st = values_st.replace(']', ')')
        return values_st

    # Method to create and insert the Metadata to the database
    def createInsertMeta(self, conn):
        # Read the metadata file
        metafile = self.readMetaData()
        # Create cursor
        cur = conn.cursor()
        # Create the meta table
        sqlc = "CREATE TABLE IF NOT EXISTS " \
               "img_meta(subjectid TEXT, image_id TEXT, gender TEXT, aspect TEXT, orient TEXT, accessories TEXT)"
        cur.execute(sqlc)
        conn.commit()
        # Insert the meta data into the database table
        values_st = self.list2string(metafile)
        sqli = "INSERT INTO img_meta VALUES {x}".format(x=values_st)
        cur.execute(sqli)
        conn.commit()
        print('Meta Data saved into Database!')
        cur.close()


