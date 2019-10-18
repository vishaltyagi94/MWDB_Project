"""
Authors: Kovid, Tharun, Vishal, Anh, Dhriti, Rinku
Last Edited By: Kovid
Last Edited On: 9/22/2019
Class Description: Class to Extract Features from images
"""
# Import statements
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from scipy.stats import skew
from PostgresDB import PostgresDB
import tqdm
import os
import cv2
from skimage import feature
from skimage.transform import downscale_local_mean
from scipy.linalg import svd
from scipy.sparse.linalg import svds
# import time
import math
import joblib

# Task 3 4 5
import csv
import matplotlib.pyplot as plt
import copy
import os

DATABASE_NAME = 'mwdb'
TABLE_NAME = 'images_demo'
PASSWORD = "mynhandepg"
# dirpath='/home/anhnguyen/ASU/CSE-515/Project/Phase 1/Project - Phase 2/Data/testset1/'
# ext='*.jpg'
csvFile = "HandInfo.csv"




class imageProcess:
    def __init__(self, dirpath, ext='*.jpg'):
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
    1. Sift: imagedata_s(imageid, data)
    2. Moments: imagedata_m(imageid, data)
    3. Hog: imagedata_h(imageid, data)
    4. LBP: imagedata_l(imageid, data)    
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
                values_st = str(np.asarray(momments).tolist())
                # values_st = str(momments).replace('[', '{')
                # values_st = values_st.replace(']', '}')
                dbname = 'imagedata_m'
            elif model == 's':
                des = self.sift_features(filename)
                values_st = str(np.asarray(des).tolist())
                # values_st = str(des.tolist()).replace('[', '{')
                # values_st = values_st.replace(']', '}')
                dbname = 'imagedata_s'
            elif model == 'h':
                h_val = self.hog_process(filename)
                values_st = str(np.asarray(h_val).tolist())
                # values_st = str(h_val.tolist()).replace('[', '{')
                # values_st = values_st.replace(']', '}')
                dbname = 'imagedata_h'
            elif model == 'l':
                lbp_val = self.lbp_preprocess(filename)
                values_st = str(np.asarray(lbpdbase_val).tolist())
                # values_st = str(lbp_val.tolist()).replace('[', '{')
                # values_st = values_st.replace(']', '}')
                dbname = 'imagedata_l'
            else:
                print('Incorrect value for Model provided')
                exit()
            sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(db=dbname)
            cur = conn.cursor()
            cur.execute(sql)
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            # create a cursor
            sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid",db=dbname,condition=name)
            # print("SQL Check Exist - HOG: ", sql)
            cur.execute(sql)

            # cur.execute(sql)
            if cur.fetchone() is None:
                print("Insert")
                # print("Not Exist HOG - Insert")
                sql = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=name,y=values_st, db=dbname)
            else:
                print("Update")
                # print("Exist HOG - Update")
                # column = "HOG"
                
                sql = "UPDATE {db} SET {x} ='{y}' WHERE IMAGEID = '{id}'".format(x=name,y=values_st, db=dbname)
            
            cur.execute(sql)
            conn.commit()
            # close cursor
            cur.close
            pbar.update(1)

    # Method to fetch data from Database
    def dbFetch(self, conn, dbname, condition = ""):
        # Create cursor
        cur = conn.cursor()
        # if model == 's':
        #     dbname = 'imagedata_sift'
        # elif model == 'm':
        #     dbname = 'imagedata_moments'
        # elif model == 'h':
        #     dbname = 'imagedata_hog'
        # elif model == 'l':
        #     dbname = 'imagedata_lbp'
        # dbname = 'imagedata_' + model
        # if condition:
        #     dbname += "_" + technique
        sql = "SELECT * FROM {db} {condition}".format(db=dbname, condition = condition)
        # print (sql)
        cur.execute(sql)
        recs = cur.fetchall()
        return recs

    # Method to access the database
    def dbProcess(self, password, process='s', model='s', host='localhost',
                  database='postgres', user='postgres', port=5432, dbase = 'imagedata_l'):
        # Connect to the database
        db = PostgresDB(password=password, host=host,
                        database=DATABASE_NAME, user=user, port=port)
        conn = db.connect()
        if process == 's':
            self.dbSave(conn, model)
            print('Data saved successfully to the Database!')
        elif process == 'f':
            recs = self.dbFetch(conn,dbase)
            recs_flt = []
            # Flatten the data structure and 
            for rec in recs:
                recs_flt.append((rec[0],np.asarray(eval(rec[1]))))
            # if model == 'm':
            #     print(recs)
            #     for rec in recs:
            #         recs_flt.append(np.asarray(eval(rec[1])))
                    # recs_flt.append((rec[0], [float(x) for y in rec[1] for x in y]))
            # elif model == 's':
            #     for rec in recs:
            #         recs_flt.append((rec[0], [[float(x) for x in y] for y in rec[1]]))
            # elif model == 'l' or model == 'h':
            #     for rec in recs:
            #         recs_flt.append((rec[0], [float(x) for x in rec[1]]))
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

    # Calculate the Euclidean distance
    def euclidean_distance(self, imageA, imageB):
        # d=math.sqrt(np.sum([((a-b) ** 2) for (a,b) in zip(imageA,imageB)]))
        # return d
        return np.sqrt(np.sum((imageA - imageB) ** 2, axis=0))

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


    def queryImageNotLabel(self, image_data, feature, technique, label):
        print("Not Same Label")
        # cursor.execute("SELECT * FROM imagedata_{0}_{1} WHERE imageid = '{2}'".format(feature,technique,image))
        # image_data = cursor.fetchall()
        # print(image_data)
        image_data = np.asarray(eval(image_data[0][1]))
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)

        model = joblib.load(path + os.sep + "{0}_{1}_{2}.joblib".format(feature, technique, label))
        latent = np.asarray(model.components_)
        
        if feature == 's':
            kmeans = joblib.load(path + os.sep + 'kmeans_{0}_{1}.joblib'.format(latent.shape[1], label))
            histo = np.zeros(latent.shape[1])
            nkp = np.size(image_data)
            for d in image_data:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp
        print(np.asarray((model.components_)).shape)
        image_data = np.asarray(histo).dot(latent.T)
        return image_data
        
    def similarity (self, feature, technique, dbase, k, image, label = ""):
        db = PostgresDB(password = "mynhandepg", database = "mwdb")
        conn = db.connect()
        if conn is None:
            print("Can not connect to database")
            exit()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM " + dbase)
        data = cursor.fetchall()
        image_id = [rec[0] for rec in data]
        similarity = {}
        if image in image_id:
            image_index = image_id.index(image)
            print(image_index)
            image_data = np.asarray(eval(data[image_index][1]))
        else:
            print("Not Same Label")
            dbase = 'imagedata_' + feature
            label = label.replace(" ", "_")
            image_data = self.dbFetch(conn,dbase, "WHERE imageid = '{0}'".format(image))
            image_data = self.queryImageNotLabel(image_data, feature, technique, label)
            similarity[image] = self.euclidean_distance(image_data,image_data)
            
        # print (image_id)
        for i in range(len(image_id)):
            image_cmp = np.asarray(eval(data[i][1]))
            # if self.metrics:
            #     # similarity[row[0]] = 1- self.cosine_similarity(image, result)
            #     similarity[image_id[i]] = 1 - st.pearsonr(image,image_cmp)[0]
            #     # similarity[row[0]] = mean_squared_error(image,result)
            #     # similarity[row[0]] = 0 - self.psnr(image,result)
            # else:
            similarity[image_id[i]] = self.euclidean_distance(image_data,image_cmp)
        similarity = sorted(similarity.items(), key = lambda x : x[1], reverse=False)
        print(similarity)
        self.dispImages(similarity,feature, technique, 11, k)

    # Method to display images
    def dispImages(self, similarity, feature, technique, no_images, k):
        columns = 4
        rows = no_images // columns
        if no_images  % columns != 0:
                rows += 1
        ax = []
        fig=plt.figure(figsize=(30, 20))
        fig.canvas.set_window_title('Task 3 - Images Similarity')
        fig.suptitle(str(no_images - 1) + ' Similar Images of ' + similarity[0][0] + ' based on ' + feature + ", "+ str(k) + " latent semantics and " + technique)
        # plt.title(str(no_images - 1) + ' Similar Images of ' + similarity[0][0] + ' based on ' + type,y=-0.01)
        plt.axis('off')
        # fig.title(str(k) + 'Similar Images of ' + similarity[0][0] + ' based on ' + type)
        f= open("../Outputs/task3_result.txt","w+")
        f.write("Task 2 - Matching Score " + str(no_images) + " images with " + similarity[0][0] + ' based on ' + feature + ", "+ str(k) + " latent semantics and " + technique + ":\n")
        for i in range(no_images):
            f.write(similarity[i][0] + ": " + str(similarity[i][1]) + "\n")
            img = mpimg.imread(self.dirpath + self.ext.replace('*', similarity[i][0]))
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, columns, i+1))
            if i == 0:
                    ax[-1].set_title("Given Image: " +similarity[i][0] )  # set title
            else:
                    ax[-1].set_title("Image "+str(i) + ": " +similarity[i][0] )  # set title
            ax[-1].axis('off')
            plt.imshow(img)
        plt.savefig('../Outputs/task3_result.png')
        f.close()
        plt.show()
        plt.close()

    # Method to write to a file
    def writeFile(self, content, path):
        with open(path, 'w') as file:
            file.write(str(content))

    # Convert list to string
    def list2string(self, lst):
        values_st = str(lst).replace('[[', '(')
        values_st = values_st.replace('[', '(')
        values_st = values_st.replace(']]', ']')
        values_st = values_st.replace(']', ')')
        return values_st
    
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
    
    
    def readMetaData(self):
        with open(self.dirpath + csvFile, 'r') as file:
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

    def CSV(self, label = ""):
        label = label.lower()
        if label in ("dorsal", "palmar", "left", "right"):
            index = "aspectOfHand"
        elif label in ("male", "female"):
            index = "gender"
        elif label in ("with accessories", "without accessories"):
            index = "accessories"
        else:
            index = ""

        with open(self.dirpath + csvFile, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            # next(cr) gets the header row (row[0])
            header = next(reader)
            i = header.index(index)
            id = header.index("imageName")
            # print(i,index)
            # list comprehension through remaining cr iterables
            if index in ("aspectOfHand", "gender"):
                filteredImage = [row[id][:len(row[id]) - 4] for row in reader if row[i].find(label) != -1]
            elif label == "with accessories":
                filteredImage = [row[id][:len(row[id]) - 4] for row in reader if int(row[i]) == 0]
            elif label == "without accessories":
                filteredImage = [row[id][:len(row[id]) - 4] for row in reader if int(row[i]) == 1]
            # else:
            #     return data, header
        # print (filteredImage)
        return filteredImage



def cosine_similarity(imageA, imageB):
        # print(imageA)
        # print(imageB)
        return np.dot(imageA, imageB)/(np.sqrt(np.sum(imageA ** 2, axis=0))*np.sqrt(np.sum(imageB ** 2, axis=0)))

