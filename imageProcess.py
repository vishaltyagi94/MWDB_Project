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
from sklearn.cluster import MiniBatchKMeans

# Task 3 4 5
import csv

DATABASE_NAME = 'mwdb'
TABLE_NAME = 'images_demo'
PASSWORD = "abcdef"
dirpath='/home/anhnguyen/ASU/CSE-515/Project/Phase 1/Project - Phase 2/Data/testset1/'
ext='*.jpg'
csvFile = "/home/anhnguyen/ASU/CSE-515/Project/Phase 2/Project - Phase 2/"

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
                values_st = str(lbp_val.tolist()).replace('[', '{')
                values_st = values_st.replace(']', '}')
                dbname = 'imagedata_lbp'
            else:
                print('Incorrect value for Model provided')
                exit()
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            # create a cursor
            cur = conn.cursor()
            sql = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=name,
                                                                  y=values_st,
                                                                  db=dbname)
            cur.execute(sql)
            conn.commit()
            # close cursor
            cur.close
            pbar.update(1)

    # Method to fetch data from Database
    def dbFetch(self, conn, model):
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

        sql = "SELECT * FROM {db}".format(db=dbname)
        cur.execute(sql)
        recs = cur.fetchall()
        return recs

    # Method to access the database
    def dbProcess(self, password, process='s', model='s', host='localhost',
                  database='postgres', user='postgres', port=5432):
        # Connect to the database
        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        if process == 's':
            self.dbSave(conn, model)
            print('Data saved successfully to the Database!')
        elif process == 'f':
            recs = self.dbFetch(conn,model)
            recs_flt = []
            # Flatten the data structure and 
            if model == 'm':
                print(recs)
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



class SVDProcess:
    def __init__(self, k, type='HOG', metrics=True):
            self.k = k
            self.type = type
            self.metrics = metrics
    
    def svd_calc(self,image):
            # number_of_component = 100
            # image = np.reshape(image,(-1,14*4*9))
            # print(image.shape)
            [U, s, Vt] = svd(image)
            # print(Sigma.shape)
            # print(np.diag(Sigma).shape)
            # S = S[:, :number_of_component]
            # VT = VT[:n_component, :]
            print(s.shape)
            # Sigma = np.zeros((image.shape[0], image.shape[1]))
            Sigma = np.diag(s)
            # print(Sigma.shape)
            # image = U[:,:self.k].dot(Sigma[:self.k, :self.k]).dot(V[:self.k,:])
            # print(image.shape)
            return [U,Sigma,Vt]
    
    def svd_latent(self):
            db = PostgresDB(password = PASSWORD, database = DATABASE_NAME)
            conn = db.connect()
            if conn is None:
                    print("Can not connect to database")
                    exit()
            cursor = conn.cursor()
            cursor.execute("select image_id, " + self.type + " from " + TABLE_NAME)
            data = cursor.fetchall()
            matrix = []
            image_id = []
            for row in data:
                    image_cmp = np.asarray(eval(row[1]))
                    image_id.append(row[0])
                    print (image_cmp.shape)
                    if self.type != "SIFT":
                            image_cmp = image_cmp.reshape((-1))
                            matrix.append(image_cmp)
                    else:
                            matrix.extend(image_cmp)   
                    # print (image_cmp.shape)
                    
            matrix = np.asarray(matrix)   
            # print(matrix.shape)
            if self.type == "SIFT":
                    Kmeans = KMeans_SIFT(150)
                    clusters = Kmeans.kmeans_process(matrix)
                    matrix = Kmeans.newMatrixSift(data, clusters)
                    # print (matrix)
                    # print(matrix.shape)
            matrix = self.svd_calc(matrix)
            # print (matrix.shape)
            # print (matrix)
            # print(self.svd_calc(matrix))
            cursor.close()
            conn.close()
            return matrix, image_id
            

    def svd_sim(self,image,no_images,matrix_svd, image_id):
            
        similarity = {}
        
        # print(image.shape)
        matrix_image = matrix_svd[0][:,:self.k].dot(matrix_svd[1][:self.k, :self.k]).dot(matrix_svd[2][:self.k,:])
        image_index = image_id.index(image)
        image = matrix_image[image_index]
        for i in range(len(image_id)):
            image_cmp = matrix_image[i]
            
            if self.metrics:
                    similarity[image_id[i]] = 1- cosine_similarity(image, image_cmp)
                    # similarity[image_id[i]] = 1 - st.pearsonr(image,image_cmp)[0]
                    # similarity[row[0]] = mean_squared_error(image,result)
                    # similarity[row[0]] = 0 - self.psnr(image,result)

            else:
                similarity[image_id[i]] = euclidean_distance(image,image_cmp)
        similarity = sorted(similarity.items(), key = lambda x : x[1], reverse=False)


        print(similarity) 

        
        dispImages(similarity, no_images + 1, self.type,self.metrics)
            # self.showResult(similarity_cos,similarity_cos)

    def svdTask1(self, matrix_SVD, image_id):
        latent = 0
        print ("Data Latent Sementics: " + str(latent))
        for data_latent in matrix_SVD[0].T:
            print ("Latent Sementics: " + str(latent))
            result = {}
            index = data_latent.argsort()[::-1]
            for i in index:
                if image_id[i] not in result:
                    result[image_id[i]] = data_latent[i]
            # print(result)
            latent += 1

        latent = 0
        print ("Feature Latent Sementics: " + str(latent))
        print(matrix_SVD[2].shape)
        # for feature_latent in matrix_SVD[2]:
        #     print ("Latent Sementics: " + str(latent))
        #     result = {}
        #     index = feature_latent.argsort()[::-1]
        #     for i in index:
        #         if "Feature " + str(i) not in result:
        #             result["Feature " + str(i)] = feature_latent[i]
        #     print(result)
        #     latent += 1


def dispImages(similarity , no_images, type, metric):
    if metric:
            metric_text = "Cosine Similarity"
    else:
            metric_text = "Euclidean Distance"
    columns = 4
    rows = no_images // columns
    if no_images  % columns != 0:
            rows += 1
    ax = []
    fig=plt.figure(figsize=(30, 20))
    fig.canvas.set_window_title('Task 3 - Images Similarity')
    fig.suptitle(str(no_images - 1) + ' Similar Images of ' + similarity[0][0] + ' based on ' + type + " and " + metric_text)
    # plt.title(str(no_images - 1) + ' Similar Images of ' + similarity[0][0] + ' based on ' + type,y=-0.01)
    plt.axis('off')
    # fig.title(str(k) + 'Similar Images of ' + similarity[0][0] + ' based on ' + type)
    f= open("../Outputs/task3_result.txt","w+")
    f.write("Task 3 - Matching Score " + str(no_images) + " images with " + similarity[0][0] + ' based on ' + type + " and " + metric_text + ":\n")
    for i in range(no_images):
            f.write(similarity[i][0] + ": " + str(similarity[i][1]) + "\n")
            img = mpimg.imread(dirpath + ext.replace('*', similarity[i][0]))
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

class KMeans_SIFT:
    def __init__(self,k):
        self.k = k

    def kmeans_process(self,matrix_image):
        batch_size = 20 * 3
        kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=batch_size, verbose=1).fit(matrix_image)
        return kmeans

    def newMatrixSift(self,data, kmeans):
        kmeans.verbose = False
        histo_list = []
        # print("AAAAAAAAAAAA")
        # print(data)
        for des in data:
                # print(des)
                kp = np.asarray(eval(des[1]))
                # print (kp.shape)
                histo = np.zeros(self.k)
                nkp = np.size(kp)
                # print(histo)
                # print(nkp)
                for d in kp:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

                histo_list.append(histo)
        print(np.asarray(histo_list).shape)
        return np.asarray(histo_list)


def euclidean_distance(imageA, imageB):
                # d=math.sqrt(np.sum([((a-b) ** 2) for (a,b) in zip(imageA,imageB)]))
                # return d
                return np.sqrt(np.sum((imageA - imageB) ** 2, axis=0))

def cosine_similarity(imageA, imageB):
        # print(imageA)
        # print(imageB)
        return np.dot(imageA, imageB)/(np.sqrt(np.sum(imageA ** 2, axis=0))*np.sqrt(np.sum(imageB ** 2, axis=0)))


def CSV(label):

arg = input("Which task would you like to run (1/2/3)? ")

featureDescriptor = input("Which feature do you want to compute - LBP or HOG: ")
try:
        k=int(input("K is: "))
except ValueError:
        print("Not an integer")
        exit()
if (featureDescriptor.upper() != "LBP" and featureDescriptor.upper() != "HOG" and featureDescriptor.upper() != "COLOR" and featureDescriptor.upper() != "SIFT"):
        print("Your feature descriptor is not correct")

technique = input("Which reduction techinique: ")

if arg == '1':
    SVD = SVDProcess(k,featureDescriptor)
    matrix_SVD, image_id = SVD.svd_latent()
    SVD.svdTask1(matrix_SVD,image_id)

elif arg == '2':
    image = input("Insert the name of your image: ")
    try:
        k=int(input("K is: "))
    except ValueError:
        print("Not an integer")
        exit()
    try:     
        similarityMetric = bool(int(input("Which similarity metric do you want to use - Cosine Similarity or Euclidean Distance (1/0): ")))
    except ValueError:
        print("Not a bool value")
        exit()
    
            # imClass.processImage(image,featureDescriptor,no_images,similarityMetric)
    SVD = SVDProcess(k,featureDescriptor,similarityMetric)
    matrix_image, image_id = SVD.svd_latent()
    SVD.svd_sim(image, no_images, matrix_image, image_id)

elif arg == '3':
    label = input("Which label do you want: ")


