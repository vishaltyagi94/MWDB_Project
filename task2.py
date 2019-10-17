from dimReduction import dimReduction
from PostgresDB import PostgresDB
import numpy as np
from imageProcess import imageProcess

def similarity (feature, technique, dbase, k, image):
    db = PostgresDB(password = "mynhandepg", database = "mwdb")
    conn = db.connect()
    if conn is None:
        print("Can not connect to database")
        exit()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM " + dbase)
    data = cursor.fetchall()
    image_id = [rec[0] for rec in data]
    image_index = image_id.index(image)
    image_data = np.asarray(eval(data[image_index][1]))

    task2 = imageProcess(path)
    similarity = {} 
    for i in range(len(image_id)):
        image_cmp = np.asarray(eval(data[i][1]))
        # if self.metrics:
        #     # similarity[row[0]] = 1- self.cosine_similarity(image, result)
        #     similarity[image_id[i]] = 1 - st.pearsonr(image,image_cmp)[0]
        #     # similarity[row[0]] = mean_squared_error(image,result)
        #     # similarity[row[0]] = 0 - self.psnr(image,result)
        # else:
        similarity[image_id[i]] = task2.euclidean_distance(image_data,image_cmp)
    similarity = sorted(similarity.items(), key = lambda x : x[1], reverse=False)
    task2.dispImages(similarity,feature, technique, 11, k)
    
        # print (image_cmp.shape)




path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(path, '*.jpg')
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda)')
image = input("Insert the name of your image: ")
k = input('Please provide the number of latent semantics(k): ')
db = 'imagedata_' + feature + '_' + technique
task2 = imageProcess("/home/anhnguyen/ASU/CSE-515/Project/Phase 2/Project - Phase 2/Data/testset1/")
task2.similarity(feature, technique, db, int(k), image)
# similarity(feature, technique, db, int(k), image)
# imgs_sort, feature_sort = dim.saveDim(feature, technique, db, int(k))
# phase1 = imageProcess("/home/anhnguyen/ASU/CSE-515/Project/Phase 2/Project - Phase 2/Data/testset1/")

# print('\n')
# print('Data Latent Semantics Saved to Output Folder!')
# dim.writeFile(imgs_sort, 'Output\\Task1\\Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
# print('\n')
# print('Feature Latent Semantics Saved to Output Folder!')
# dim.writeFile(imgs_sort, 'Output\\Task1\\Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))