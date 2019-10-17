from PostgresDB import PostgresDB
from imageProcess import imageProcess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import joblib
from sklearn.cluster import KMeans
import os




class KMeans_SIFT:
    def __init__(self,k):
        self.k = k

    def kmeans_process(self,matrix_image):
        batch_size = 20 * 3
        kmeans = KMeans(n_clusters=self.k, verbose=0).fit(matrix_image)
        return kmeans

    def newMatrixSift(self,data, kmeans):
        kmeans.verbose = False
        histo_list = []
        for des in data:
            # print(des)
            kp = np.asarray(des[1])
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


class dimReduction(imageProcess):
    def __init__(self, dirpath, ext='*.jpg'):
        super().__init__(dirpath=dirpath, ext=ext)

    # Function to fetch Reduced dimensions from image
    def nmf(self, imageset, k):
        model = NMF(n_components=k, init='random', random_state=0)
        scaler = StandardScaler(with_mean=False, with_std=True).fit(imageset)
        imageset= scaler.transform(imageset)
        W = model.fit_transform(imageset)
        H = model.components_

        return W, H
    # Function to perform PCA
    def pca(self, imageset, k,feature):
        # dtd = np.cov(imageset.T)
        # ddt = np.cov(imageset)
        # u1,s1,v1 = svds(dtd, k)
        # u2,s2,v2 = svds(ddt, k)
        pca = PCA(n_components=k)
        data = pca.fit_transform(imageset)
        Sigma = np.diag(pca.explained_variance_)
        path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Models'  +os.sep)
        

        with open(path + os.sep  +'pca_'+ feature +'.joblib', 'wb') as f1:
            joblib.dump(pca, f1)

        return data, np.dot(data,np.linalg.inv(Sigma)), pca.components_
        # return u1, v2

    def svd(self,image, k):
            U, s, Vt = svds(image, k)
            # print(s.shape)
            # Sigma = np.zeros((image.shape[0], image.shape[1]))
            Sigma = np.diag(s)
            # image = U[:,:self.k].dot(Sigma[:self.k, :self.k]).dot(V[:self.k,:])
            # print(image.shape)
            return U,Sigma,Vt

    # Function to convert the List into string to insert into database
    def convString(self, lst):
        values_st = str(lst).replace('[', '\'{')
        values_st = values_st.replace(']', '}\'')
        return values_st

    # Method to get the sorted list of image contributions to the Basis Vectors
    def imgSort(self, h, imgs_meta):
        h_sort = [np.argsort(x)[::-1] for x in h]
        print(imgs_meta)
        print(h_sort)
        print(np.asarray(imgs_meta).shape)
        print(np.asarray(h_sort).shape)
        img_sort = []
        for idx, hs in enumerate(h_sort):
            img_sort.append([(imgs_meta[x], h[idx][x]) for x in hs])
        return img_sort

    # Method to get the sorted list of image contributions to features
    def imgFeatureSort(self, u, imgs):
        targ_imgs = []
        for vec in u:
            x = [(np.dot(vec, img), id) for id, img in imgs]
            y = sorted(x, key=lambda z: z[0], reverse=True)
            targ_imgs.append(y[0])

        return targ_imgs

    def ImgViz(self, images, savepath):
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

    # Create table and insert data into it
    def createInsertDB(self, dbname, imgs_red, conn):
        cur = conn.cursor()
        # Create the table
        sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT NOT NULL, imagedata TEXT, PRIMARY KEY (imageid))".format(db=dbname)
        cur.execute(sql)
        conn.commit()

        for image in imgs_red:
            print(image)
            sql = "SELECT {field} FROM {db} WHERE {field} = '{condition}';".format(field="imageid",db=dbname,condition=image[0])
            # print("SQL Check Exist - HOG: ", sql)
            cur.execute(sql)

            # cur.execute(sql)
            insert_value = str(np.asarray(image[1]).tolist())
            if cur.fetchone() is None:
                print("Insert")
                # print("Not Exist HOG - Insert")
                sql2 = "INSERT INTO {db} VALUES('{x}', '{y}');".format(x=image[0],y=insert_value, db=dbname)
            else:
                print("Update")
                # print("Exist HOG - Update")
                # column = "HOG"
                
                sql2 = "UPDATE {db} SET {x} ='{y}' WHERE IMAGEID = '{z}'".format(x="imagedata",y=insert_value,z = image[0], db=dbname)
            # Insert Values into the created table
            # sql2 = "INSERT INTO {db} VALUES {x}".format(db=dbname, x=imgs_red[2:-2])
            cur.execute(sql2)
        conn.commit()
        cur.close()
        print('Reduced Features saved successfully to Table {x}'.format(x=dbname))

    # Function to save the reduced dimensions to database
    def saveDim(self, feature, model, dbase, k, password='1Idontunderstand',
                host='localhost', database='postgres',
                user='postgres', port=5432, label=None):

        imageDB = imageProcess(self.dirpath)
        imgs = imageDB.dbProcess(password=password, process='f', model=feature, dbase = dbase)
        
        if label is not None:
            dbase += '_' + model + '_' + label
            filteredImage = imageDB.CSV(label)
        else:
            dbase += '_' + model

        imgs_data = []
        imgs_meta = []

        # print(feature)
        # if label is not None:
        #     index_array = np.asarray([i if x[0] in filteredImage for x in imgs])
        #     imgs = imgs[index_array]
        # print(filteredImage)
        i = -1
        while i < len(imgs)-1:
            # print (x[1].shape)
            i += 1
            if label is not None and imgs[i][0] not in filteredImage:
                # print("label")
                del imgs[i]
                i -= 1
                continue
            if feature != "s":
                imgs_data.append(imgs[i][1].reshape((-1)))
            else:
                imgs_data.extend(imgs[i][1])   
                    # print (image_cmp.shape)
            imgs_meta.append(imgs[i][0])
            # print(i)
            # print(len(imgs))
        
        
        imgs_data = np.asarray(imgs_data)
        print(imgs_data.shape)
        print(imgs_meta)
        # imgs_meta = [x[0] if x[0] in filteredImage for x in imgs]
        imgs_zip = list(zip(imgs_meta, imgs_data))
        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        # if meta != None:
        #     self.createInsertMeta(conn)

        model = model.lower()

        if feature == "s":
            Kmeans = KMeans_SIFT(300)
            clusters = Kmeans.kmeans_process(imgs_data)
            # print (imgs_zip)
            imgs_data = Kmeans.newMatrixSift(imgs, clusters)
            imgs_zip = list(zip(imgs_meta, imgs_data))

        if model == 'nmf':
            w, h = self.nmf(imgs_data.T, k)
            imgs_red = np.dot(imgs_data, w).tolist()
            imgs_sort = self.imgSort(h, imgs_meta)
            feature_sort = self.imgFeatureSort(w.T, imgs_zip)

        elif model == 'pca':
            data, U, Vt = self.pca(imgs_data, k, feature)
            imgs_red = data.tolist()
            # imgs_red = np.dot(imgs_data, u).tolist()
            # print(imgs_red.shape)
            print(Vt.shape)
            print(U.shape)
            # print(np.asarray(imgs_red).shape)
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        elif model == 'svd':
            # print(imgs_data.shape)
            U,Sigma,Vt = self.svd(imgs_data, k)
            # print(U.shape)
            # print(Sigma.shape)
            # print(Vt.shape)
            # imgs_red = np.dot(U, Sigma, Vt).tolist()
            imgs_red = U.dot(Sigma).dot(Vt).tolist()
            # print(im)
            # U[:,:self.k].dot(Sigma[:self.k, :self.k]).dot(V[:self.k,:])
            imgs_sort = self.imgSort(U.T, imgs_meta)
            feature_sort = self.imgFeatureSort(Vt, imgs_zip)

        # print("=======================")
        # print(imgs_sort)
        # print("=======================")
        # print(feature_sort)
        # Process the reduced Images
        imgs_red = list(zip(imgs_meta, imgs_red))
        # print(imgs_red[0])
        # imgs_red = self.convString(imgs_red)
        # print(imgs_red[0])
        # Images ranked based on contribution to the Latent semantics
        self.createInsertDB(dbase, imgs_red, conn)
        return imgs_sort, feature_sort