from PostgresDB import PostgresDB
from imageProcess import imageProcess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse.linalg import svds
from sklearn import svm
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
    def pca(self, imageset, k):
        dtd = np.cov(imageset.T)
        ddt = np.cov(imageset)
        u1,s1,v1 = svds(dtd, k)
        u2,s2,v2 = svds(ddt, k)
        return u1, v2

    # Function to convert the List into string to insert into database
    def convString(self, lst):
        values_st = str(lst).replace('[', '\'{')
        values_st = values_st.replace(']', '}\'')
        return values_st

    # Method to get the sorted list of image contributions to the Basis Vectors
    def imgSort(self, h, imgs_meta):
        h_sort = [np.argsort(x)[::-1] for x in h]
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
    def createInsertDB(self, dbase, imgs_red, conn):
        cur = conn.cursor()
        # Create the table
        sql = "CREATE TABLE IF NOT EXISTS {db} (imageid TEXT, imagedata TEXT[][])".format(db=dbase)
        cur.execute(sql)
        conn.commit()
        # Insert Values into the created table
        sql2 = "INSERT INTO {db} VALUES {x}".format(db=dbase, x=imgs_red[2:-2])
        cur.execute(sql2)
        conn.commit()
        cur.close()
        print('Reduced Features saved successfully to Table {x}'.format(x=dbase))

    def simMetric(self, d1, d2):
        return 1 / (1 + self.l2Dist(d1, d2))

    # Function to create subject id matrix
    def subMatrix(self, conn, dbname, mat=True):
        # Read from the database and join with Meta data
        cur = conn.cursor()
        sqlj = "SELECT t2.subjectid, ARRAY_AGG(t1.imageid), ARRAY_AGG(t1.imagedata) FROM {db} " \
               "t1 INNER JOIN img_meta t2 ON t1.imageid = t2.image_id GROUP BY t2.subjectid".format(db=dbname)
        cur.execute(sqlj)
        subjects = cur.fetchall()
        sub_dict = {x: np.mean(np.array(y,dtype=float), axis=0) for x,z,y in subjects}
        sub_sim = {x:'' for x in sub_dict.keys()}
        sub_mat = []
        for x in sub_dict.keys():
            sub_sim[x] = sorted([(el, self.simMetric(sub_dict[x], sub_dict[el])) for el in sub_dict.keys() if el != x], key=lambda x:x[1], reverse=True)[0:3]
            sub_mat.append([self.simMetric(sub_dict[x], sub_dict[el]) for el in sub_dict.keys()])

        if mat == False:
            return sub_sim
        else:
            k = input('Please provide the number of latent semantics(k): ')
            w, h = self.nmf(np.array(sub_mat), int(k))
            img_sort = self.imgSort(h, list(sub_dict.keys()))
        return np.array(img_sort)

    def binMat(self, conn, dbname):
        # Read from the database and join with Meta data
        cur = conn.cursor()
        sqlj = "SELECT t1.imageid, CASE WHEN t2.orient = 'left' THEN 1 ELSE 0 END , " \
               "CASE WHEN t2.orient = 'right' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.aspect = 'dorsal' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.aspect = 'palmar' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.accessories = '1' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.accessories = '0' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.gender = 'male' THEN 1 ELSE 0 END ," \
               "CASE WHEN t2.gender = 'female' THEN 1 ELSE 0 END" \
               " FROM {db} " \
               "t1 INNER JOIN img_meta t2 ON t1.imageid = t2.image_id".format(db=dbname)
        cur.execute(sqlj)
        subjects = cur.fetchall()
        img_meta = []
        bin_mat = []
        for x in subjects:
            img_meta.append(x[0])
            bin_mat.append(x[1:])
        k = input('Please provide the number of latent semantics(k): ')
        w, h = self.nmf(np.array(bin_mat).T, int(k))
        img_sort = self.imgSort(h, img_meta)
        features = ['left', 'right', 'dorsal', 'palmar', 'acessories', 'no_accessories', 'male', 'female']
        feature_sort = [np.argsort(x)[::-1] for x in w.T]
        feat_ls = []
        for idx, x in enumerate(feature_sort):
            feat_ls.append([(features[i], w.T[idx][i]) for i in x])
        return img_sort, feat_ls

    # Function to save the reduced dimensions to database
    def saveDim(self, feature, model, dbase, k, password='1Idontunderstand',
                host='localhost', database='postgres',
                user='postgres', port=5432, meta=False):

        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        imgs = self.dbProcess(password=password, process='f', model=feature)
        imgs_data = np.array([x[1] for x in imgs])
        imgs_meta = [x[0] for x in imgs]
        imgs_zip = list(zip(imgs_meta, imgs_data))
        if meta == True:
            self.createInsertMeta(conn)
            exit()
        if model == 'nmf':
            w, h = self.nmf(imgs_data.T, k)
            imgs_red = np.dot(imgs_data, w).tolist()
            imgs_sort = self.imgSort(h, imgs_meta)
            feature_sort = self.imgFeatureSort(w.T, imgs_zip)

        elif model == 'pca':
            u, v = self.pca(imgs_data, k)
            imgs_red = np.dot(imgs_data, u).tolist()
            imgs_sort = self.imgSort(v, imgs_meta)
            feature_sort = self.imgFeatureSort(u.T, imgs_zip)

        # Process the reduced Images
        imgs_red = list(zip(imgs_meta, imgs_red))
        imgs_red = self.convString(imgs_red)

        # Images ranked based on contribution to the Latent semantics
        self.createInsertDB(dbase, imgs_red, conn)
        return imgs_sort, feature_sort

    # Classify images based on label
    def classifyImg(self, conn, feature, img, label, dim):
        # fetch image dataset
        if feature == 'm':
            db_feature = 'imagedata_moments'
        elif feature == 's':
            db_feature = 'imagedata_sift'
        elif feature == 'h':
            db_feature = 'imagedata_hog'
        elif feature == 'l':
            db_feature = 'imagedata_lbp'

        # Fetch the data for a particular label
        if label in ['left', 'right']:
            field = 'orient'
        elif label in ['dorsal', 'palmar']:
            field = 'aspect'
        elif label in ['0', '1']:
            field = 'accessories'
        elif label in ['male', 'female']:
            field = 'gender'
        cur = conn.cursor()
        sqlj = "SELECT t1.imageid, t1.data FROM {db} t1 INNER JOIN img_meta t2 " \
               "ON t1.imageid = t2.image_id WHERE t2.{field} = '{label}'".format(db=db_feature, field=field, label=label)
        cur.execute(sqlj)
        label_data = cur.fetchall()
        sqlf = "SELECT t1.data FROM {db} t1 where t1.imageid = '{img}'".format(db=db_feature, img=img)
        cur.execute(sqlf)
        image = cur.fetchall()[0][0]
        if feature == 'm':
            image = [float(x) for y in image for x in y]
        else:
            image = [float(x) for x in image]
        recs_flt = []
        img_meta = []
        if feature == 'm':
            for rec in label_data:
                recs_flt.append([float(x) for y in rec[1] for x in y])
                img_meta.append(rec[0])
        else:
            for rec in label_data:
                recs_flt.append([float(y) for y in rec[1]])
                img_meta.append(rec[0])

        u, v = self.pca(np.array(recs_flt), 10)
        imgs_red = np.dot(recs_flt, u).tolist()
        clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
        clf.fit(imgs_red)
        image = np.dot(np.array(image), u)
        pred = clf.predict([image])
        x = clf.decision_function([image])
        print(x)
        print(pred)
        print(img_meta)
        print('test image', sum(image))
        print('label images:', sorted([sum(x) for x in imgs_red]))
        centroid = np.mean(imgs_red, axis=0)
        print('label distance from centroid:',sorted([self.l2Dist(centroid, i) for i in imgs_red], reverse=True))
        print('image:', self.l2Dist(centroid, image))
