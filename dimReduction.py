from PostgresDB import PostgresDB
from imageProcess import imageProcess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse.linalg import svds

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

    # Function to save the reduced dimensions to database
    def saveDim(self, feature, model, dbase, k, password='1Idontunderstand',
                host='localhost', database='postgres',
                user='postgres', port=5432, meta=False):
        imgs = self.dbProcess(password=password, process='f', model=feature)
        imgs_data = np.array([x[1] for x in imgs])
        imgs_meta = [x[0] for x in imgs]
        imgs_zip = list(zip(imgs_meta, imgs_data))
        db = PostgresDB(password=password, host=host,
                        database=database, user=user, port=port)
        conn = db.connect()
        if meta == True:
            self.createInsertMeta(conn)

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


