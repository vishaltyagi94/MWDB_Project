CREATE TABLE IF NOT EXISTS imagedata_lbp (imageid text, image_data text[][], PRIMARY KEY (imageid));
CREATE TABLE IF NOT EXISTS imagedata_hog (imageid text, image_data text[][], PRIMARY KEY (imageid));
CREATE TABLE IF NOT EXISTS imagedata_sift (imageid text, image_data text[][], PRIMARY KEY (imageid));
CREATE TABLE IF NOT EXISTS imagedata_moments (imageid text, image_data text[][], PRIMARY KEY (imageid));
