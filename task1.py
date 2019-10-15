from dimReduction import dimReduction

arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(arg, '*.jpg')
arg2 = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if arg2 not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
arg3 = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda)')
arg4 = input('Please provide the number of latent semantics(k): ')
db = 'imagedata_' + arg2 + '_' + arg3
imgs_sort, feature_sort = dim.saveDim(arg2, arg3, db, int(arg4))
print('\n')
print('Data Latent Semantics Saved to Output Folder!')
dim.writeFile(imgs_sort, 'Output\\Task1\\Data_ls_{x}_{y}_{z}.txt'.format(x=arg2,y=arg3,z=arg4))
print('\n')
print('Feature Latent Semantics Saved to Output Folder!')
dim.writeFile(imgs_sort, 'Output\\Task1\\Feature_ls_{x}_{y}_{z}.txt'.format(x=arg2,y=arg3,z=arg4))
