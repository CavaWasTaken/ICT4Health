import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def main():
    plt.ion()   # turn on interactive mode
    infolder = 'lab2_moles'   # folder with images
    outfolder = 'Plot'  # folder to save the images

    #for filename in os.listdir(folder): # loop through all files in the folder
    filename = 'low_risk_4.jpg'
    file_path = os.path.join(infolder, filename)  # get the path to the file
    img = mpimg.imread(file_path)   # read the image

    [nrows, ncols, nchannels] = img.shape # get the image size
    img_2D = img.reshape(nrows*ncols, nchannels)    # reshape the image to a 2D array
    kmeans = KMeans(n_clusters=3, random_state=0) # create KMeans object, we want to have 3 clusters
    kmeans.fit(img_2D) # fit the model kmeans to the data

    # the image is in RGB, so it has 3 channels, when we execute kmean on the image, we get 3 centroids, each of them has 3 coordinates
    kmeans.cluster_centers_ # get the cluster centers (k=3 centroids with D=3 coordinates), k is the hyperparameter of the KMeans algorithm, D is the number of features
    kmeans.labels_ # get the labels of the each point in the image, so we get the cluster index for each pixel
    Ncluster = len(kmeans.cluster_centers_) # get the number of clusters, which is equal to the number of centroids. We read the number of rows of the centroids matrix
    centroids = kmeans.cluster_centers_.astype('uint8') # convert the centroids to 8-bit unsigned integers
    img_2D_quant = img_2D.copy() # create a copy of the 2D image
    for kc in range(Ncluster):  # loop through all clusters
        img_2D_quant[(kmeans.labels_ == kc),:] = centroids[kc,:]    # produce a quantized image, where each pixel is assigned to the closest centroid
    img_quant = img_2D_quant.reshape(nrows, ncols, nchannels)   # reshape the image to the original size
    plt.figure()
    plt.imshow(img_quant, interpolation=None)
    plt.title('Quantized image')
    #plt.show()
    plt.savefig(os.path.join(outfolder, 'quantized_'+filename))  # Save the plot as a file in the specified directory

    centroids = kmeans.cluster_centers_
    conv_to_gray = np.array([0.2125, 0.7154, 0.0721]) # conversion to grayscale
    centroids_gray = centroids@conv_to_gray # convert the centroids to grayscale
    i_col = centroids_gray.argmin() # get the index of the darkest centroid

    img_clust = kmeans.labels_.reshape(nrows, ncols) # reshape the labels to the image size
    mole_pos = np.argwhere(img_clust == i_col) # get the positions of the darkest cluster

    img_mole = np.ones([nrows, ncols]) * 255  # create a white image

    clusters = DBSCAN(eps=1, min_samples=4).fit(mole_pos)  # Fit DBSCAN
    labels = clusters.labels_  # Get cluster labels
    unique_labels = np.unique(labels)  # Get unique labels

    # find the more dense cluster
    most_compact_cluster_label = None
    smallest_inertia = float('inf')
    for label in unique_labels:
        if label != -1:  # Exclude noise
            cluster_points = mole_pos[labels == label]  # select the pixels of the cluster having the label selected
            centroid = np.mean(cluster_points, axis=0)  # Compute the centroid of the cluster
            inertia = np.sum(np.sum((cluster_points - centroid) ** 2, axis=1))  # Compute the inertia of the cluster
            if inertia < smallest_inertia and len(mole_pos[labels == label]) >= 1000 :  # Check if the current cluster is more compact than the previous one and if it has more than 1000 points
                smallest_inertia = inertia  # Update the smallest inertia
                most_compact_cluster_label = label  # Update the most compact cluster label

    # Visualize only the most compact cluster
    if most_compact_cluster_label is not None:
        print(f'Most compact cluster: {most_compact_cluster_label}')
        print(f'Number of points in the most compact cluster: {len(mole_pos[labels == most_compact_cluster_label])}')
        most_compact_cluster_points = mole_pos[labels == most_compact_cluster_label]
        img_mole[most_compact_cluster_points[:, 0], most_compact_cluster_points[:, 1]] = 0  # Set most compact cluster points to black

        # Crop the image to the bounding box of the mole
        min_row, min_col = np.min(most_compact_cluster_points, axis=0)
        max_row, max_col = np.max(most_compact_cluster_points, axis=0)
        cropped_img_mole = img_mole[min_row:max_row+1, min_col:max_col+1]
    
    cropped_img = img[min_row:max_row+1, min_col:max_col+1].copy()  # Crop the image to the bounding box of the mole

    # Save the segmented and cropped image
    plt.figure()
    plt.matshow(cropped_img_mole, cmap='gray')
    plt.title('Segmentation of moles')
    plt.savefig(os.path.join(outfolder, f'segmented_cropped.png'))  # Save the plot
    plt.close()

    # Save the segmented image
    plt.figure()
    plt.matshow(img_mole, cmap='gray')
    plt.title('Segmentation of moles')
    plt.savefig(os.path.join(outfolder, f'segmented.png'))  # Save the plot
    plt.close()

    delta = 6
    # add padding to the cropped image to avoid black borders
    padded_img_mole = np.pad(cropped_img_mole, pad_width=delta, mode='edge')

    img_mole_filt = np.zeros_like(cropped_img_mole)  # create a black image
    for kr in range(delta, max_row-min_row+1+delta):   # loop through all rows
        for kc in range(delta, max_col-min_col+1+delta):  # loop through all columns
            img_mole_filt[kr-delta, kc-delta] = np.mean(padded_img_mole[kr-delta:kr+delta+1, kc-delta:kc+delta+1])
            if(img_mole_filt[kr-delta, kc-delta] > 100):    # thresholding
                img_mole_filt[kr-delta, kc-delta] = 255 # set the pixel to white
            else:      # if the pixel is darker than 100   
                img_mole_filt[kr-delta, kc-delta] = 0   # set the pixel to black

    plt.figure()
    plt.matshow(img_mole_filt)
    plt.title('Filtered segmentation of moles')
    plt.savefig(os.path.join(outfolder, f'filtered_segmented.png'))  # Save the plot
    plt.close()

    # sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel filter in x direction
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # sobel filter in y direction
    img_mole_filt_sobel = np.zeros_like(img_mole_filt) # create a black image
    for kr in range(1, max_row-min_row+1-1):  # loop through all rows
        for kc in range(1, max_col-min_col+1-1):  # loop through all columns
            Gx = np.sum(np.sum(sobel_x * img_mole_filt[kr-1:kr+2, kc-1:kc+2])) # compute the gradient in x direction
            Gy = np.sum(np.sum(sobel_y * img_mole_filt[kr-1:kr+2, kc-1:kc+2])) # compute the gradient in y direction
            img_mole_filt_sobel[kr, kc] = np.sqrt(Gx**2 + Gy**2) # compute the gradient magnitude
            if(img_mole_filt_sobel[kr,kc] != 0):
                cropped_img[kr,kc] = [255,0,0] # set the pixel to red

    plt.figure()
    plt.matshow(img_mole_filt_sobel)
    plt.title('Sobel filter')
    plt.savefig(os.path.join(outfolder, f'sobel.png'))  # Save the plot
    plt.close()

    plt.figure()
    plt.imshow(cropped_img)
    plt.title('Original image with bounding box')
    plt.savefig(os.path.join(outfolder, f'original_with_bounding_box.png'))  # Save the plot
    plt.close()

    # plot the original image with the bounding founded with sobel filter
if __name__ == "__main__":
    main()