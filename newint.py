from flask import Flask, render_template, request, jsonify,  Response
import numpy as np
import pickle
import requests
import pandas as pd
from skimage.measure import shannon_entropy
from skimage.filters import gabor
import os
import cv2 as cv
import scipy
from scipy.stats import skew
import pywt
from skimage.feature import local_binary_pattern
from PIL import Image
import base64
import io
# def preprocess_image(img):

#   #converting into grayscale image
#   #gray = np.dot(img,[0.2989, 0.5870, 0.1140])
#   gray=img

#   #increasing contrast by histogram equalization
#   hist, bins = np.histogram(gray, bins=256, range=(0, 256))
#   cdf = hist.cumsum()
#   cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
#   contrast = cdf_normalized[gray.astype('uint8')]


#   #applying median filter
#   kernel_half = 3 // 2
#   median = np.zeros((256, 256), dtype=np.uint8)
#   for y in range(kernel_half, 256 - kernel_half):
#     for x in range(kernel_half, 256 - kernel_half):
#       kernel = contrast[y - kernel_half:y + kernel_half + 1, x - kernel_half:x + kernel_half + 1]
#       median[y, x] = np.median(kernel)


#   #normalization
#   normalized = median.astype(float) / 255.0


#   #windowing
#   window_center = 0.5
#   window_width = 1
#   lower_bound = (window_center - window_width / 2.0)
#   upper_bound = (window_center + window_width / 2.0)
#   windowed_image = np.clip((normalized - lower_bound) / (upper_bound - lower_bound), 0, 1)


#   return windowed_image


def preprocess_image(img):
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    # Kernel for the morphological filtering
    ksize = 17 # higher kernel_size => more detail
    kernel = cv.getStructuringElement(1,(ksize,ksize))

    # BlackHat filtering
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    thresholdVal = 20
    ret,thresh2 = cv.threshold(blackhat,thresholdVal,255,cv.THRESH_BINARY)

    res = cv.inpaint(img,thresh2,1,cv.INPAINT_TELEA)

    medFiltered = cv.medianBlur(res,ksize=11)

    # Convert the image to LAB {Luminance, A Color Channel, B Color Channel} color space
    lab_image = cv.cvtColor(medFiltered, cv.COLOR_RGB2LAB)
    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv.split(lab_image)

    clahe = cv.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab_image = cv.merge((l_channel_clahe, a_channel, b_channel))

    # Convert the enhanced LAB image back to BGR
    enhancedImage = cv.cvtColor(enhanced_lab_image, cv.COLOR_LAB2RGB)

    return enhancedImage


# def feature_extraction(image):
#   # GLCM TEXTURE FEATURES
#   #GLCM parameters
#   distance = 1
#   # imagecopy = image
#   # image = (image*255).astype('uint8')
#   #0 and 45-degree angles chosen

#   #gray level co-occurrence matrix
#   glcm = np.zeros((256, 256), dtype=np.uint8)

#   for i in range(image.shape[0] - distance):
#      for j in range(image.shape[1] - distance):
#         p = image[i, j]
#         p=int(p)
#         q = image[i + distance, j + distance]
#         q=int(q)
#         glcm[p, q] = glcm[p,q]+1
#         q = image[i + distance, j]
#         q=int(q)
#         glcm[p, q] += 1
#         q = image[i, j + distance]
#         q=int(q)
#         glcm[p, q] += 1

#   # Normalizing GLCM
#   glcm = glcm / glcm.sum()

#   contrast = np.sum(glcm * np.square(np.arange(256) - np.mean(np.arange(256))))
#   dissimilarity = np.sum(glcm * np.abs(np.arange(256) - np.mean(np.arange(256))))
#   #Energy quantifies the uniformity or regularity of pixel intensities within the image patch. Higher energy values suggest more uniform textures.
#   energy = np.sum(np.square(glcm))
#   #Correlation measures the linear dependency between pixel pairs. High correlation values indicate strong linear relationships between pixels.
#   correlation = np.sum(glcm * (np.outer(np.arange(256), np.arange(256)) - np.outer(np.mean(np.arange(256)), np.mean(np.arange(256)))))
#   entropy = shannon_entropy(glcm)
#   homogeneity = 0
#   for i in range(256):
#         for j in range(256):
#             homogeneity += glcm[i, j] / (1 + (i - j)**2)
#   asm = np.sum(glcm**2) #angular second moment



#   # GABOR TEXTURE FEATURES
#   frequencies = [0.1, 0.5, 1.0]
#   angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

#   for f in frequencies:
#     for a in angles:
#       gabor_filter = np.abs(gabor(image,frequency = f,theta = a)[0])
#       mean_gabor = np.mean(gabor_filter)
#       std_gabor = np.std(gabor_filter)



#   # HISTOGRAM BASED TEXTURE FEATURES
#   flattened = image.flatten()
#   hist , bins = np.histogram(flattened,bins = 256)
#   #normalizing histogram
#   histogram = hist/sum(hist)
#   mean = np.mean(flattened)
#   variance = np.var(flattened)
#   hist , bins = np.histogram(flattened,bins = 256)
#   skewness = np.mean((flattened - mean) ** 3) / np.power(variance, 3/2)
#   kurtosis = np.mean((flattened - mean) ** 4) / np.power(variance, 2) - 3
#   energy_histogram = np.sum(histogram ** 2)
#   entropy_histogram = -np.sum(histogram * np.log2(histogram + 1e-10))



#   # RUN LENGTH MATRIX FEATURES
#   def calculate_run_length_matrix(image1, direction):
#       runs = [np.asarray(row) for row in image1]
#       run_lengths = [np.unique(run, return_counts=True) for run in runs]
#       max_length = max(len(run) for run in runs)

#       run_length_matrix = np.zeros((len(runs), max_length), dtype=int)

#       for i, run in enumerate(runs):
#           unique, counts = np.unique(run, return_counts=True)
#           run_length_matrix[i, unique] = counts
          

#       return run_length_matrix

#   rlm_matrix = calculate_run_length_matrix(image.tolist(), 'horizontal')

#   # Extracting features from the run length matrix
#   short_run_emphasis = np.sum(rlm_matrix / (np.arange(1, rlm_matrix.shape[1] + 1))[:, np.newaxis]) / np.sum(rlm_matrix)
#   long_run_emphasis = np.sum(rlm_matrix * (np.arange(1, rlm_matrix.shape[1] + 1))[:, np.newaxis]) / np.sum(rlm_matrix)
#   gray_level_nonuniformity = np.sum(np.sum(rlm_matrix, axis=1)**2) / np.sum(rlm_matrix)
#   run_length_nonuniformity = np.sum(np.sum(rlm_matrix, axis=0)**2) / np.sum(rlm_matrix)
#   low_gray_level_run_emphasis = np.sum(rlm_matrix / (np.arange(1, rlm_matrix.shape[0] + 1))[:, np.newaxis]) / np.sum(rlm_matrix)
#   high_gray_level_run_emphasis = np.sum(rlm_matrix * (np.arange(1, rlm_matrix.shape[0] + 1))[:, np.newaxis]) / np.sum(rlm_matrix)




#   #SHAPE BASED FEATURES
#   _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
#   # Finding contours in the binary mask
#   binary_mask = np.array(binary_mask, np.uint8)
#   contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   area=0
#   perimeter=0
#   compactness=0
#   circularity=0
#   aspect_ratio=0
#   features = [contrast,dissimilarity,energy,correlation,entropy,homogeneity,asm,mean_gabor,std_gabor,mean,variance,skewness,kurtosis,energy_histogram,entropy_histogram,short_run_emphasis,long_run_emphasis,gray_level_nonuniformity,run_length_nonuniformity,low_gray_level_run_emphasis,high_gray_level_run_emphasis,area,perimeter,compactness,circularity,aspect_ratio]
#   features_list = []
#   for contour in contours:
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)
#     compactness = (perimeter ** 2) / area if area > 0 else 0
#     circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
#     # Calculating bounding box
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = w / h if h > 0 else 0

#     features = [contrast,dissimilarity,energy,correlation,entropy,homogeneity,asm,mean_gabor,std_gabor,mean,variance,skewness,kurtosis,energy_histogram,entropy_histogram,short_run_emphasis,long_run_emphasis,gray_level_nonuniformity,run_length_nonuniformity,low_gray_level_run_emphasis,high_gray_level_run_emphasis,area,perimeter,compactness,circularity,aspect_ratio]
#   return features

def entropyCalc(img):
    entropyFeatureVector = {}
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    binsCount = 128
    hist, _ = np.histogram(gray.ravel(), bins=binsCount, range=(0, binsCount))
    if hist.sum() == 0:
        entropyFeatureVector["SE"]=0.0
        entropyFeatureVector["TE"]=0
        entropyFeatureVector["NE"]=0.0
        entropyFeatureVector["E"]=0.0
        return entropyFeatureVector

    prob_dist = hist / hist.sum()
    image_entropy = scipy.stats.entropy(prob_dist, base=2)
    entropyFeatureVector["E"] = image_entropy                                   ## ENTROPY (E)
    shEntropy = -np.sum(prob_dist * np.log2(prob_dist + np.finfo(float).eps))
    entropyFeatureVector["SE"] = shEntropy                                      ## SHANNON ENTROPY (SE)
    best_threshold = 0
    max_entropy = 0.0
    for threshold in range(1, 256):
        lower_prob = prob_dist[:threshold].sum()
        upper_prob = prob_dist[threshold:].sum()

        lower_entropy = -np.sum(prob_dist[:threshold] / (lower_prob + np.finfo(float).eps) * np.log2(prob_dist[:threshold] / (lower_prob + np.finfo(float).eps) + np.finfo(float).eps))
        upper_entropy = -np.sum(prob_dist[threshold:] / (upper_prob + np.finfo(float).eps) * np.log2(prob_dist[threshold:] / (upper_prob + np.finfo(float).eps) + np.finfo(float).eps))

        combined_entropy = lower_prob * lower_entropy + upper_prob * upper_entropy
        if combined_entropy > max_entropy:
            max_entropy = combined_entropy
            best_threshold = threshold
    entropyFeatureVector["TE"] = best_threshold                                 ## THRESHOLD ENTROPY (TE)

    num_pixels = gray.size
    normalized_entropy = shEntropy / (np.log2(num_pixels))
    entropyFeatureVector["NE"] = normalized_entropy                             ## NORMALISED SHANNON ENTROPY (NE)
    return entropyFeatureVector

def entropyFeatureExtraction(preProcessedImgs):
    return entropyCalc(preProcessedImgs)


def textureCal(img):
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    txFeatureVector = {}
    distances = [1]  # You can use multiple distances
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Four angles (0, 45, 90, and 135 degrees)
    glcm = np.zeros((256, 256), dtype=np.float64)
    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            for i in range(img.shape[0] - distance):
                for j in range(img.shape[1] - distance):
                    glcm[img[i, j], img[i + dx, j + dy]] += 1

    glcm /= glcm.sum()
    energy = np.sum(glcm ** 2)
    contrast = np.sum((np.arange(256).reshape(256, 1) - np.arange(256)) ** 2 * glcm)
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    homogeneity = np.sum(glcm / (1 + np.abs(np.arange(256).reshape(256, 1) - np.arange(256))))
    txFeatureVector["GLCMenergy"]=energy
    txFeatureVector["GLCMcontrast"]=contrast
    txFeatureVector["GLCMentropy"]=entropy
    txFeatureVector["GLCMhomogeneity"]=homogeneity
    return txFeatureVector

def textureFeatureExtraction(preProcessedImgs):
    return textureCal(preProcessedImgs)

def statFeatureCalc(img):
    statFeatureVector = {}
    ##GRAY SCALE MEAN
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    grayMean = np.mean(gray)
    statFeatureVector["GSM"] = grayMean
    ##GRAY SCALE STD
    grayStd = np.std(gray)
    statFeatureVector["GSTD"] = grayStd
    #GRAY L1 NORM
    grayl1Norm = np.linalg.norm(gray, ord=1)
    statFeatureVector["GL1N"] = grayl1Norm
    #GRAY L2 NORM
    grayl2Norm = np.linalg.norm(gray, ord=2)
    statFeatureVector["GL2N"] = grayl2Norm
    #GRAY P2P
    grayPixelRange = np.ptp(gray)
    statFeatureVector["GRAYP2P"] = grayPixelRange

    skewValue = skew(gray.flatten())
    # print("Skewness:", skewness_value)
    statFeatureVector["GSKEW"]=skewValue

    ##GRAY MEDIAN
    grayMedian = np.median(gray)
    statFeatureVector["GMEDIAN"] = grayMedian

    ##WAVELET FEATURES
    waveletFeatures = {}
    wavelet_family = 'haar'
    level = 2

    coeffs = pywt.wavedec2(gray,wavelet_family,level = level)

    wgEnergy = [np.sum(np.square(c)) for c in coeffs]
    wgSTD = [np.std(c) for c in coeffs]

    # Calculate wavelet contrast
    wgContrast = []

    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        mean = np.mean(detail)
        contrast = np.sum(np.square(detail - mean))
        wgContrast.append(contrast)

    waveletFeatures["WGENERGY"] = wgEnergy
    waveletFeatures["WGSTD"] = wgSTD
    waveletFeatures["WGCONTRAST"] = wgContrast
    statFeatureVector["WAVELET"] = waveletFeatures
    return statFeatureVector

def statsFeatureExtraction(preProcessedImgs):
    return statFeatureCalc(preProcessedImgs)

def lbpFeatureCalc(img):
    lbpFeatureVector = {}
    # LBP parameters
    radius = 3
    n_points = 8 * radius
    # method = 'uniform'

    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    # Calculate LBP
    # lbp = local_binary_pattern(gray, n_points, radius, method)
    lbp = local_binary_pattern(gray, n_points, radius)

    # Calculate histogram of LBP and normalize
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram

    for itr in range(len(hist)):
        lbpFeatureVector["LBP-"+str(itr+1)] = hist[itr]
    # print("LBP Histogram:")
    # print(hist)
    
    return lbpFeatureVector


def lbpFeatureExtraction(preProcessedImgs):
    return lbpFeatureCalc(preProcessedImgs)



def dfRowGen(lbpFeatures,enFeatures,stFeatures,txFeatures):
    x = {}
    for iter in enFeatures.items():
        x[iter[0]] = iter[1]
    for iter in list(stFeatures.items())[:7]:
        x[iter[0]] = iter[1]
    for iter in list(stFeatures.items())[7][1].items():
        for q in range(len(iter[1])):
            x[iter[0]+str(q+1)] = iter[1][q]
    for iter in txFeatures.items():
        x[iter[0]] = iter[1]
    for iter in lbpFeatures.items():
        x[iter[0]] = iter[1]
    x = np.array(list(x.items()))
    return(x)

def dfGenerator(lbpFeatures,enFeatures,stFeatures,txFeatures):
    # print (dfRowGen(lbpFeatures,enFeatures,stFeatures,txFeatures))   
    return (dfRowGen(lbpFeatures,enFeatures,stFeatures,txFeatures))   


def extract_features(img):
    # Add your feature extraction logic here
    # For example, using the functions you defined earlier
    # Return a list of feature values
    ehImg = preprocess_image(img)
    enFeatures = entropyFeatureExtraction(ehImg)
    stFeatures = statsFeatureExtraction(ehImg)
    txFeatures = textureFeatureExtraction(ehImg)
    lbpFeatures = lbpFeatureExtraction(ehImg)
    featureDF = dfGenerator(lbpFeatures, enFeatures, stFeatures, txFeatures)
    # feature_values = featureDF.values.flatten().tolist()
    feature_values = [x[1] for x in featureDF]
    # print(type(featureDF))
    print(feature_values)
    return feature_values
    # return featureDF


app= Flask(__name__)
svm=pickle.load(open('skinCancer.pkl','rb'))
#pca=pickle.load(open('pca.pickle','rb'))
#scaler=pickle.load(open('scaler.pickle','rb'))

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    
    imagefile=request.files['imagefile']
    print(imagefile)
    #image_bytes = base64.b64decode(imagefile)
    #image_np_array = np.array(Image.open(io.BytesIO(image_bytes)))
    image_path="./images"+imagefile.filename
    imagefile.save(image_path)
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    # s1 = preprocess_image(image)
    s1 = preprocess_image(image)
    s2 = extract_features(s1)
    s2 = np.asarray(s2)
    s2 = s2.reshape(1,-1)
    #s2 = scaler.transform(s2)
    feature_names = ['E','SE','TE','NE','GSM','GSTD','GL1N','GL2N','GRAYP2P','GSKEW','GMEDIAN','WGENERGY1','WGENERGY2','WGENERGY3','WGSTD1','WGSTD2','WGSTD3','WGCONTRAST1','WGCONTRAST2','GLCMenergy','GLCMcontrast','GLCMentropy','GLCMhomogeneity','LBP-1','LBP-2','LBP-3','LBP-4','LBP-5','LBP-6','LBP-7','LBP-8','LBP-9','LBP-10','LBP-11','LBP-12','LBP-13','LBP-14','LBP-15','LBP-16','LBP-17','LBP-18','LBP-19','LBP-20','LBP-21','LBP-22','LBP-23','LBP-24','LBP-25','LBP-26']
    s2 = pd.DataFrame(s2 , columns = feature_names)
    #s2 = pd.DataFrame(pca.transform(s2))
    prediction =svm.predict(s2)
    if prediction==0:
       result = "akiec"
    elif prediction==1:
       result = "bcc"
    elif prediction==2:
       result = "bkl"
    elif prediction==3:
       result = "df"
    elif prediction==4:
       result = "mel"
    elif prediction==5:
       result = "nv"
    else:
       result = "vasc"
    return jsonify({'prediction': result})

if __name__=="__main__":
    app.run(port=3000,debug=True)