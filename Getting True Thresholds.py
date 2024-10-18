import cv2
import pandas as pd
import csv
import os
import time
import numpy as np
import tifffile
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from itertools import product


def get_otsu_threshold(im):
    # Thanks, Wikipedia: https://en.wikipedia.org/wiki/Otsu%27s_method
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(im, bins=bins_num)

    # Get normalized histogram if it is required
    is_normalized = hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result:", threshold)
    return threshold


def imsize_filter(im, size=500):
    """
    perform a size filter on an image
    https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/

    Args:
        im: the image
        size: the minimum size of objects allowed
    """
    total_labels, label_id, values, _ = cv2.connectedComponentsWithStats(
        im.astype(np.uint8), connectivity=8
    )
    output = np.zeros(im.shape, 'uint8')
    for i in range(1, total_labels):
        area = values[i, cv2.CC_STAT_AREA]
        if area >= size:
            output[label_id == i] = 1
    #
    return output


def positive_pixels(tif_location_list, layer_img, threshold, pixels, mask_dict):
    """
    :param tif_location_list: location of each specific tif in the combined tif array
    :param layer_img: array of compiled tifs
    :param threshold: threshold desired for positive pixel generation
    :param pixels: number of connected pixels desired
    :return: total number of positive pixels for a given experimental condition
    """
    layer_img_copy = np.copy(layer_img)
    mask_array = (layer_img_copy >= threshold)
    pixels_per_field_list = []
    total_pixels = 0
    for idx, locations in enumerate(tif_location_list):
        ind_tif = mask_array[int(float(locations.split(':')[0])):int(float(locations.split(':')[1]))]
        ind_tif = imsize_filter(ind_tif, pixels)
        total_pixels += ind_tif.sum()
        pixels_per_field_list.append(ind_tif.sum())
        if tifs_list[idx].replace('component_data', 'composite_image') not in mask_dict.keys():
            mask_dict[tifs_list[idx].replace('component_data', 'composite_image')] = ind_tif
    return total_pixels, pixels_per_field_list


def read_mifto(key_split, true_threshold_dict, thresh_experiment_df):
    """
    :param key_split: How the key in true_threshold_dict is being structured
    :param true_threshold_dict: idk if this is neccessary, but dict that holds all threshold info
    :param thresh_experiment_df: idk if this is neccessary, but df of read mifto excel file
    :return: returns true_threshold_dict to ensure changes are made
    """
    true_threshold_dict[paths][f'{v}{key_split}{(thresh_experiment_df.iloc[i][thresh_experiment_df.columns[0]])}'] = {}
    true_threshold_dict[paths][f'{v}{key_split}{(thresh_experiment_df.iloc[i][thresh_experiment_df.columns[0]])}']['True'] = {}
    true_threshold_dict[paths][f'{v}{key_split}{(thresh_experiment_df.iloc[i][thresh_experiment_df.columns[0]])}']['True']['Threshold']=\
        (thresh_experiment_df.at[i, v])
    true_threshold_dict[paths][f'{v}{key_split}{(pixels_experiment_df.iloc[i][thresh_experiment_df.columns[0]])}']['Pixels'] = float(
        pixels_experiment_df.at[i, v])
    return true_threshold_dict


def generate_nonblurred_threshold(split_number=-4):
    """
    :param split_number: split indice for the keys in true_threshold_dict
    :return: returns true_threshold_dict to ensure changes are made
    """
    pixels = \
        true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['Pixels']
    true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['Otsu'] = {}
    true_threshold = true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['True']['Threshold']
    no_blur_threshold = \
    true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['Otsu'][
        'Threshold'] \
        = float(get_otsu_threshold(compiled_tifs))
    true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['True'][
        'Positive Pixels'], true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['True'][
        'Positive Pixels Per Field'] = positive_pixels(tif_location_list, compiled_tifs, true_threshold, pixels, mask_dict)
    true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['Otsu'][
        'Positive Pixels'], true_threshold_dict[k.split('\\Data')[0]][f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}']['Otsu'][
        'Positive Pixels Per Field'] = positive_pixels(tif_location_list, compiled_tifs, no_blur_threshold, pixels, mask_dict)
    return pixels, true_threshold_dict


def ksize_test(compiled_tifs, split_number=-4, ksize_min=1, ksize_max=8, sigma_min=1, sigma_max=4):
    """
    :param compiled_tifs: array with all compiled tifs for a condition
    :param split_number: split indice for the keys in true_threshold_dict
    :param ksize_min: minimum ksize desired to test
    :param ksize_max: this minus 1 is the maximum ksize test (due to range function)
    :param sigma_min: minimum sigma value desired to test
    :param sigma_max: this minus 1 is the maximum sigma test (due to range function)
    :return: returns true_threshold_dict to ensure changes are made
    """
    for ksize in range(ksize_min, ksize_max, 2):
        for sigma in range(sigma_min, sigma_max, 1):
            compiled_tifs_copy = np.copy(compiled_tifs)
            blur = cv2.GaussianBlur(compiled_tifs_copy, (ksize, ksize), sigma)
            true_threshold_dict[k.split('\\Data')[0]][
                f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}'][
                f'{sigma} Sigma, {ksize} ksize Otsu Threshold'] = {}
            blur_threshold = true_threshold_dict[k.split('\\Data')[0]][
                f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}'][
                f'{sigma} Sigma, {ksize} ksize Otsu Threshold']['Threshold'] = float(
                get_otsu_threshold(blur))  # changing the split here might fix the wierd naming otsu gives
            true_threshold_dict[k.split('\\Data')[0]][
                f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}'][
                f'{sigma} Sigma, {ksize} ksize Otsu Threshold']['Positive Pixels'], true_threshold_dict[k.split('\\Data')[0]][
                f'{tissue_tifs.split(" tifs")[0]}_{k.split("_")[split_number]}'][
                f'{sigma} Sigma, {ksize} ksize Otsu Threshold']['Positive Pixels Per Field'] = (
                positive_pixels(tif_location_list, compiled_tifs, blur_threshold, pixels, mask_dict))
    return true_threshold_dict


def cull_outliers(query_list):
    """
    :param query_list: list you want to check for outliers (err or diff)
    :return: changes to list
    """
    q1 = np.percentile(query_list, 25)
    q3 = np.percentile(query_list, 75)
    iqr = q3 - q1
    upper_fence = q3 + (iqr * 1.5)
    lower_fence = q1 - (iqr * 1.5)
    for i in range(len(query_list)):
        if lower_fence > query_list[i] or query_list[i] > upper_fence:
            query_list.pop(query_list.index(query_list[i]))
    return query_list


def generate_immask(mask_array_with_filter, color_tif_path):
    # img_mask_list = []
    color_tif = tifffile.imread(color_tif_path)
    mask_3_channel = cv2.cvtColor(mask_array_with_filter, cv2.COLOR_GRAY2RGB)
    masked_img = np.copy(color_tif)
    masked_img[np.where((mask_3_channel == [1, 1, 1]).all(axis=2))] = (255, 0, 0)
    file_path = os.path.join(color_tif_path, 'with_mask.png')
    # img_mask_list.append(masked_img)
    # cv2.imwrite(file_path, masked_img) #change if you want to save the images on computer
    return masked_img


#  reading and turning excel sheet into a dataframe, then into a dict
start = time.time()  # start timer
excel_file = r"C:\Users\gsalvuc1\Desktop\experiment_records_true.xlsx"  # xl sheet from scrubbin script
fields = ['Path', 'IsGood']  # columns of interest to sort by in the above sheet
df = pd.read_excel(excel_file, sheet_name='experiment_records', usecols=fields)  # turning the columns into a dataframe
experiment_records_temp = df.set_index('Path').T.to_dict('list')  # dataframe -> dict
#  establishing empty data structures for later use
true_threshold_dict = {}
path_does_not_exist = []
auto_otsu_dict = {}
exclude = ['im3', 'IM3', 'Algorithms', 'Project_Algorithms', 'project_algorithms']  # used to exclude dirs from os.walk

# establishing lists and dicts used later to get the correct layer of the tif used for thresholding
opal_list = ['DAPI', 'IHC', '480', '520', '540', '570', '620', '650', '690', '780', 'AF']  # all opals used
layers = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # layers in the tifs that correspond to each opal
opal_layer_dict = dict(zip(opal_list, layers))  # dict, keys = opals, values = numbers
layer_opal_dict = dict(zip(layers, opal_list))  # dict, keys = numbers, values = opals
# keywords to help determine experiment type
titration_keywords = ['Primary', 'primary', 'TSA', 'tsa']


# Looks for experiments in the scrubbin excel sheet that are useable (IsGood == 1)
for key, value in experiment_records_temp.items():  # this dict should have all the experiment paths and IsGood values
    # take all experiments with an IsGood value == 1 and put them into a separate dictionary
    if len(value) == 1 and 1 in value:
        true_threshold_dict[key] = value
        true_threshold_dict[key] = {}
del experiment_records_temp, key, value

# finding threshold and connected pixel csvs from mifto as well as the tifs for the experiment
for keys in true_threshold_dict.keys():
    for root, dirs, file in os.walk(keys):
        dirs[:] = [d for d in dirs if d not in exclude]  # this excludes all dirs, makes os.walk a bit faster
        for programs in file:
            # finding the thresholds csv
            if programs.endswith('.csv') and 'Threshold values' in programs:
                if 'Results.pixels' in root.split('\\'):  # this prevents finding repeat folder 'Result.pixels1'
                    true_threshold_dict[keys]['mIFTO Threshold Path'] = os.path.join(root, programs)
            # finding the connected pixels csv
            elif programs.endswith('.csv') and 'connected pixel values' in programs:
                if 'Results.pixels' in root.split('\\'):  # this prevents finding repeat folder 'Result.pixels1'
                    true_threshold_dict[keys]['mIFTO Pixels Path'] = os.path.join(root, programs)
            # finding tifs and making a new dict for info storage
            elif programs.endswith(".tif") and 'component_data' in str(programs):
                if root not in auto_otsu_dict.keys():  # the 'and not' makes only 1 condition
                    auto_otsu_dict[root] = {}

# Parses the mIFTO threshold sheet to get the titration used. This is where the sorting lists come in handy
for paths in true_threshold_dict.keys():  # keys = paths to experiment folders
    if os.path.exists(paths):
        print(f'{paths} path exists')
        # reading the csvs into dataframes
        thresh_experiment_df = pd.read_csv(true_threshold_dict[paths]['mIFTO Threshold Path'], index_col=False)
        pixels_experiment_df = pd.read_csv(true_threshold_dict[paths]['mIFTO Pixels Path'], index_col=False)
        tissue_list = []
        for k, v in enumerate(thresh_experiment_df):  # k = nums, v= column names
            if v != 'Unnamed: 0':  # gets rid of index col in dataframe, which might be redundant (see index_col=False)
                tissue_list.append(v)  # gets a list of all tissues used in the experiment
                for folder in auto_otsu_dict.keys():
                    if paths in folder:
                        for tissues in tissue_list:
                            auto_otsu_dict[folder][f'{tissues} tifs'] = []
                            for tifs in os.listdir(folder):
                                if tissues in tifs and 'component_data' in tifs:
                                    # appends file path of tif to auto_otsu_dict
                                    auto_otsu_dict[folder][f'{tissues} tifs'].append(os.path.join(folder, tifs))
                # determining experiment type and adding threshold/pixels under the experiment in dict
                for i in range(0, len(thresh_experiment_df), 1):
                    if any(keywords in paths for keywords in titration_keywords):
                        if 'ihc' in thresh_experiment_df.iloc[i][thresh_experiment_df.columns[0]].lower():
                            true_threshold_dict = read_mifto('_', true_threshold_dict, thresh_experiment_df)
                        else:
                            true_threshold_dict = read_mifto('_1to', true_threshold_dict, thresh_experiment_df)
                    else:
                        print(
                            f'{paths} is an experiment type that does not have mIFTO or I do not care about, RIP Bozo')
    else:
        print(f'{paths} does not have a valid path')
        path_does_not_exist.append(paths)

# determining opal for each experiment and compiling tifs based on tissues used
for folder in auto_otsu_dict.keys():  # keys = path to a specific dilution folder (see mifto/FOP folder setup)
    if any(opals in folder for opals in opal_list) and 'IHC' not in folder:  # checking for opals, IHC is just in case
        folder_split = (folder.lower()).split('opal')[1]
        auto_otsu_dict[folder]['Opal'] = folder_split.split('_')[0]  # might be better to add somewhere else?
    else:
        auto_otsu_dict[folder]['Opal'] = 'IHC'
    # separates the tifs by tissue so they can be thresholded
    # for tissues in tissue_list:
    #     auto_otsu_dict[folder][f'{tissues} tifs'] = []
    #     for tifs in os.listdir(folder):
    #         if tissues in tifs and 'component_data' in tifs:
    #             # appends file path of tif to auto_otsu_dict
    #             auto_otsu_dict[folder][f'{tissues} tifs'].append(os.path.join(folder, tifs))

# Checking and correcting for repeating tifs
for k, v in auto_otsu_dict.items():  # k = path to a dilution folder, v = dict with lists for each tissue's tifs
    for tifs_list in v.values():
        if isinstance(tifs_list, list):  # the opal in the dict and errors
            for ind_tifs in tifs_list:
                if 'M' in ind_tifs.split('_')[-3] and '[' not in ind_tifs.split('_')[-3]:
                    delete_key = ind_tifs.split('_')[-3]  # should be something like 'M3'
                    split_tif = ind_tifs.split('_')
                    split_tif.remove(delete_key)
                    need_deleted_tif = '_'.join(split_tif)  # assumes we use highest M# tif
                    m_list = [need_deleted_tif, ind_tifs]
                    for i in range(int(delete_key.split('M')[1]) + 1, len(tifs_list), 1):  # iterates thru M#s
                        m_test_tif = ind_tifs.replace(f'{delete_key}', f'M{i}')
                        if m_test_tif in tifs_list:
                            m_list.append(m_test_tif)
                        else:
                            print(f'{ind_tifs} is the highest m-value')
                            break
                    sorted_m_tifs = sorted(m_list)  # sorts list in ascending order of M#
                    delete_list = sorted_m_tifs[1:]
                    for delete_me in delete_list:
                        tifs_list.remove(delete_me)  # deletes everything except highest M# from tifs in auto_otsu_dict

# initializing an array and changing the values over a specific area (new compiled array creation)
start_time = time.time()
for k, v in auto_otsu_dict.items():
    for tissue_tifs, tifs_list in v.items():
        if isinstance(tifs_list, list):
            tif_location_list = []
            counter = 0
            for tif_paths in tifs_list:
                print(tif_paths)  # keep track of progress
                if counter == 0:
                    layer = opal_layer_dict[auto_otsu_dict[k]['Opal']]
                    # reads one image and uses its shape to make a zeroed array
                    img = tifffile.imread(tif_paths)[layer]
                    shape = img.shape
                    # makes a zeroed array appropriate length for the # of tifs
                    compiled_tifs = np.zeros((img.shape[0] * img.shape[1] * len(tifs_list)), dtype='float32')
                    # flattens array to enable changing the zeros for the area a specific tif should take up
                    img = img.flatten()
                    # create an array that has all the non-zero values from the tif img
                    nonzero_values = img != 0
                    # replaces 0s starting at the beginning of the array until the value ==  to the size of the tif
                    compiled_tifs[:img.size][nonzero_values] = img[nonzero_values]
                    # creates a list containing each tif image boundary
                    tif_location_list.append(f'0:{img.size / shape[1]}')
                    counter += 1
                else:
                    layer = opal_layer_dict[auto_otsu_dict[k]['Opal']]
                    img = tifffile.imread(tif_paths)[layer]
                    img = img.flatten()
                    nonzero_values = img != 0
                    # assuming all the tifs are the same size, should give the next boundary for img placement
                    start_replace = (img.size * counter)
                    # end replace should be equal to start + size
                    end_replace = start_replace + img.size
                    tif_location_list.append(f'{start_replace / shape[1]}:{end_replace / shape[1]}')
                    # replacing zeros between boundaries specified with nonzero values from img
                    compiled_tifs[start_replace:end_replace][nonzero_values] = img[nonzero_values]
                    counter += 1
            # reshape the array by dividing by the length of each tif. Results in each img being vertically stacked
            compiled_tifs = compiled_tifs.reshape(-1, shape[1])
            # used for holding each tif img as its thresholded and connected pixels are applied
            mask_dict = {}
            # used to generate and store the true threshold and unblurred otsu threshold.
            # Also stores the total and per field positive pixels in true_threshold_dict
            if 'primary' in k.lower() and 'ihc' not in k.lower():
                split_number = -4
                pixels, true_threshold_dict = generate_nonblurred_threshold(split_number)
                true_threshold_dict = ksize_test(compiled_tifs, split_number)
            else:
                split_number = -1
                pixels, true_threshold_dict = generate_nonblurred_threshold(split_number)
                true_threshold_dict = ksize_test(compiled_tifs, split_number)

end_time = time.time()
print(f'This took {end_time - start_time} seconds to complete')


# export data that's in true_threshold_dict
# establish column headers
headers = ['experiment', 'condition', 'true_threshold', 'sigma', 'ksize', 'otsu_threshold',
           'percent_threshold_difference', 'total_true_num_of_positive_pixels', 'true_fop', 'total_otsu_num_of_positive_pixels',
           'otsu_fop', 'total_percent_positive_pixel_error_from_true', 'total_percent_positive_pixel_difference_from_true',
           'true_positive_pixels_per_field', 'otsu_positive_pixels_per_field', 'percent_positive_pixel_error_per_field',
           'percent_positive_pixel_diff_per_field']
export_list = []
for k, v in true_threshold_dict.items():  # k = path to experiment folder, v = dict for each condition (TB3_1to100)
    for keys in v.keys():  # keys = list of conditions in the experiment (TB3_1to100)
        # want to ignore strings like filepaths that are stored in the dict
        if isinstance(true_threshold_dict[k][keys], str):
            continue
        else:
            # condition = sigma/ksize variable, threshold_data gives the dict of thresholds and the connected pixels
            for condition, threshold_data in true_threshold_dict[k][keys].items():
                if isinstance(threshold_data, dict):  # to ignore connected pixels
                    if len(threshold_data) > 1:  # only look at conditions that ksize tests were run
                        if condition == 'True':
                            # makes variables to be exported
                            total_true_positive_pixels = threshold_data['Positive Pixels']
                            true_threshold = threshold_data['Threshold']
                            true_pixels_per_field = threshold_data['Positive Pixels Per Field']
                            # string of positive pixels per field separated by '_'
                            true_str = '_'.join(map(str, true_pixels_per_field))
                        if 'Sigma' and ',' in condition:
                            total_otsu_positive_pixels = threshold_data['Positive Pixels']
                            otsu_threshold = threshold_data['Threshold']
                            otsu_pixels_per_field = threshold_data['Positive Pixels Per Field']
                            # string of positive pixels per field separated by '_'
                            otsu_str = '_'.join(map(str, otsu_pixels_per_field))
                            err_list = []
                            diff_list = []
                            for i in range(len(true_pixels_per_field)):
                                # iterate through pixels per field and get the percent error and difference
                                # percent error
                                err_list.append(((abs((float(otsu_pixels_per_field[i]) - float(true_pixels_per_field[i]))) /
                                                  float((true_pixels_per_field[i]) + 1)) * 100))
                                # percent difference
                                diff_list.append((abs((float(otsu_pixels_per_field[i]) - float(true_pixels_per_field[i]))) /
                                                  ((float(otsu_pixels_per_field[i]) + float((true_pixels_per_field[i]) + 1)) / 2)) * 100)
                                # convert the values for each field to a string, then combine
                                err_str = '_'.join(map(str, err_list))
                                diff_str = '_'.join(map(str, diff_list))
                            export_list.append(
                                # experiment path , condition(TB3_1to100), true threshold
                                f'{k.split("Results.pixels")[0]},{keys},{true_threshold},'
                                # sigma value, ksize value
                                f'{condition.split("Sigma")[0]}, {(condition.split(",")[1]).split("ksize")[0]},'
                                # otsu threshold, percent threshold difference
                                f' {otsu_threshold}, {round(((1 - (otsu_threshold / true_threshold)) * 100), 2)},'
                                # total positive pixels with true threshold, total FOP with true threshold
                                f' {total_true_positive_pixels}, {total_true_positive_pixels / compiled_tifs.size},'
                                # total positive pixels with otsu threshold, total FOP with otsu threshold
                                f' {total_otsu_positive_pixels}, {total_otsu_positive_pixels / compiled_tifs.size},'
                                # total positive pixel error from true
                                f' {(abs((total_otsu_positive_pixels - total_true_positive_pixels)) / total_true_positive_pixels) * 100},'
                                # total positive pixel difference from true
                                f'{(abs((total_otsu_positive_pixels - total_true_positive_pixels)) / ((total_otsu_positive_pixels + total_true_positive_pixels) / 2)) * 100},'
                                # positive pixels per field for true, otsu, % err per field, % difference per field
                                f'{true_str}, {otsu_str}, {err_str}, {diff_str}')
# testing if the file is open
try:
    os.rename(os.path.join(r"C:\Users\gsalvuc1\Desktop", "Sigma_Ksize_Tests.csv"),
              os.path.join(r"C:\Users\gsalvuc1\Desktop", "Sigma_Ksize_Tests.csv").replace('Ksize', 'ksize'))
except PermissionError:
    print('close the csv file')
    time.sleep(5)
# writing to a csv, first row is the headers, rest is the data per condition
with open(os.path.join(r"C:\Users\gsalvuc1\Desktop", "Sigma_Ksize_Tests.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in export_list:
        writer.writerow(row.split(','))
# end timer and calculate how long it took to complete
end = time.time()
print(f'This took {end - start} seconds to complete')


# making violinplots, boxplots, and tables for each sigma/ksize per tissue, per dilution
start = time.time()
data_csv = pd.read_csv(r"C:\Users\gsalvuc1\Desktop\Sigma_ksize_Tests.csv")
# gets the 'experiment' column from the dataframe
experiment_name = set(data_csv['experiment'])
# generate empty lists for later use
for experiments in experiment_name:
    exp_df = data_csv.loc[data_csv['experiment'] == experiments]
    # gets the 'condition' column from the dataframe
    exp_condition = exp_df['condition']
    sorted_exp_condition = []  # can probably delete this
    perc_error_data = []
    perc_diff_data = []
    dilution_list = []
    tissue_list = []
    diff_big_df = pd.DataFrame()
    err_big_df = pd.DataFrame()
    for items in exp_condition:
        # generating a list of all conditions, I think this isn't used anymore
        if items not in sorted_exp_condition:
            sorted_exp_condition.append(''.join(items))
        # generating a dilution list helpful for global graphs
        if items.split('_')[1] not in dilution_list:
            dilution_list.append(items.split('_')[1])
        # generating a tissue list helpful for single tissue condition graphs
        if items.split('_')[0] not in tissue_list:
            tissue_list.append(items.split('_')[0])
    # was used to determine limit per page for subplots, but isn't used anymore
    for items in sorted_exp_condition:
        split_items = items.split('_')[0]
        break
    # placer_limit = (str(sorted_exp_condition).count(split_items) + 1) * 2
    subplot_placer = 1
    # create a pdf named the same as the experiment folder (aMSA_Primary_Titration_08.23.2023)
    filename = os.path.join(r"C:\Users\gsalvuc1\Desktop", f"{experiments.split('\\')[2]}_{experiments.split('\\')[3]}.pdf")
    pdf = PdfPages(filename)
    # generate all possible combinations of used sigma (s) and ksizes (k)
    s = [1, 2, 3]
    k = [1, 3, 5, 7]
    combo_list = list(product(s, k))
    for combos in combo_list:
        ksize = combos[1]
        sigma = combos[0]
        # make a smaller dataframe from the original with only things of desired ksize
        ksize_filter = exp_df.loc[exp_df['ksize'] == combos[1]]
        # make even smaller dataframe from previous one with only things of desired sigma and ksize
        final_df = ksize_filter.loc[ksize_filter['sigma'] == combos[0]]
        # generating data structures for global graphs (all tissues combined per dilution per sigma/ksize combo)
        global_error_list = []
        global_diff_list = []
        condition_list = []
        # used for generating the tables
        diff_summary_stats_dict = {}
        err_summary_stats_dict = {}
        # iterate through dilutions used and query dataframe for conditions matching them
        for dilutions in dilution_list:
            err_temp_list = []
            diff_temp_list = []
            for i in range(len(final_df)):
                if dilutions == final_df.iloc[i]['condition'].split('_')[1]:
                    for diffs in final_df.iloc[i]['percent_positive_pixel_diff_per_field'].split('_'):
                        diff_temp_list.append(float(diffs))
                    for errs in final_df.iloc[i]['percent_positive_pixel_error_per_field'].split('_'):
                        err_temp_list.append(float(errs))
                    condition = final_df.iloc[i]['condition']
                    threshold = final_df.iloc[i]['true_threshold']
                    antibody = final_df.iloc[i]['experiment'].split('\\')[2]
                    condition_list.append(dilutions)
            # each diltuion gets a separate list in the global lists
            # cull_outliers(diff_temp_list)
            # cull_outliers(err_temp_list)
            global_diff_list.append(diff_temp_list)
            global_error_list.append(err_temp_list)
        # break
            # try to find the opal used for the experiment
            if auto_otsu_dict:  # to allow for running of only this part without running the whole script
                for keys in auto_otsu_dict.keys():
                    if (keys.split('\\Data')[0]) == experiments:
                        opal = auto_otsu_dict[keys]['Opal']
            else:
                # looks at the actual folders in the server compared to the auto_otsu_dict above
                if os.path.exists(os.path.join(experiments, 'Data')):
                    for folder in os.listdir(os.path.join(experiments, 'Data')):
                        if any(opals in folder for opals in opal_list) and 'IHC' not in folder:
                            opal = folder.split('_')[-2]
                else:
                    opal = 'Not Found'
        # plotting global graphs
        plt.subplot(3, 2, 1)  # hard coded bc Ben said I could
        # violin plots for global percent error and difference
        sns.violinplot(global_error_list, inner=None)  # plotting a list of lists to get multiple violins on one plot
        # sns.swarmplot(y=perc_error_list, color='k', alpha=0.5)
        plt.title(f'Violin Plot of % Error Per Field', fontsize='x-small')
        plt.xticks([])  # removing x-ticks to make it look cleaner
        plt.tick_params(labelsize=6)  # size picked based off trial and error, makes it look clean
        # subplot_placer += 1
        plt.subplot(3, 2, 2)
        # inner allows for points or boxes inside
        sns.violinplot(global_diff_list, inner=None)
        plt.title(f'Violin Plot of % Difference Per Field', fontsize='x-small')
        plt.xticks([])
        plt.tick_params(labelsize=6)
        plt.subplot(3, 2, 3)
        # boxplots for global percent error and difference
        # arguments are changing what outliers look like
        plt.boxplot(global_error_list, flierprops={'marker': '.', 'markersize': '2', 'markerfacecolor': 'red'}, vert=True)
        # sets number of x ticks equal to number of dilutions and labels them with the dilutions
        plt.xticks(ticks=[i for i in range(1, len(dilution_list) + 1)], labels=dilution_list, rotation='horizontal')
        plt.title('Boxplot of % Error Per Field', fontsize='x-small')
        plt.tick_params(labelsize=6)
        plt.subplot(3, 2, 4)
        plt.boxplot(global_diff_list, flierprops={'marker': '.', 'markersize': '2', 'markerfacecolor': 'red'}, vert=True)
        plt.xticks(ticks=[i for i in range(1, len(dilution_list) + 1)], labels=dilution_list, rotation='horizontal')
        plt.title('Boxplot of % Difference Per Field', fontsize='x-small')
        plt.tick_params(labelsize=6)
        plt.suptitle(f'{antibody} {opal} Global Data \u03C3={sigma} k={ksize}')  # /u03C3 is the unicode for sigma signal
        # making tables using a dictionary and dataframe
        err_count = 0
        # make columns for percent error global data
        for ind_err_list in global_error_list:
            # get mean, standard dev, median, and IQR
            err_summary_stats_dict.update(Mean=f'{round(np.average(ind_err_list), 2)}',
                                  SD=f'{round(np.std(ind_err_list), 2)}',
                                  Median=f'{round(np.median(ind_err_list), 2)}',
                                  IQR=f'{round(np.percentile(ind_err_list, 75) - np.percentile(ind_err_list, 25), 2)}')
            # dict -> dataframe
            err_mini_df = pd.DataFrame.from_dict(err_summary_stats_dict, orient='index', columns=[f'{dilution_list[err_count].split('_')[0]}'])
            # add a column for x axis labels
            if len(err_big_df.columns) == 0:
                err_mini_df.insert(0, '', ['\u03BC', 'SD', 'Med', 'IQR'])
            # add column to a bigger dataframe
            err_big_df = pd.concat((err_big_df, err_mini_df), axis=1)
            err_count += 1
        # make columns for percent difference global data table
        diff_count = 0
        for ind_diff_list in global_diff_list:
            diff_summary_stats_dict.update(Mean=f'{round(np.average(ind_diff_list), 2)}',
                                  SD=f'{round(np.std(ind_diff_list), 2)}',
                                  Median=f'{round(np.median(ind_diff_list), 2)}',
                                  IQR=f'{round(np.percentile(ind_diff_list, 75) - np.percentile(ind_diff_list, 25), 2)}')
            diff_mini_df = pd.DataFrame.from_dict(diff_summary_stats_dict, orient='index', columns=[f'{dilution_list[diff_count].split('_')[0]}'])
            if len(diff_big_df.columns) == 0:
                diff_mini_df.insert(0, '', ['\u03BC', 'SD', 'Med', 'IQR'])
            diff_big_df = pd.concat((diff_big_df, diff_mini_df), axis=1)
            diff_count += 1
        # placing the tables on the figure
        plt.subplot(3, 2, 5)
        cell_text = []
        for row in range(len(err_big_df)):
            cell_text.append(err_big_df.iloc[row])
        table = plt.table(cellText=cell_text, colLabels=err_big_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)  # trial and error, makes table fit and look clean
        table.auto_set_column_width(range(1, len(err_big_df.columns)))
        plt.axis('off')
        plt.subplots_adjust(left=.1, right=.9)  # change placement of tables to allow them to fit without overlap
        plt.subplot(3, 2, 6)
        cell_text = []
        for row in range(len(diff_big_df)):
            cell_text.append(diff_big_df.iloc[row])
        table = plt.table(cellText=cell_text, colLabels=diff_big_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.auto_set_column_width(range(1, len(diff_big_df.columns)))
        plt.axis('off')
        plt.subplots_adjust(left=.1, right=.9)
        pdf.savefig(bbox_inches='tight')
        err_big_df = pd.DataFrame()
        diff_big_df = pd.DataFrame()
        plt.clf()
        plt.close()
        perc_diff_data = []
        perc_error_data = []
        break
    # this is for each tissue per sigma/ksize per dilution
    for combos in combo_list:
        # major differnce is that there's no global list, otherwise follows same logic as above
        ksize = combos[1]
        sigma = combos[0]
        ksize_filter = exp_df.loc[exp_df['ksize'] == combos[1]]
        final_df = ksize_filter.loc[ksize_filter['sigma'] == combos[0]]
        for tissues in tissue_list:  # sorting by tissues instead dilutions
            perc_error_list = []
            perc_diff_list = []
            condition_list = []
            diff_summary_stats_dict = {}
            err_summary_stats_dict = {}
            for i in range(len(final_df)):
                if tissues == final_df.iloc[i]['condition'].split('_')[0]:
                    perc_diff_list.append(list(map(float, final_df.iloc[i]['percent_positive_pixel_diff_per_field'].split('_'))))
                    perc_error_list.append(list(map(float, final_df.iloc[i]['percent_positive_pixel_error_per_field'].split('_'))))
                    condition = final_df.iloc[i]['condition']
                    threshold = final_df.iloc[i]['true_threshold']
                    antibody = final_df.iloc[i]['experiment'].split('\\')[2]
                    condition_list.append((condition.split('_')[1]))
            # break
                if auto_otsu_dict:
                    for keys in auto_otsu_dict.keys():
                        if (keys.split('\\Data')[0]) == experiments:
                            opal = auto_otsu_dict[keys]['Opal']
                else:
                    if os.path.exists(os.path.join(experiments, 'Data')):
                        for folder in os.listdir(os.path.join(experiments, 'Data')):
                            if any(opals in folder for opals in opal_list) and 'IHC' not in folder:
                                opal = folder.split('_')[-2]
                    else:
                        opal = 'Not Found'
            plt.subplot(3, 2, 1)
            sns.violinplot(perc_error_list, inner='point')
            plt.title(f'Violin Plot of % Error Per Field', fontsize='x-small')
            plt.xticks([])
            plt.tick_params(labelsize=6)
            plt.subplot(3, 2, 2)
            sns.violinplot(perc_diff_list, inner='point')
            plt.title(f'Violin Plot of % Difference Per Field', fontsize='x-small')
            plt.xticks([])
            plt.tick_params(labelsize=6)
            plt.subplot(3, 2, 3)
            plt.boxplot(perc_error_list, flierprops={'marker': '.', 'markersize': '2', 'markerfacecolor': 'red'}, vert=True)
            plt.xticks(ticks=[i for i in range(1, len(condition_list) + 1)], labels=condition_list, rotation='horizontal')
            plt.title('Boxplot of % Error Per Field', fontsize='x-small')
            plt.tick_params(labelsize=6)
            plt.subplot(3, 2, 4)
            plt.boxplot(perc_diff_list, flierprops={'marker': '.', 'markersize': '2', 'markerfacecolor': 'red'}, vert=True)
            plt.xticks(ticks=[i for i in range(1, len(condition_list) + 1)], labels=condition_list, rotation='horizontal')
            plt.title('Boxplot of % Difference Per Field', fontsize='x-small')
            plt.tick_params(labelsize=6)
            plt.suptitle(f'{tissues} {antibody} {opal}  \u03C3={sigma} k={ksize}')
            # making tables, same as above, but also includes Manual Threshold
            # percent error table
            err_count = 0
            for ind_err_lists in perc_error_list:
                err_summary_stats_dict.update(Mean=f'{round(np.average(ind_err_lists), 2)}',
                                      SD=f'{round(np.std(ind_err_lists), 2)}',
                                      Median=f'{round(np.median(ind_err_lists), 2)}',
                                      IQR=f'{round(np.percentile(ind_err_lists, 75) - np.percentile(ind_err_lists, 25), 2)}',
                                      ManualThreshold=threshold)
                err_mini_df = pd.DataFrame.from_dict(err_summary_stats_dict, orient='index', columns=[f'{condition_list[err_count].split('_')[0]}'])
                if len(err_big_df.columns) == 0:
                    err_mini_df.insert(0, '', ['\u03BC', 'SD', 'Med', 'IQR', 'Th.'])
                err_big_df = pd.concat((err_big_df, err_mini_df), axis=1)
                err_count += 1
            # percent diff table
            diff_count = 0
            for ind_diff_lists in perc_diff_list:
                diff_summary_stats_dict.update(Mean=f'{round(np.average(ind_diff_lists), 2)}',
                                      SD=f'{round(np.std(ind_diff_lists), 2)}',
                                      Median=f'{round(np.median(ind_diff_lists), 2)}',
                                      IQR=f'{round(np.percentile(ind_diff_lists, 75) - np.percentile(ind_diff_lists, 25), 2)}',
                                      ManualThreshold=threshold)
                diff_mini_df = pd.DataFrame.from_dict(diff_summary_stats_dict, orient='index', columns=[f'{condition_list[diff_count].split('_')[0]}'])
                if len(diff_big_df.columns) == 0:
                    diff_mini_df.insert(0, '', ['\u03BC', 'SD', 'Med', 'IQR', 'Th.'])
                diff_big_df = pd.concat((diff_big_df, diff_mini_df), axis=1)
                diff_count += 1
            plt.subplot(3, 2, 5)
            # can probably make this a function to save some lines and get a cleaner feel
            cell_text = []
            for row in range(len(err_big_df)):
                cell_text.append(err_big_df.iloc[row])
            table = plt.table(cellText=cell_text, colLabels=err_big_df.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.auto_set_column_width(range(1, len(err_big_df.columns)))
            plt.axis('off')
            plt.subplots_adjust(left=.1, right=.9)
            plt.subplot(3, 2, 6)
            cell_text = []
            for row in range(len(diff_big_df)):
                cell_text.append(diff_big_df.iloc[row])
            table = plt.table(cellText=cell_text, colLabels=diff_big_df.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.auto_set_column_width(range(1, len(diff_big_df.columns)))
            plt.axis('off')
            plt.subplots_adjust(left=.1, right=.9)
            pdf.savefig(bbox_inches='tight')
            err_big_df = pd.DataFrame()
            diff_big_df = pd.DataFrame()
            plt.clf()
            plt.close()
            perc_diff_data = []
            perc_error_data = []
    pdf.close()
    end = time.time()
    print(f'This took {end-start} seconds')


# mask generation
comp_tif = tifffile.imread(
    r"C:\Users\gsalvuc1\Desktop\use me\CM48_IHC\CM48_TCRDelta_1to150_[8920,54390]_component_data.tif")
color_tif = tifffile.imread(
    r"C:\Users\gsalvuc1\Desktop\use me\Data\FOP\aSMA_IHC\TN_aSMA_IHC_JHUPolaris_1_[11090,43537]_component_data.tif")
layer_img = comp_tif[opal_layer_dict['IHC']]
layer_img_copy = np.copy(layer_img)
mask_array = (layer_img_copy >= .33)
mask_array_with_filter = imsize_filter(mask_array, 15)
# mask_3_channel = cv2.cvtColor(mask_array_with_filter, cv2.COLOR_GRAY2RGB)
# masked_img = np.copy(color_tif)
# masked_img[np.where((mask_3_channel == [1, 1, 1]).all(axis=2))] = (255, 0, 0)
my_positive_pixels = mask_array_with_filter.sum()
file_dest = os.path.join(r"C:\Users\gsalvuc1\Desktop", 'TB3_aSMA_1to100_[16660,38418].png')
cv2.imwrite(file_dest, cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))


def make_violinplot(data_list, title, num_subplots=0, vp_color=None, inner=None):
    sns.violinplot(data_list, color=vp_color, inner=inner)
    if num_subplots < 9:
        plt.title(title, fontsize='x-small')
        plt.tick_params(labelsize=8)
    else:
        plt.title(title, fontsize='xx-small')
        plt.tick_params(labelsize=4)
    plt.xticks([])
    return

