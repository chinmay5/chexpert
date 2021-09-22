import os
import csv
import pickle
import re
from collections import defaultdict

import numpy as np

import pandas as pd
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR

extra_train_labels = [
    "thorax", "heart atria", "heart", "instruments", "heart ventricles", "aorta", "cough", "mediastinum",
    "trachea", "exudates", "spine", "diaphragm", "aorta thoracic", "ribs", "pleura", "thoracic wall",
    "abdomen", "chest tubes", "thoracic cavity", "lung", "abdominal cavity", "sternotomy"
]

# Let us create a massive dictionary where everything is saved
extra_labels_dict = dict()
# Also, the key of this dictionary has a fixed format
p =re.compile("p\d{3,}\/s\d+")
# /home/chinmayp/workspace/chexpert/dataset/physionet.org/files/mimic-cxr/2.0.0/files
data_dir = os.path.join(PROJECT_ROOT_DIR, "dataset", "physionet.org", "files", "mimic-cxr", "2.0.0")
image_dir = os.path.join(data_dir, "files")
# Unfortunately I have placed the text based dataset and the images separately. We can not use them together
split_file = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-split.csv")
view_file = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-metadata.csv")
labels_file = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-chexpert.csv")


def get_data_splits():
    split_df = pd.read_csv(split_file)
    # Filter away the samples that do not have at least 2 images
    # split_df = split_df[split_df['study_id'].map(split_df['study_id'].value_counts()) >= 2]
    # split_df = split_df.groupby(['study_id', 'subject_id']).filter(lambda x: len(x) >= 2)
    # We also need to the dictionary mentioning views so that we select only frontal and lateral
    view_df = pd.read_csv(view_file)
    valid_sample_df = view_df.loc[view_df.ViewPosition.isin(['AP', 'LATERAL', 'PA'])]
    # Now we split them into train-val-test
    train_dict = defaultdict(lambda: defaultdict(list))
    valid_dict = defaultdict(lambda: defaultdict(list))
    test_dict = defaultdict(lambda: defaultdict(list))
    for idx, row in tqdm(split_df.iterrows()):
        # Let us check if the study_id has a lateral view or not.
        if not row['study_id'] in valid_sample_df.study_id.values:
            continue
        if row['split'] == 'train':
            train_dict[row.subject_id][row.study_id].append(row['dicom_id']+"_"+row['ViewPosition']+".jpg")
        elif row['split'] == 'validate':
            valid_dict[row.subject_id][row.study_id].append(row['dicom_id']+"_"+row['ViewPosition']+".jpg")
        elif row['split'] == 'test':
            test_dict[row.subject_id][row.study_id].append(row['dicom_id']+"_"+row['ViewPosition']+".jpg")
        else:
            print(row)
            raise AttributeError("Invalid split type")
    # Let us remove all the entries that do not have two images. So, in our case, the list would have a length of 2.
    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp"), exist_ok=True)
    select_images_with_2_views(input_dictionary=train_dict, valid_sample_df=valid_sample_df)
    select_images_with_2_views(input_dictionary=valid_dict, valid_sample_df=valid_sample_df)
    select_images_with_2_views(input_dictionary=test_dict, valid_sample_df=valid_sample_df)
    # Now we add the target labels as well
    add_target_labels(input_dictionary=train_dict)
    add_target_labels(input_dictionary=valid_dict)
    add_target_labels(input_dictionary=test_dict)
    pickle.dump(dict(train_dict), open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "train.pkl"), "wb"))
    pickle.dump(dict(valid_dict), open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "valid.pkl"), "wb"))
    pickle.dump(dict(test_dict), open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "test.pkl"), "wb"))


def select_images_with_2_views(input_dictionary, valid_sample_df):
    samples_trimmed = 0
    for subject_id, study_dict in tqdm(input_dictionary.copy().items()):
        for study_id, jpeg_list in study_dict.copy().items():
            if len(jpeg_list) < 2:
                study_dict.pop(study_id)
            # if the list has more elements, select only the AP/PA and Lateral
            elif len(jpeg_list) >= 3:
                samples_trimmed += 1
                new_curtailed_list = []
                has_lateral, has_frontal = False, False
                for dicom_id in jpeg_list:
                    canidate_view_pos = valid_sample_df.loc[valid_sample_df.dicom_id == dicom_id[:-4]].ViewPosition.values.tolist()
                    if 'LATERAL' in canidate_view_pos and not has_lateral:
                        new_curtailed_list.append(dicom_id)
                        has_lateral = True
                    elif ('AP' in canidate_view_pos or 'PA' in canidate_view_pos) and not has_frontal:
                        new_curtailed_list.append(dicom_id)
                        has_frontal = True
                    if has_frontal and has_lateral:
                        study_dict[study_id] = new_curtailed_list
                        break
                # We iterated through the candidates which are more than three but either frontal or lateral is missing
                if not has_frontal or not has_lateral:
                    study_dict.pop(study_id)

        # If removal of study_id caused study_dict to become empty, we need to remove it as well
        remove_empty_key(input_dictionary, subject_id)
    print(f"{samples_trimmed} for input dictionary")


def remove_empty_key(input_dictionary, subject_id):
    if len(input_dictionary[subject_id]) == 0:
        input_dictionary.pop(subject_id)


def get_train_study_ids(train_list):
    study_ids = set()
    patient_ids = set()
    for train_item in train_list:
        study_ids.add("s" + str(train_item['study_id']))
        patient_ids.add("p" + str(train_item['subject_id']))
    return patient_ids, study_ids


def process_files(study_id_txt_file):
    labels = [0 for _ in range(len(extra_train_labels))]
    with open(study_id_txt_file, 'r') as file:
        data = file.read().replace('\n', '').lower()
        for idx, name in enumerate(extra_train_labels):
            # Some of the labels contain multiple words. So, we try
            # an AND condition to select them
            if all([data.find(x) != -1 for x in name.split()]):
                labels[idx] = 1
    # Let us create an entry in our massive dictionary
    key = p.search(study_id_txt_file)
    extra_labels_dict[key.group()] = labels
    # For each file we need to create a csv file
    # dest_filename = txt_file[:txt_file.find('.txt')] + '.csv'
    # with open(dest_filename, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(extra_labels)
    #     writer.writerow(labels)

def add_target_labels(input_dictionary):
    df_chexpert_labels = pd.read_csv(labels_file)
    label_keys = list(df_chexpert_labels.columns[2:])
    print('LABEL_KEYS:', label_keys)
    for patient_id, study_dict in tqdm(input_dictionary.copy().items()):
        for study_id, img_list in study_dict.copy().items():
            try:
                labels = df_chexpert_labels.loc[(df_chexpert_labels.subject_id == patient_id) &
                                                (df_chexpert_labels.study_id == study_id)]
                # Replace NA values with zeros
                labels = labels.fillna(0)
                # Now just take the labels and ignore patient_id, study_id
                labels = labels.values[0][2:]  # Index 0 taken since it is a dataframe
                # Replace the uncertainty labels of -1 with a 0
                labels[labels == -1] = 0
                # Add this to the image list as the last element
                img_list.append(labels.tolist())
            except Exception:
                print(f"No entry for p{patient_id}/s{study_id}. Deleting the entry")
                study_dict.pop(study_id)
                remove_empty_key(input_dictionary=input_dictionary, subject_id=patient_id)


def process_patients(patient_id, valid_study_ids):
    base_id = patient_id[:3]
    for study_id in valid_study_ids:
        study_id_txt_file = os.path.join(image_dir, base_id, patient_id, f"s{study_id}.txt")
        process_files(study_id_txt_file=study_id_txt_file)


def store_extra_labels_for_training_data():
    print("Generating extra labels for our training Graph")
    train_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "train.pkl"), "rb"))
    for patient_id, study_dict in tqdm(train_dict.items()):
        process_patients(patient_id=f"p{patient_id}", valid_study_ids=study_dict.keys())


if __name__ == '__main__':
    # get_data_splits()
    store_extra_labels_for_training_data()
    pickle.dump(extra_labels_dict, open(os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "extra_label.pkl"), "wb"))
    label_list = []
    for k, val in extra_labels_dict.items():
        label_list.append(np.array(val))
    zz = np.stack(label_list)
    print(zz.shape)
