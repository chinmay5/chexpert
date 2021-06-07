import os
import pandas as pd
from tqdm import tqdm

import numpy as np

from environment_setup import PROJECT_ROOT_DIR


data_dir = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0")

def load_data():
    sub_dirs = [os.path.join(data_dir + '/files', x) for x in os.listdir(data_dir + '/files')]
    sub_dirs = [x for x in sub_dirs if os.path.isdir(x)]  # filter out 'index.html'

    patient_dirs = []
    patient_ids = []

    for sub_dir in sub_dirs:
        temp_dirs = os.listdir(sub_dir)
        for x in temp_dirs:
            if '.html' not in x:
                patient_dirs.append(os.path.join(sub_dir, x))
                patient_ids.append(x)

    print('# patients:', len(patient_ids))


def get_images_dict(study_dir):
    # -- metadata
    metadata_file = os.path.join(data_dir, 'mimic-cxr-2.0.0-metadata.csv')
    df_metadata = pd.read_csv(metadata_file)
    im_paths = [os.path.join(study_dir, x) for x in os.listdir(study_dir) if x.endswith('.jpg')]
    images_dict = {}
    for im_path in im_paths:
        dicom_id = os.path.splitext(os.path.basename(im_path))[0]
        im_view = df_metadata.loc[df_metadata['dicom_id'] ==dicom_id]['ViewPosition'].values[0]
        images_dict[dicom_id] = {
            'path': im_path,
            'view': im_view
        }
    return images_dict


def get_labels(patient_id, study_id):
    # -- chexpert labels
    labels_file_chexpert = os.path.join(data_dir, 'mimic-cxr-2.0.0-chexpert.csv')
    df_chexpert_labels = pd.read_csv(labels_file_chexpert)
    label_keys = list(df_chexpert_labels.columns[2:])
    print('LABEL_KEYS:', label_keys)
    df_temp_patient = df_chexpert_labels.loc[df_chexpert_labels['subject_id'] == int(patient_id[1:])]
    labels = df_temp_patient.loc[df_temp_patient['study_id'] == int(study_id[1:])].values[0][2:]  # remove 's' from study id
    return labels


def get_report_path(study_id):
    # -- report paths
    reports_dir = os.path.join(data_dir, 'mimic-cxr-reports')
    report_paths = []
    report_ids = []

    for root, dirs, files in os.walk(reports_dir):
        for file in files:
            if file.endswith(".txt"):
                report_paths.append(os.path.join(root, file))
                report_ids.append(os.path.splitext(file)[0])

    index = report_ids.index(study_id)
    return report_paths[index]


def get_error_ids(patient_ids, patient_dirs):
    error_ids = []
    mimic_dict = {}
    for idx, (patient_id, patient_dir) in enumerate(zip(patient_ids, patient_dirs)):
        # path
        mimic_dict[patient_id] = {'path': patient_dir}
        try:
            # study dict
            study_ids = os.listdir(patient_dir)
            studies_dict = {}

            for study_id in study_ids:
                study_dir = os.path.join(patient_dir, study_id)
                if os.path.isdir(study_dir):
                    studies_dict[study_id] = {}

                    studies_dict[study_id]['images'] = get_images_dict(study_dir)
                    studies_dict[study_id]['labels'] = get_labels(patient_id, study_id)
                    studies_dict[study_id]['report_path'] = get_report_path(study_id)

            mimic_dict[patient_id]['studies'] = studies_dict

        except Exception as e:
            print('--- {} ---'.format(idx))
            print(patient_id)
            print(e)
            error_ids.append(patient_id)


def build_mimic_graph(train_data_items, LABELS):
    complete_label_list = []
    # Iterate over all the labels present in the form of a list of tuple
    for _, label in tqdm(train_data_items):
        label_arr_np = np.array(label)
        complete_label_list.append(label_arr_np)
    labels_mat = np.stack(complete_label_list)
    co_occur_mat = np.zeros((len(LABELS), len(LABELS)))
    num_samples = labels_mat.shape[0]
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i == j:
                # along the diagonal, save its frequency
                co_occur_mat[i, j] = np.sum(labels_mat[:, i])
            else:
                # temp only has the columns for 'i' and 'j' labels
                a = np.zeros_like(labels_mat)
                a[:, i] = labels_mat[:, i]
                a[:, j] = labels_mat[:, j]

                # is np.sum(temp, -1) = 2, i and j co-occur
                b = np.sum(a, -1)
                freq_ij = np.sum(b == 2)
                # Because of symmetry of the matrix
                co_occur_mat[i, j] = freq_ij

    # We need to convert the co-occurrence matrix into prob matrix
    prob_mat = co_occur_mat / num_samples  # eqn 5
    prob_i = np.diag(co_occur_mat).reshape(1, -1) / num_samples  # row vector, eqn 6
    prob_j = np.diag(co_occur_mat).reshape(-1, 1) / num_samples  # column vector, eqn 6
    pmi = np.log(prob_mat/(prob_i * prob_j))
    # Remove the edges that occur less than expected value
    pmi[pmi < 0] = 0
    # Next we enter diagonal values
    for i in range((len(LABELS))):
        pmi[i, i] = 1  # eqn 3

    return co_occur_mat, pmi


if __name__ =='__main__':
    load_data()