"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Root class for X-ray data sets.
    The base data seet logs parameters as attributes, reducing code duplication
    across the various public X-ray data loaders.
    Args:
        dataset_name: Name of the dataset.
        directory: Location of the data.
        split: One of ('train', 'val', 'test', 'all').
        label_list: A list of labels for the data loader to extract.
        subselect: Argument to pass to `pandas` subselect.
        transform: A set of data transforms.
    """

    def __init__(
            self,
            dataset_name: str,
            directory: Union[str, os.PathLike],
            split: str,
            label_list: Union[str, List[str]],
            subselect: Optional[str],
            transform: Optional[Callable],
    ):
        self.dataset_name = dataset_name

        split_list = ["train", "val", "test", "all"]

        if split not in split_list:
            raise ValueError("split {} not a valid split".format(split))

        self.directory = Path(directory)
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.subselect = subselect
        self.transform = transform
        self.metadata_keys: List[str] = []

    def preproc_csv(self, csv: pd.DataFrame, subselect: str) -> pd.DataFrame:
        if subselect is not None:
            csv = csv.query(subselect)

        return csv

    def open_image(self, path: Union[str, os.PathLike]) -> Image:
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("F")

    def __len__(self) -> int:
        return 0

    @property
    def calc_pos_weights(self) -> float:
        if self.csv is None:
            return 0.0

        pos = (self.csv[self.label_list] == 1).sum()
        neg = (self.csv[self.label_list] == 0).sum()

        neg_pos_ratio = (neg / np.maximum(pos, 1)).values.astype(np.float)

        return neg_pos_ratio

    def retrieve_metadata(
            self, idx: int, filename: Union[str, os.PathLike], exam: pd.Series
    ) -> Dict:
        metadata = {}
        metadata["dataset_name"] = self.dataset_name
        metadata["dataloader class"] = self.__class__.__name__
        metadata["idx"] = idx  # type: ignore
        for key in self.metadata_keys:
            # cast to string due to typing issues with dataloader
            metadata[key] = str(exam[key])
        metadata["filename"] = str(filename)

        metadata["label_list"] = self.label_list  # type: ignore

        return metadata

    def __repr__(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    @property
    def labels(self) -> Union[str, List[str]]:
        return self.label_list


class MimicCxrJpgDataset(BaseDataset):
    """
    Data loader for MIMIC CXR data set.
    Args:
        directory: Base directory for data set.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
                'test': Include testing split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composible transform list to be applied to the data.
    """

    def __init__(self, directory, split="train", label_list="all", subselect=None, transform=None):
        super().__init__(
            "mimic-cxr-jpg", directory, split, label_list, subselect, transform
        )

        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "dicom_id",
            "subject_id",
            "study_id",
            "PerformedProcedureStepDescription",
            "ViewPosition",
            "Rows",
            "Columns",
            "StudyDate",
            "StudyTime",
            "ProcedureCodeSequence_CodeMeaning",
            "ViewCodeSequence_CodeMeaning",
            "PatientOrientationCodeSequence_CodeMeaning",
        ]

        self.label_csv_path = (
                self.directory / "2.0.0" / "mimic-cxr-2.0.0-chexpert.csv.gz"
        )
        self.meta_csv_path = (
                self.directory / "2.0.0" / "mimic-cxr-2.0.0-metadata.csv.gz"
        )
        self.split_csv_path = self.directory / "2.0.0" / "mimic-cxr-2.0.0-split.csv.gz"
        if self.split in ("train", "val", "test"):
            split_csv = pd.read_csv(self.split_csv_path)["split"].str.contains(
                self.split
            )
            meta_csv = pd.read_csv(self.meta_csv_path)[split_csv].set_index(
                ["subject_id", "study_id"]
            )
            label_csv = pd.read_csv(self.label_csv_path).set_index(
                ["subject_id", "study_id"]
            )

            self.csv = meta_csv.join(label_csv).reset_index()
        elif self.split == "all":
            meta_csv = pd.read_csv(self.meta_csv_path).set_index(
                ["subject_id", "study_id"]
            )
            label_csv = pd.read_csv(self.label_csv_path).set_index(
                ["subject_id", "study_id"]
            )

            self.csv = meta_csv.join(label_csv).reset_index()
        else:
            logging.warning(
                "split {} not recognized for dataset {}, "
                "not returning samples".format(split, self.__class__.__name__)
            )

        self.csv = self.preproc_csv(self.csv, self.subselect)

    @staticmethod
    def default_labels() -> List[str]:
        return [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

    def preproc_csv(self, csv, subselect: Optional[str]) -> pd.DataFrame:
        if csv is not None:

            def format_view(s):
                if s in ("AP", "PA", "AP|PA"):
                    return "frontal"
                elif s in ("LATERAL", "LL"):
                    return "lateral"
                else:
                    return None

            csv["view"] = csv.ViewPosition.apply(format_view)

            if subselect is not None:
                csv = csv.query(subselect)

        return csv

    def __len__(self):
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        subject_id = str(exam["subject_id"])
        study_id = str(exam["study_id"])
        dicom_id = str(exam["dicom_id"])

        filename = self.directory / "2.0.0" / "files"
        filename = (
                filename
                / "p{}".format(subject_id[:2])
                / "p{}".format(subject_id)
                / "s{}".format(study_id)
                / "{}.jpg".format(dicom_id)
        )
        image = self.open_image(filename)

        metadata = self.retrieve_metadata(idx, filename, exam)

        # retrieve labels while handling missing ones for combined data loader
        labels = np.array(exam.reindex(self.label_list)[self.label_list]).astype(
            np.float
        )

        sample = {"image": image, "labels": labels, "metadata": metadata}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    pass