import numpy as np
import nibabel as nib
import math
import os
from skimage.util import view_as_windows


class Train_dataset(object):
    def __init__(self, batch_size, subject_list, overlapping=1):
        self.batch_size = batch_size
        self.data_path = '/fs/scratch/PFS0238/gaurangpatel/adversarialML/srgan_input_data'
        self.subject_list = subject_list
        self.width_patch = 86  # 102
        self.heigth_patch = 110  # 126
        self.depth_patch = 78  # 94
        self.margin = 16
        self.overlapping = overlapping
        self.num_patches = (math.ceil((220 / (self.heigth_patch)) / (self.overlapping))) * (
            math.ceil((172 / (self.width_patch)) / (self.overlapping))) * (
                               math.ceil((156 / (self.depth_patch)) / (self.overlapping)))

    def mask(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects_true = np.empty([self.batch_size, 204, 252, 188])
        i = 0
        for subject in subject_batch:
            if subject != 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv':
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'T1_brain_extractedBrainExtractionMask.nii.gz')
                proxy = nib.load(filename)
                data = np.array(proxy.dataobj)

                paddwidthr = int((204 - proxy.shape[0]) / 2)
                paddheightr = int((252 - proxy.shape[1]) / 2)
                paddepthr = int((188 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 204:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 252:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 188:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)
                subjects_true[i] = data_padded
                i = i + 1
        mask = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        mask[i] = p
                        i = i + 1
        return mask

    def patches_true(self, iteration):
        subjects_true, subject_batch = self.data_true(iteration)
        patches_true = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        patches_true[i] = p
                        i = i + 1
        return patches_true

    def data_true(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects = np.empty([self.batch_size, 172, 220, 156])
        i = 0
        for subject in subject_batch:
            if subject != 'ADNI_SCREENING_CLINICAL_FILE_08_02_17.csv':
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
                proxy = nib.load(filename)
                data = np.array(proxy.dataobj)

                paddwidthr = int((204 - proxy.shape[0]) / 2)
                paddheightr = int((252 - proxy.shape[1]) / 2)
                paddepthr = int((188 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 204:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 252:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 188:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)

                subjects[i] = data_padded[16:188, 16:236, 16:172]  # remove background
                i = i + 1
        return subjects, subject_batch
