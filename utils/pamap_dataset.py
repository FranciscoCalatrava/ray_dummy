import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

class PAMAP():
    def __init__(self, train, validation, test):
        self.train_participant = train
        self.validation_participant = validation
        self.test_participant = test

        self.training = None
        self.test = None
        self.validation = None

        self.training_cleaned = None
        self.test_cleaned = None
        self.validation_cleaned = None

        self.training_normalized = None
        self.test_normalized = None
        self.validation_normalized = None

        self.training_normalized_segmented = None
        self.test_normalized_segmented = None
        self.validation_normalized_segmented = None

        self.training_final = None
        self.validation_final = None
        self.test_final = None


        self.headers = [
        "timestamp",
        "activityID",
        "heart_rate",
        ] + [
            f"hand_temp",
            *(f"hand_acc16g_{i}" for i in range(1, 4)),
            *(f"hand_acc6g_{i}" for i in range(1, 4)),
            *(f"hand_gyro_{i}" for i in range(1, 4)),
            *(f"hand_mag_{i}" for i in range(1, 4)),
            *(f"hand_orient_{i}" for i in range(1, 5)),
        ] + [
            f"chest_temp",
            *(f"chest_acc16g_{i}" for i in range(1, 4)),
            *(f"chest_acc6g_{i}" for i in range(1, 4)),
            *(f"chest_gyro_{i}" for i in range(1, 4)),
            *(f"chest_mag_{i}" for i in range(1, 4)),
            *(f"chest_orient_{i}" for i in range(1, 5)),
        ] + [
            f"ankle_temp",
            *(f"ankle_acc16g_{i}" for i in range(1, 4)),
            *(f"ankle_acc6g_{i}" for i in range(1, 4)),
            *(f"ankle_gyro_{i}" for i in range(1, 4)),
            *(f"ankle_mag_{i}" for i in range(1, 4)),
            *(f"ankle_orient_{i}" for i in range(1, 5)),
        ]

    def get_datasets(self):
        training = {a:0 for a in self.train_participant}
        test = {a:0 for a in self.test_participant}
        validation ={a:0 for a in self.validation_participant}

        print(training)
        
        for b in training.keys():
            data = pd.read_csv(f"/home/calatrava/Documents/PhD/Papers/Cross_Subject_Transfer_Learning/experiments/other_works/pamap_supervised/dataset/PAMAP2/PAMAP2_Dataset/Protocol/subject10{b}.dat", sep= ' ')
            data.columns = self.headers
            training[b] = data
        for b in validation.keys():
            data = pd.read_csv(f"/home/calatrava/Documents/PhD/Papers/Cross_Subject_Transfer_Learning/experiments/other_works/pamap_supervised/dataset/PAMAP2/PAMAP2_Dataset/Protocol/subject10{b}.dat", sep= ' ')
            data.columns = self.headers
            validation[b] = data
        for b in test.keys():
            data = pd.read_csv(f"/home/calatrava/Documents/PhD/Papers/Cross_Subject_Transfer_Learning/experiments/other_works/pamap_supervised/dataset/PAMAP2/PAMAP2_Dataset/Protocol/subject10{b}.dat", sep= ' ')
            data.columns = self.headers
            test[b] = data
        
        self.training = training
        self.test = test
        self.validation = validation

    def normalize(self):
        training_normalized = {a:0 for a in self.train_participant}
        test_normalized = {a:0 for a in self.test_participant}
        validation_normalized ={a:0 for a in self.validation_participant}

        max = pd.DataFrame(np.zeros((1,len(self.headers))), columns= self.headers)
        min = pd.DataFrame(np.zeros((1,len(self.headers))), columns= self.headers)

        min_aux, max_aux = None, None

        for a in training_normalized.keys():
            max_aux = self.training_cleaned[a].max(axis = 'rows')
            min_aux = self.training_cleaned[a].min(axis = 'rows')
            max.where(max > max_aux, max_aux,axis=1, inplace=True)
            min.where(min < min_aux, min_aux,axis=1, inplace= True)
        
        for a in training_normalized.keys():
            training_normalized[a] = pd.DataFrame(((self.training_cleaned[a].values - min.values)/(max.values- min.values)), columns= self.headers)
            training_normalized[a]["activityID"] = self.training_cleaned[a]["activityID"]        
        for a in test_normalized.keys():
            test_normalized[a] = pd.DataFrame((self.test_cleaned[a].values - min.values)/(max.values- min.values), columns= self.headers)
            test_normalized[a]["activityID"] = self.test_cleaned[a]["activityID"]
        for a in validation_normalized.keys():
            validation_normalized[a] = pd.DataFrame(((self.validation_cleaned[a].values - min.values)/(max.values- min.values)), columns= self.headers)
            validation_normalized[a]["activityID"] = self.validation_cleaned[a]["activityID"]

        self.training_normalized = training_normalized
        self.test_normalized = test_normalized
        self.validation_normalized = validation_normalized

    def segment_data(self, data_dict, window_size, overlap):
        """
        Segments the data into fixed-width windows with overlapping.

        :param data_dict: Dictionary with participant ID as keys and DataFrames as values.
        :param window_size: The size of each window (number of rows).
        :param overlap: The overlap between consecutive windows (number of rows).
        :return: A dictionary with the same keys as data_dict and values as lists of segmented DataFrames.
        """
        segmented_data = {}

        for participant_id, df in data_dict.items():
            num_rows = len(df)
            segments = []
            start = 0
            while start < num_rows:
                end = start + window_size
                if end > num_rows:
                    break
                segment = df.iloc[start:end]
                # Check if the segment contains more than one unique label, if so, skip this segment
                if len(segment.iloc[:, 1].unique()) > 1:
                    start += overlap
                    continue
                segments.append(segment)
                start += overlap
            segmented_data[participant_id] = segments
        return segmented_data
    
    def clean_nan(self, data):
        data_clean = {a:0 for a in data.keys()}
        for a in data.keys():
            data_aux =  data[a].ffill(axis = 0).bfill(axis = 0)
            data_clean[a] = data_aux
        return data_clean

    def butter_lowpass(self,cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    
    def preprocessing(self):
        self.training_cleaned = self.clean_nan(self.training)
        self.test_cleaned = self.clean_nan(self.test)
        self.validation_cleaned = self.clean_nan(self.validation)

        # exclude_columns = ['activityID']

        # for a in self.training_cleaned.keys():
        #     for col in self.training_cleaned[a].columns:
        #         if col not in exclude_columns:
        #             self.training_cleaned[a][col] = self.butter_lowpass_filter(self.training_cleaned[a][col], 20, 100, 3)

    
    def data_segmentation(self):
        train_data_segmented = self.segment_data(self.training_normalized, 512, 256)
        validation_data_segmented = self.segment_data(self.validation_normalized, 512, 256)
        test_data_segmented = self.segment_data(self.test_normalized, 512, 256)

        self.training_normalized_segmented = train_data_segmented
        self.test_normalized_segmented = test_data_segmented
        self.validation_normalized_segmented = validation_data_segmented

    def prepare_dataset(self):

        training, validation, testing = [], [], []
        signals = np.array([4,5,6,10,11,12,21,22,23,27,28,29,38,39,40,44,45,46])
        new_labels = {1:0,  2:1,  3:2,  4:3,  5:4,  6:5,  7:6, 12:7, 13:8, 16:9, 17:10, 24:11}


        for a in self.training_normalized_segmented.keys():
            for b in self.training_normalized_segmented[a]:
                if int(b.iloc[0,1]) !=0:
                    training.append((np.transpose(b.iloc[:,signals].to_numpy())[:,np.newaxis,:], new_labels[int(b.iloc[0,1])], int(a)))
        for a in self.validation_normalized_segmented.keys():
            for b in self.validation_normalized_segmented[a]:
                if int(b.iloc[0,1]) !=0:
                    validation.append((np.transpose(b.iloc[:,signals].to_numpy())[:,np.newaxis,:], new_labels[int(b.iloc[0,1])], int(a)))
        for a in self.test_normalized_segmented.keys():
            for b in self.test_normalized_segmented[a]:
                if int(b.iloc[0,1]) !=0:
                    testing.append((np.transpose(b.iloc[:,signals].to_numpy())[:,np.newaxis,:], new_labels[int(b.iloc[0,1])], int(a)))
        
        self.training_final = training
        self.validation_final = validation
        self.testing_final = testing