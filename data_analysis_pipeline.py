# encoding=utf-8
import os
import sys
import numpy as np
from numpy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import argparse
import matplotlib.pyplot as plt

class Transform(object):
    @staticmethod
    def standard_scale(data_mat):
        '''
        scale data: mean:0, variance:1
        :param data_mat: input data mat
        :return: scaled data mat
        '''
        scaled_data_mat = preprocessing.scale(data_mat)
        # print (scaled_data_mat.mean(axis=0))
        # print (scaled_data_mat.std(axis=0))
        return np.mat(scaled_data_mat)

    @staticmethod
    def normalize(data_mat):
        '''
        normalize data to [0,1]
        :param data_mat: input data mat
        :return: normalized data mat
        '''
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_data_mat = min_max_scaler.fit_transform(data_mat)
        return scaled_data_mat

    @staticmethod
    def regularize(data_mat, regularization_type = "L1"):
        '''
        use L1 or L2 to regularize data
        :param data_mat: input data mat
        :param regularization_type: L1 or L2
        :return: regularized data mat
        '''
        scaled_data_mat = preprocessing.normalize(data_mat, norm = regularization_type)
        return scaled_data_mat

    @staticmethod
    def feature_mapping(data_mat, data_index = None, mapping_dict = None):
        '''
        transform string data(data_mat[:data_index]) to float
        :param data_mat: input data mat
        :param data_index: transform index
        :param mapping_dict: mapping dict
        :return: processed data mat
        '''
        if data_index is None or mapping_dict is None:
            return data_mat
        #gender_map = {"Male": 0, "Female": 1}
        data_mat[:, data_index] = [[mapping_dict[v[0]]] for v in data_mat[:, data_index].A]
        return data_mat

    @staticmethod
    def float_data(data_mat):
        '''
        transform data to float
        :param data_mat: input data mat
        :return: processed data mat
        '''
        data_mat = data_mat.astype(float)
        return data_mat

class Cleaner(object):
    @staticmethod
    def process_missing_value(data_mat, method_type = "del"):
        '''
        deal with missing value
        :param data_mat: input data mat
        :param method_type: del:delete missing value; mean:replace with mean value; median:replace with median value
        :return: processed data mat
        '''
        if method_type not in ["del", "mean", "median"]:
            return data_mat

        if method_type == "del":
            rows, cols = data_mat.shape
            status = np.isnan(data_mat)
            del_rows = []
            for idx in xrange(rows):
                if True in status[idx, :]:
                    del_rows.append(idx)
            data_mat = np.delete(data_mat, del_rows, axis =0)
            print data_mat
        else:
            imp = Imputer(missing_values="NaN", strategy=method_type, axis=0)
            imp.fit(data_mat)
            data_mat = imp.transform(data_mat)
        data_mat = np.mat(data_mat)
        return data_mat

    @staticmethod
    def filter(data, method_type):
        win_size = 25 # calculate mean in a window
        num = win_size / 2
        ret = []
        for idx, ele in enumerate(data):
            tmp = []
            if idx >= num:
                tmp += data[idx - num:idx]
            else:
                tmp += data[:idx]
            tmp += [ele]
            tmp += data[idx: idx + num]
            tmp = np.array(tmp)
            if method_type == "mean":
                val = np.mean(tmp)
            else:
                val = np.median(tmp)
            ret.append([val])
        return ret

    @staticmethod
    def process_noise(data_mat, method_type = "mean"):
        '''
        process noisy data
        :param data_mat: input data mat
        :param method_type: mean:mean filter; median:median filter
        :return: processed data mat
        '''
        if method_type not in ["mean", "median"]:
            return data_mat

        rows, cols = data_mat.shape
        for col_idx in xrange(cols):
            data = data_mat[:, col_idx].tolist()
            data = [float(v[0]) for v in data]
            data = Cleaner.filter(data, method_type)
            data_mat[:, col_idx] = data

        return data_mat

    @staticmethod
    def del_repeated_data(data_mat):
        '''
        delete repeated data
        :param data_mat: input data mat
        :return: processed data mat
        '''
        data_mat = np.mat(list(set([tuple(t) for t in data_mat.A])))
        return data_mat

class DimensionReducer(object):
    REDUCE_DIMS_PCA = "pca"
    REDUCE_DIMS_SVD = "svd"

    @staticmethod
    def pca_reducer(data_mat, top_n_feat = 9999999):
        '''
        use pca algorithm to reduce dimension
        :param data_mat: input data mat
        :param top_n_feat: selected features num
        :return: top n useful features
        '''
        # subtract mean values
        mean_vals = np.mean(data_mat, axis = 0)
        mean_removed = data_mat - mean_vals

        # calculate covariance matrix
        cov_mat = np.cov(mean_removed, rowvar = 0)

        # calculate eigenvalues and eigenvectors
        eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))

        # sort eigenvalues in ascending order
        eig_vals_index = np.argsort(eig_vals)
        eig_vals_index = eig_vals_index[:-(top_n_feat + 1):-1]
        red_eig_vects = eig_vects[:, eig_vals_index]

        # transform data to a new sapce
        low_dim_data_mat = mean_removed * red_eig_vects
        new_mat = (low_dim_data_mat * red_eig_vects.T) + mean_vals
        # return low_dim_data_mat, new_mat
        return new_mat

    @staticmethod
    def svd_reducer(data_mat, top_n_feat = 99999):
        '''
        use SVD algorithm to reduce dimension
        :param data_mat: input data mat
        :param top_n_feat: selected features num
        :return: top n useful features
        '''
        data_mat = data_mat.T
        rows = np.shape(data_mat)[0]
        cols = np.shape(data_mat)[1]
        u, sigma, vt = linalg.svd(data_mat)
        feat_num = min(top_n_feat, cols)
        sig = np.mat(np.eye(feat_num) * sigma[: feat_num]) #arrange sig into a diagonal matrix
        #final_mat = data_mat.T * u[:,:feat_num] * sig.I  #create transformed items
        final_mat = u[:, :feat_num] *sig * vt[:feat_num, :]
        return final_mat.T

    @staticmethod
    def reducer(data_mat, top_feature_num, method = None):
        '''
        use different methods to reduce dimensionality
        :param method:
        :return:
        '''
        if method == DimensionReducer.REDUCE_DIMS_PCA:
            return DimensionReducer.pca_reducer(data_mat, top_feature_num)
        elif method == DimensionReducer.REDUCE_DIMS_SVD:
            return DimensionReducer.svd_reducer(data_mat, top_feature_num)
        else:
            return data_mat


class DataAnalysisPipeline(object):
    def __init__(self, args):
        self.args = args

    def read_csv_file(self, file_path):
        '''
        read data from csv: the last column is target column
        :param file_path: local file path
        :return:
        '''
        data = []
        index = 0
        for line in open(file_path):
            index += 1
            if index == 1: # title row
                continue
            # index fea_1 fea_2 ... fea_n target
            info = line[:-1].split(",")
            info = info[1:-1]
            data.append(info)
        data = np.mat(data)
        data[data == ''] = np.nan
        return data

    def display(self, data_mat, new_data_mat):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90, c='blue')
        #ax.scatter(new_data_mat[:, 0].flatten().A[0], new_data_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')

        data =  data_mat[:, 1].flatten().A[0]
        new_data = new_data_mat[:, 1].flatten().A[0]
        ax.scatter(range(1, len(data) + 1), data, marker='^', s=90, c='blue')
        ax.scatter(range(1, len(data) + 1), new_data, marker='o', s=50, c='red')
        plt.show()

    def save(self, data_mat, file_path):
        np.savetxt(file_path, data_mat, fmt = "%s", delimiter = ",")

    def clean(self, data_mat):
        # process missing value
        if "process_missing_value" in self.args:
            print ("  processing missing value...")
            data_mat = Cleaner.process_missing_value(data_mat, self.args.process_missing_value)

        # process repeated data
        if "del_repeated" in self.args and self.args.del_repeated == 1:
            print ("  deleting repeated data... ")
            data_mat = Cleaner.del_repeated_data(data_mat)

        # process noisy data
        if "process_noise" in self.args:
            print ("  processing noisy data...")
            data_mat = Cleaner.process_noise(data_mat, self.args.process_noise)

        return data_mat

    def transform(self, data_mat):
        if "normalize"  in self.args:
            method = self.args.normalize.split(",")
            for m in method:
                # standard scale
                if m == "standard_scale":
                    print ("  standard scale...")
                    data_mat = Transform.standard_scale(data_mat)

                # normalize
                if m == "normalize":
                    print ("  normalize...")
                    data_mat = Transform.normalize(data_mat)

                # regularize
                if m == "regularize_L1":
                    print ("  L1 regularize...")
                    data_mat = Transform.regularize(data_mat, "l1")
                elif m == "regularize_L2":
                    print ("  L2 regularize...")
                    data_mat = Transform.regularize(data_mat, "l2")

        return data_mat

    def reduce_dims(self, data_mat):
        if "feature_num" in self.args:
            data_mat = DimensionReducer.reducer(data_mat, self.args.feature_num, self.args.reduce_dims)

        return data_mat

    def run_pipeline(self):
        data_mat = self.read_csv_file(self.args.input_file)

        # data convert mapping
        if "convert_data_type" in self.args:
            gender_map = {"Male": 0, "Female": 1}
            data_index = 0
            print ("  convert data type...")
            data_mat = Transform.feature_mapping(data_mat, data_index, gender_map)
            data_mat = Transform.float_data(data_mat)

        if "clean" in self.args and self.args.clean == 1:
            print ("clean data start...")
            data_mat = self.clean(data_mat)
            print ("clean data end...")

        if "transform" in self.args and self.args.transform == 1:
            print ("transform data start...")
            data_mat = self.transform(data_mat)
            print ("transform data end...")

        if "reduce_dims" in self.args and self.args.reduce_dims in ["pca", "svd"]:
            print ("reduce_dims start...")
            data_mat = self.reduce_dims(data_mat)
            print ("reduce_dims end...")

        if "output_file" in self.args and self.args.output_file != "":
            print ("save result start...")
            self.save(data_mat, self.args.output_file)
            print ("save result end...")

def run_pipeline():
    '''
    run pipeline
    :return:
    '''
    parser = argparse.ArgumentParser("python data_analysis_pipeline.py")
    parser.add_argument("--input_file", type=str, required=True, help="input file path, required")

    # clean
    parser.add_argument("--clean", type=int, required=False, help="clean data or not, optional, default 0", default=0)
    parser.add_argument("--process_missing_value", type=str, required=False, help="process missing value using method:del, mean, median")
    parser.add_argument("--del_repeated", type=int, required=False, help="delete repeate data, 1:yes 0:no")
    parser.add_argument("--process_noise", type=str, required=False, help="process noisy data using method:mean, median")

    # transform
    parser.add_argument("--transform", type=int, required=False, help="transform data, optional, default 1", default=1)
    parser.add_argument("--convert_data_type", type=int, required=False, help="transform string data to float, 0:no 1:yes")
    parser.add_argument("--normalize", type=str, required=False, help="normalize data using method: standard_scale,normalize,regularize_L1,regularize_L2;Multiple methods are separated by commas")

    # reduce dims
    parser.add_argument("--reduce_dims", type=str, required=False, help="reduce dims use pca,svd,none, default none", default="none")
    parser.add_argument("--feature_num", type=int, required=False, help="select num features in reducing dims")

    # output
    parser.add_argument("--output_file", type=str, required=False, help="final processed data will be saved if given output_file")

    args = parser.parse_args()
    #print (args)

    pipeline_obj = DataAnalysisPipeline(args)
    pipeline_obj.run_pipeline()

if __name__ == "__main__":
    # python data_analysis_pipeline.py --input_file GroupTwo.csv --clean 0 --transform 1 --reduce_dims pca --feature_num 45 --output_file result.txt
    run_pipeline()
