# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features
"""
import pandas
import scipy.io
import numpy as np
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy.stats
from concurrent import futures
import functools
import warnings
warnings.filterwarnings("ignore")
# ----------------------- Set System logger ------------- #
class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='RAPIQUE',
                      help='Evaluated BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='KONVID_1K',
                      help='Evaluation dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='feat_files/KONVID_1K_RAPIQUE_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--mos_file', type=str,
                      default='mos_files/KONVID_1K_metadata.csv',
                      help='Dataset MOS scores.')
  parser.add_argument('--num_cont', type=int,
                      default=10,
                      help='Number of contents.')
  parser.add_argument('--num_dists', type=int,
                      default=15,
                      help='Number of distortions per content.')
  parser.add_argument('--out_file', type=str,
                      default='result/KONVID_1K_RAPIQUE_SVR_corr.mat',
                      help='Output correlation results')
  parser.add_argument('--log_file', type=str,
                      default='logs/KONVID_1K_RAPIQUE_SVR.log',
                      help='Log files.')
  parser.add_argument('--color_only', action='store_true',
                      help='Evaluate color values only. (Only for YouTube UGC)')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--num_iterations', type=int, default=50,
                      help='Number of iterations of train-test splits')
  parser.add_argument('--max_thread_count', type=int, default=4,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4, bayta5):
  # 5-parameter logistic function
  logisticPart = 0.5 - np.divide(1,(1+np.exp(bayta2 * (X - bayta3))))
  yhat = np.multiply(bayta1, logisticPart) + np.multiply(bayta4, X) + bayta5
  return yhat


def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5, 0.1]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, duration):
  print('======================================================')
  print('SRCC_test: ', snapshot[0])
  print('KRCC_test: ', snapshot[1])
  print('PLCC_test: ', snapshot[2])
  print('RMSE_test: ', snapshot[3])
  print('======================================================')
  print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def final_avg(snapshot):
  def formatted(args, pos):
    mean = np.nanmean(list(map(lambda x: x[pos], snapshot)))
    stdev = np.nanstd(list(map(lambda x: x[pos], snapshot)))
    print('{}: {} (std: {})'.format(args, mean, stdev))

  print('======================================================')
  print('Average test results among all repeated 80-20 holdouts:')
  formatted("SRCC Test", 0)
  formatted("KRCC Test", 1)
  formatted("PLCC Test", 2)
  formatted("RMSE Test", 3)
  print('\n\n')

def idx_expand(idx, num_dists):
  idx_out = []
  for ii in idx:
    idx_out.extend(range(ii*num_dists,(ii+1)*num_dists))
  return idx_out

def evaluate_bvqa_one_split(i, X, y, num_cont, num_dists, log_short):
  if log_short:
    print('{} th repeated holdout test'.format(i))
    t_start = time.time()
  # train test split
  idx_train, idx_test = train_test_split(range(585), test_size=0.2, random_state=math.ceil(8.8*i)) # num_cont

  X_train = X[idx_train]
  X_test = X[idx_test]
  y_train = y[idx_train]
  y_test = y[idx_test]

  metrics_test = compute_metrics(X_test, y_test)
  # print values
  if log_short:
    t_end = time.time()
    formatted_print(metrics_test, (t_end - t_start))
  return  metrics_test
  
def main(args):
  df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
  array = df.values
  if args.dataset_name == 'LIVE_VQC':
      y = array[1:,1]
  elif args.dataset_name == 'KoNVid': # for LIVE-VQC & KONVID_1k
      y = array[1:,4]
  elif args.dataset_name == 'Youtube-UGC':
      y = array[1:,4]
  elif args.dataset_name == 'LIVE_VQA':
      y = array[1:,1]
  elif args.dataset_name == 'LIVE_HFR':
      y = array[1:,1]
  elif args.dataset_name == 'BVI_HFR':
      y = array[1:,1]
  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['new'], dtype=np.float)
  X = X.flatten('F')

  '''57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison'''
  if args.color_only and args.dataset_name == 'YOUTUBE_UGC':
      gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
      639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
      1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
      gray_indices = [idx - 1 for idx in gray_indices]
      X = np.delete(X, gray_indices, axis=0)
      y = np.delete(y, gray_indices, axis=0)

  all_iterations = []
  t_overall_start = time.time()
  # 100 times random train-test splits
  if args.use_parallel is True:
    evaluate_bvqa_one_split_partial = functools.partial(
       evaluate_bvqa_one_split, X=X, y=y, num_cont=args.num_cont,
       num_dists=args.num_dists, log_short=args.log_short)
    with futures.ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
      iters_future = [
          executor.submit(evaluate_bvqa_one_split_partial, i)
          for i in range(1, args.num_iterations)]
      for future in futures.as_completed(iters_future):
        metrics_test = future.result()
        all_iterations.append(metrics_test)
  else:
    for i in range(1, args.num_iterations):
      metrics_test = evaluate_bvqa_one_split(
          i, X, y, args.num_cont, args.num_dists, args.log_short)
      all_iterations.append(metrics_test)

  # formatted print overall iterations
  final_avg(all_iterations)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  # save figures
  dir_path = os.path.dirname(args.out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.out_file, 
      mdict={'all_iterations': np.asarray(all_iterations,dtype=np.float)})

if __name__ == '__main__':
  args = arg_parser()
  log_file = args.log_file
  log_dir = os.path.dirname(log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(log_file)
  print(args)
  main(args)

'''

python evaluate_bvqa_features_regression.py \
  --model_name BRISQUE \
  --dataset_name LIVE_VQC \
  --feature_file mos_feat_files/KONIQ_10K_BRISQUE_feats.mat \
  --mos_file mos_feat_files/KONIQ_10K_metadata.csv \
  --out_file result/KONIQ_10K_BRISQUE_SVR_corr.mat \
  --use_parallel


'''
