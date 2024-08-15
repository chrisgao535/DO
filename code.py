# read dataset (The pandas loop was employed to read the text data set, and get the 'B02', 'B03', 'B04', 'B08' band value and dissolved oxygen values)
txt_file_list = sorted(glob(r'.\data\*.txt'))
df = pd.read_csv(txt_file_list[0])
for i in range(1, len(txt_file_list)):
    df = pd.concat([df, pd.read_csv(txt_file_list[i], encoding='gbk')], axis=0)
X_data_arr, y_data_arr = df[['B02', 'B03', 'B04', 'B08']].values, df['DO(mg/L)'].values

# create models(Create 9 machine learning models and store them as dictionaries)
regr_rf_dict = {'BRR': linear_model.BayesianRidge(),
                'KNR': neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance'),
                'DTR': tree.DecisionTreeRegressor(), 'SVM': svm.SVR(),
                'ABR': AdaBoostRegressor(random_state=0, n_estimators=100),
                'RFR': RandomForestRegressor(n_estimators=145, max_features=1, criterion="squared_error",
                                             max_depth=60, min_samples_split=2, min_samples_leaf=1,
                                             min_weight_fraction_leaf=0.0, max_leaf_nodes=277,
                                             min_impurity_decrease=0.0, bootstrap=False, n_jobs=-1, verbose=1,
                                             warm_start=False, ccp_alpha=0.01, ),
                'ETR': ExtraTreesRegressor(n_estimators=353, criterion="friedman_mse", max_depth=37,
                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                           max_features=2, max_leaf_nodes=200, min_impurity_decrease=0.0,
                                           bootstrap=False, n_jobs=-1, verbose=1, warm_start=False,
                                           ccp_alpha=0.0, ), 'GBR': GradientBoostingRegressor(random_state=0),
                'ANN': MLPRegressor(random_state=1, max_iter=10)}

# Model training and save the result (The above 9 machine learning models are cyclically trained, and calculate the R2, MAE, RMSE, EVS, STD, T-value, P-value and confidence interval)
r_list, mae_list, mse_list, rmse_list, evs_list, std_list, t_list, p_list, lower_list, upper_list = [[]] * 10
result_dir = r'.'
for mol_name, regr_rf in regr_rf_dict.items():
    regr_rf.fit(X_data_arr, y_data_arr)
    with open(r'%s\%s.joblib' % (result_dir, mol_name), 'wb') as f:
        joblib.dump(regr_rf, f, compress=6)
    y_rf = regr_rf.predict(X_data_arr)
    pd.DataFrame({'real_value': y_data_arr.reshape(-1), 'predict_value': y_rf.reshape(-1)}).to_csv(
        r'%s\%s.csv' % (result_dir, mol_name), index=False)
    r, mae, mse, rmse, evs, std_arr, t_stat, p_value, lower_bound, upper_bound = Correlation_And_Draw(
        y_rf.reshape(-1), y_data_arr.reshape(-1), 'Predicted value(mg/L)', 'Real value(mg/L)', 0,
        y_data_arr.max() + 2, 0, y_data_arr.max() + 2, color_list=[], title='',
        pic_name=r'%s\%s_%s.png' % (result_dir, mol_name, os.path.basename(txt_file_list[0])[:-4]))
    r_list.append(r * r)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    evs_list.append(evs)
    std_list.append(std_arr)
    t_list.append(t_stat)
    p_list.append(p_value)
    lower_list.append(lower_bound)
    upper_list.append(upper_bound)
pd.DataFrame({'Algorithm': regr_rf_dict.keys(), 'R2': r_list, 'Mean absolute error': mae_list,
              'Mean squared error': mse_list, 'Root mean squared error': rmse_list,
              'Explained variance score': evs_list, 't': t_list, 'p': p_list, 'lower': lower_list,
              'upper': upper_list}, ).to_csv(r'%s\sta.csv' % (result_dir), index=False)

# Inversion of dissolved oxygen concentration (The trained machine learning model was used to retrieve the dissolved oxygen in spring, summer, and autumn)
tif_file = r".\PRE_SENTINEL2_L2A_*_layer_stacking_clip.tif"
temp_dir = os.path.dirname(tif_file) + '/temp'
if not os.path.exists(temp_dir): os.makedirs(temp_dir)
res_arr, gt, proj, datatype, no_data = read_tif_multi(tif_file, aux_data=False)
mask = res_arr[0] == no_data
res_arr_shape = res_arr.shape
models = sorted(glob(r".\*.joblib"))
for mod in models:
    regr_rf = joblib.load(filename=mod)
    pre_image = regr_rf.predict(res_arr[(1, 2, 3, 7), ...].reshape(4, -1).T).reshape(res_arr_shape[1],res_arr_shape[2])
    pre_image[mask] = -9999
    res_tif = r'%s\%s' % (temp_dir, os.path.basename(tif_file))
    write_tif(res_tif, pre_image, gt, proj, no_val=-9999, return_mode='TIFF')
    clip(res_tif, r'.\*_%s.tif' % (os.path.basename(mod)[:-7]), r".\water_buf.shp", dstNodata=-9999)
