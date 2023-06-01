from feature_extract import get_feature_vector
import polars as pl 
import subprocess
import time

# #demo
# objpath = "models/hhi_5.ply"
# start = time.time()
# features = get_feature_vector(objpath)
# end = time.time()
# time_cost = end-start
#
# print(len(features))
# names = """l_mean,l_std,l_entropy,a_mean,a_std,a_entropy,b_mean,b_std,b_entropy,curvature_mean,curvature_std,curvature_entropy,curvature_ggd1,curvature_ggd2,curvature_aggd1,curvature_aggd2,curvature_aggd3,curvature_aggd4,curvature_gamma1,curvature_gamma2,anisotropy_mean,anisotropy_std,anisotropy_entropy,anisotropy_ggd1,anisotropy_ggd2,anisotropy_aggd1,anisotropy_aggd2,anisotropy_aggd3,anisotropy_aggd4,anisotropy_gamma1,anisotropy_gamma2,linearity_mean,linearity_std,linearity_entropy,linearity_ggd1,linearity_ggd2,linearity_aggd1,linearity_aggd2,linearity_aggd3,linearity_aggd4,linearity_gamma1,linearity_gamma2,planarity_mean,planarity_std,planarity_entropy,planarity_ggd1,planarity_ggd2,planarity_aggd1,planarity_aggd2,planarity_aggd3,planarity_aggd4,planarity_gamma1,planarity_gamma2,sphericity_mean,sphericity_std,sphericity_entropy,sphericity_ggd1,sphericity_ggd2,sphericity_aggd1,sphericity_aggd2,sphericity_aggd3,sphericity_aggd4,sphericity_gamma1,sphericity_gamma2""".split(',')
# df = pl.DataFrame({'metric': names, 'value': features}) 
# #show the features
# with pl.Config(tbl_rows=-1):
#     print(df)
# print("Cost " + str(time_cost) + " sec.")
