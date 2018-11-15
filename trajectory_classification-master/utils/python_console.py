import numpy as np
import pandas as pd
import pickle
import time
from sklearn.metrics import roc_curve, auc
from sklearn.mixture import GMM
import warnings
import time
from sklearn.metrics import roc_curve, auc
from sklearn.mixture import GMM
import warnings
np.set_printoptions(threshold=np.NaN)
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
DATA_DIR = "D:\\MeetingMinutes\\Data\\test_basemap\\trajectory_classification-master\\data\\"

fsd_saobento = pd.read_pickle(DATA_DIR + "Paired_45.pkl")
fsd_saobento = pd.read_pickle(DATA_DIR + "fsd_saobento.pkl")
Paired_45 = pd.read_pickle(DATA_DIR + "Paired_45.pkl")
sao_bento_traj_labels = pd.read_pickle(DATA_DIR + "sao_bento_traj_labels.pkl")
fsd_saobento_cumsum_head1 = pd.read_csv(DATA_DIR + "fsd_saobento_cumsum.csv")
id_traj_suspect = fsd_saobento_cumsum_head1.id_traj[fsd_saobento_cumsum_head1.sfzh.isin(fsd_saobento_cumsum_head1.sfzh.head(10))]
id_traj_suspect_szfh = fsd_saobento_cumsum_head1[["id_traj","sfzh"]][fsd_saobento_cumsum_head1.sfzh.isin(fsd_saobento_cumsum_head1.sfzh.head(10))]
id_traj_suspect_szfh.szfh.drop_duplicates()
id_traj_suspect_szfh.sfzh.drop_duplicates()
id_traj_suspect_szfh.sfzh.drop_duplicates().index
np.hstack(id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index)
np.hstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))
np.vstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))
np.vstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[1]
np.concatenate((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index),axis=1)[1]
np.concatenate((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index),axis=0)[1]
np.concatenate((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index),axis=0)
dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[1]
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[0,0]
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[0,2]
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[0,1]
np.shape(np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index)))
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[1]
np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[0]
id_traj_suspect_szfh_ditinct=np.dstack((id_traj_suspect_szfh.sfzh.drop_duplicates(),id_traj_suspect_szfh.sfzh.drop_duplicates().index))[0]
fsd_saobento.reindex(id_traj_suspect_szfh.sfzh.index)
id_traj_suspect_szfh.reindex(id_traj_suspect_szfh_ditinct.index)
id_traj_suspect_szfh.reindex(id_traj_suspect_szfh.sfzh.index)
DataFrame(id_traj_suspect_szfh_ditinct,index=id_traj_suspect_szfh_ditinct.index,column=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=id_traj_suspect_szfh_ditinct.index,column=['sfzh','id'])
numpy.where(array==id_traj_suspect_szfh_ditinct)
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=[range(0,9)],columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=[0,1,2,3,4,5,6,7,8,9],columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=[np.arange(0,9)],columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=[[np.arange(0,9)]],columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=[list(np.arange(0,9))],columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=list(np.arange(0,9)),columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=np.arange(0,9),columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=np.arange(1,10),columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=np.arange(0,8),columns=['sfzh','id'])
pd.DataFrame(id_traj_suspect_szfh_ditinct,index=np.arange(0,10),columns=['sfzh','id'])
id_traj_suspect_szfh_ditinct=pd.DataFrame(id_traj_suspect_szfh_ditinct,index=np.arange(0,10),columns=['sfzh','id'])
id_traj_suspect_szfh.join(id_traj_suspect_szfh_ditinct,on='sfzh',how='left')
id_traj_suspect_szfh.join(id_traj_suspect_szfh_ditinct.id,on='sfzh',how='left')
id_traj_suspect_szfh.join(id_traj_suspect_szfh_ditinct.id,on=['sfzh'],how='left')
id_traj_suspect_szfh.concat(id_traj_suspect_szfh_ditinct.id,on=['sfzh'],how='left')
pd.concat([id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id],on=['sfzh'],how='left')
pd.concat([id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id],how='left')
pd.concat([id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id],axis=1,join='letf')
pd.concat([id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id],axis=1,join='outer')
pd.merge(id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id,on=['sfzh'],join='outer')
pd.merge(id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct.id,on=['sfzh'],how='outer')
pd.merge(id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct,on=['sfzh'],how='outer')
id_traj_suspect_szfh_id = pd.merge(id_traj_suspect_szfh,id_traj_suspect_szfh_ditinct,on=['sfzh'],how='outer')
fsd_saobento[fsd_saobento.id_traj.isin(id_traj_suspect_szfh_id.id_traj)]
fsd_saobento_suspect = fsd_saobento[fsd_saobento.id_traj.isin(id_traj_suspect_szfh_id.id_traj)]
fsd_saobento_suspect.index=np.arange(0,331)
pd.DataFrame(np.arange(0,78),index=np.arange(0,78),columns=["id"])
pd.concat((fsd_saobento_suspect.id_traj.drop_duplicates(),pd.DataFrame(np.arange(0,78),index=np.arange(0,78),columns=["id"])))
fsd_saobento_suspect.id_traj.drop_duplicates()
fsd_saobento_suspect_distinct = fsd_saobento_suspect.id_traj.drop_duplicates()
fsd_saobento_suspect_distinct = fsd_saobento_suspect["id_traj"].drop_duplicates()
fsd_saobento_suspect_distinct = fsd_saobento_suspect["id_traj"]
fsd_saobento_suspect_distinct = fsd_saobento_suspect[["id_traj"]]
fsd_saobento_suspect_distinct = fsd_saobento_suspect[["id_traj"]].drop_duplicates()
fsd_saobento_suspect_distinct.columns=['id_traj_old']
fsd_saobento_suspect_distinct["id_traj"]=np.arange(0,78)
fsd_saobento_suspect_distinct.columns=['id_traj','id']
pd.merge(fsd_saobento_suspect, fsd_saobento_suspect_distinct,how='outer',on='id_traj')
pd.merge(fsd_saobento_suspect, fsd_saobento_suspect_distinct,how='outer',on='id_traj')[["lats","lons","occupancy","timestamp","id"]]
fsd_saobento_suspect_new = pd.merge(fsd_saobento_suspect, fsd_saobento_suspect_distinct,how='outer',on='id_traj')[["lats","lons","occupancy","timestamp","id"]]
fsd_saobento_suspect_new.clumns = ["lats","lons","occupancy","timestamp","id_traj"]
fsd_saobento_suspect_new.columns = ["lats","lons","occupancy","timestamp","id_traj"]
df_Paired_45 = pd.DataFrame(list(Paired_45.items()),columns=["id","color"])
df_Paired_45["flag"] = mod(df_Paired_45["id"]/5)
df_Paired_45["flag"] = divmod(df_Paired_45["id"],5)[1]
df_Paired_10=df_Paired_45.where(df_Paired_45.flag == 0).dropna(axis = 0)
df_Paired_10.id = np.arange(0,10)
df_Paired_10 = df_Paired_10.drop(["flag"],axis=1)
Paired_10 = {0:(0.6509804129600525, 0.8078431487083435, 0.8901960849761963, 1.0), 1:(0.16678200852052832, 0.5022683605259539, 0.6929642628220951, 1.0), 2:(0.5984313875436783, 0.8250980496406556, 0.46745100319385535, 1.0), 3:(0.4183775495080386, 0.6208996765753803, 0.2915647927452536, 1.0), 4:(0.946666669845581, 0.40313726961612706, 0.40392158329486855, 1.0), 5:(0.9389773200539981, 0.41153403324823756, 0.26552864514729557, 1.0), 6:(0.9968627452850342, 0.5984313786029816, 0.17411764860153212, 1.0), 7:(0.858992703521953, 0.6337255024442485, 0.5693502564056242, 1.0), 8:(0.49098039865493803, 0.33098039627075226, 0.6509804129600527, 1.0), 9:(0.8949942392461441, 0.13241061331594697, 0.12512110831106413, 1.0)}
POI_DIR = DATA_DIR + "POI.csv"
POI_DF = pd.read_csv(POI_DIR, index_col=0)
LON_SB, LAT_SB, TOL_LONS_SB, TOL_LATS_SB = POI_DF.loc["SaoBento"].values
LON_C, LAT_C, TOL_LONS_C, TOL_LATS_C = POI_DF.loc["Caltrain"].values
fsd_sao_bento_suspect_traj_labels = id_traj_suspect_szfh_id.id.values
int(fsd_sao_bento_suspect_traj_labels)
fsd_sao_bento_suspect_traj_labels.astype(int)
fsd_sao_bento_suspect_traj_labels = fsd_sao_bento_suspect_traj_labels.astype(int)