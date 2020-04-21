import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import sys

data = pd.read_csv("2008_Mossong_POLYMOD_contact_common.csv")
data = pd.merge(
        data,
        pd.read_csv("2008_Mossong_POLYMOD_participant_common.csv"), on='part_id',
        )
data = pd.merge(data,
    pd.read_csv("2008_Mossong_POLYMOD_hh_common.csv"),
    on='hh_id')

data['cnt_age'] = data.cnt_age_exact.fillna((data.cnt_age_est_min + data.cnt_age_est_max)/2)
data = data.dropna(subset=["cnt_age", "part_age"])

agegroups = np.array(list(range(0, 70 + 1, 5)))
ag_labels = [f"{s}-{e-1}" for s, e in zip(agegroups, agegroups[1:])] + [f"{agegroups[-1]}+"]
data['part_age_group'] = np.digitize(data.part_age.values, agegroups, right=False) - 1

data['cnt_age_group'] = np.digitize(data.cnt_age.values, agegroups, right=False) - 1

place_types = ['cnt_home', 'cnt_work', 'cnt_school', 'cnt_transport', 'cnt_leisure', 'cnt_otherplace']

def get_contact_matrix(data, ag_participants=None):
    M = np.zeros((len(agegroups), len(agegroups)))
    if ag_participants is None:
        ag_participants = data.groupby('part_age_group')['part_id'].nunique()
    for part_ag, d in data.groupby('part_age_group'):
        n = ag_participants[part_ag]
        for cnt_ag, cd in d.groupby('cnt_age_group'):
            M[part_ag, cnt_ag] += cd['contact_weight'].sum()/n
    os = np.sum(M)
    M = scipy.ndimage.gaussian_filter(M, 0.25)
    M *= os/np.sum(M)
    # Assume the result should be symmetric. TODO: Verify
    #M += M.T; M /= 2 
    return M

data = data[data[place_types].values.sum(axis=1) > 0]
data['contact_weight'] = 1.0

M_total = get_contact_matrix(data)
#os = np.sum(M_total)
#M_total = scipy.ndimage.gaussian_filter(M_total, 0.25)
#M_total *= os/np.sum(M_total)
M_cumtotal = np.zeros_like(M_total)


Ms = []
for country, cd in data.groupby('country'):
    ag_participants = cd.groupby('part_age_group')['part_id'].nunique()
    for place_type in place_types:
        plt.title(place_type)
        d = cd.copy()
        d['contact_weight'] = d[place_type]/d[place_types].values.sum(axis=1)
        
        OM = get_contact_matrix(d, ag_participants)
        # TODO: This isn't very well founded, but at least it gets rid of zeros
        #os = np.sum(OM)
        #OM = scipy.ndimage.gaussian_filter(OM, 0.25)

        #OM /= np.sum(OM)
        #OM *= os

        M_cumtotal += OM
        
        M = pd.DataFrame(OM, columns=ag_labels)
        M.insert(0, "participant_age", ag_labels)
        M.insert(0, "place_type", place_type)
        M.insert(0, "country", country)
        Ms.append(M)
        #plt.title(place_type)
        #plt.imshow(OM, origin='lower', extent=(0, 80, 0, 80))
        #plt.show()

out = pd.concat(Ms)
out.to_csv(sys.stdout, index=False)
