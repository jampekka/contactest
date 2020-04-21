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


agegroups = np.array(list(range(0, 70 + 1, 5)))
ag_labels = [f"{s}-{e-1}" for s, e in zip(agegroups, agegroups[1:])] + [f"{agegroups[-1]}+"]
data['part_age_group'] = np.digitize(data.part_age.values, agegroups, right=False) - 1

data['cnt_age'] = data.cnt_age_exact.fillna((data.cnt_age_est_min + data.cnt_age_est_max)/2)
data['cnt_age_group'] = np.digitize(data.cnt_age.values, agegroups, right=False) - 1

place_types = ['cnt_home', 'cnt_work', 'cnt_school', 'cnt_transport', 'cnt_leisure', 'cnt_otherplace']

def get_contact_matrix(data):
    M = np.zeros((len(agegroups), len(agegroups)))
    for part_ag, d in data.groupby('part_age_group'):
        n = d.part_id.nunique()
        for cnt_ag, cd in d.groupby('cnt_age_group'):
            M[part_ag, cnt_ag] += cd['contact_weight'].sum()/n
    return M

Ms = []
for country, cd in data.groupby('country'):
    for place_type in place_types:
        plt.title(place_type)
        d = cd.copy()
        d['contact_weight'] = d[place_type]/d[place_types].values.sum(axis=1)
        
        OM = get_contact_matrix(d)
        # TODO: This isn't very well founded, but at least it gets rid of zeros
        os = np.sum(OM)
        OM = scipy.ndimage.gaussian_filter(OM, 1.0)
        OM *= os/np.sum(OM)
        
        M = pd.DataFrame(OM, columns=ag_labels)
        M.insert(0, "participant_age", ag_labels)
        M.insert(0, "place_type", place_type)
        M.insert(0, "country", country)
        Ms.append(M)
        #plt.imshow(OM, origin='lower', extent=(0, 80, 0, 80))
        #plt.show()

out = pd.concat(Ms)
out.to_csv(sys.stdout, index=False)
