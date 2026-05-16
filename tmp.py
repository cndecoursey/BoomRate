# Temporary file just to make the SEDs_ia.pkl, then delete once it's there
import os, pickle
from numpy import loadtxt

type       = ['ia']
models_used = ['Hsiao07']
sed_path   = '/Users/christadecoursey/Documents/SNANA/SNANA_2025/snsed/'

model_pkl = 'SEDs_' + '_'.join(type) + '.pkl'

if not os.path.isfile(model_pkl):
    models_used_dict = {}
    total_age_set    = []

    for model in models_used:
        if 'ia' not in type:
            data = loadtxt(os.path.join(sed_path, model + '.SED'))
        else:
            data = loadtxt(os.path.join(sed_path, model + '.dat'))

        # Extract unique ages and store full SED array keyed by model name
        ages = list(set(data[:, 0]))
        models_used_dict[model] = data

        for age in ages:
            if age not in total_age_set:
                total_age_set.append(age)

    # Only open the file once all data is successfully loaded —
    # this prevents a 0-byte pkl being left on disk if loadtxt fails
    with open(model_pkl, 'wb') as pkl_file:
        pickle.dump(models_used_dict, pkl_file)

    print('Written: %s' % model_pkl)

else:
    print('Already exists: %s — delete it first to regenerate.' % model_pkl)