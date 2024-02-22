import nibabel as nib
from nilearn import plotting
from nilearn import image
file ="H:\\raw\\pMCI_caps\\subjects\\sub-ADNI003S1057\\ses-M000\\t1_linear\\sub-ADNI003S1057_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
plotting.plot_anat(file, title='data')
