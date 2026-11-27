IMAGE="image"
MASK="mask"
TARGET="target"
SOURCE="source"
PAT_ID="pat_id"
NIMAGE="noisy"
SIGMA = "sigma"

INIT_ZERO = "zero"
INIT_NOISY = "noisy"

NOISE_MULTI = "multiplicative"
NOISE_EDM = "edm"
NOISE_ADD = "additive"
NOISE_COMB = "combined"
NOISE_FLOW = "flow"

MODEL_UNETPLUSPLUS = "unetplusplus"
MODEL_UNET = "unet"
MODEL_ENERGY = "energy"
MODEL_TRANSUPNET = "transupnet"
SONG_UNET = "song"
JIT = "jit"
SONG_3DUNET = "song3d"
SONG_3DCONDUNET = "song3dcond"
TRANS_UNET = "transunet"



STUDY_SUDETES = "SUDETES"
STUDY_APENNINES = "APENNINES"

MMSE_MODEL = "mmse_model"
FLOW_MODEL = "flow_model"

NOISE_TYPE_NORMAL = "normal"
NOISE_TYPE_CONSTANT = "constant"
NOISE_TYPE_SLERP = "slerp"
NOISE_TYPE_LINEAR_VM = "linear_vm"
NOISE_TYPE_LINEAR = "linear"
# Frequency-split couplings: low B-scan frequencies shared (freqsplit) or slerp-interpolated
# (freqsplit_slerp) across slices, high frequencies independent per slice. An optional
# "_fc<cutoff>" suffix sets the cutoff in cycles/pixel, e.g. "freqsplit_fc0.2".
NOISE_TYPE_FREQSPLIT = "freqsplit"
NOISE_TYPE_FREQSPLIT_SLERP = "freqsplit_slerp"
FREQSPLIT_DEFAULT_CUTOFF = 0.1

MODALITY_OCT = "oct"
MODALITY_CT = "ct"

FORMAT_AUTO = "auto"
FORMAT_NII = "nii"
FORMAT_H5 = "h5"
FORMAT_NPY = "npy"