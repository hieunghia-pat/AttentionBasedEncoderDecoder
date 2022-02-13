# dataset configuration
batch_train = 32
batch_test = 1
image_dirs = [
                "../UIT-HWDB-dataset/UIT_HWDB_line_v2/train_data",
                "../UIT-HWDB-dataset/UIT_HWDB_line_v2/test_data"
            ] # for making vocab
train_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line_v2/train_data"
                    ] # for training
test_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line_v2/test_data"
                    ] # for testing
image_size = (-1, 128)
out_level = "character"
pretrained = None
            # "fasttext.vi.300d"
            # "phow2v.syllable.100d"
            # "phow2v.syllable.300d"

# model configuration
d_model = 256
image_channel = 3
embedding_dim = 300
dropout = 0.5
extractor = "resnet101"

## training configuration
max_epoch = 500
learning_rate = 1e-3
checkpoint_path = "saved_models/UIT-HWDB-line-character-level"
start_from = None
smoothing = 0.1