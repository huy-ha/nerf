from os.path import exists


def read_dataset(data_dir_path):
    train_txt = open(f'{data_dir_path}/train.txt', 'r')
    test_txt = open(f'{data_dir_path}/test.txt', 'r')
    train_scene_dirs = [f'{data_dir_path}/{dir_path.rstrip()}'
                        for dir_path in train_txt.readlines()]
    test_scene_dirs = [f'{data_dir_path}/{dir_path.rstrip()}'
                       for dir_path in test_txt.readlines()]
    for dir_path in train_scene_dirs:
        assert exists(dir_path)
    for dir_path in test_scene_dirs:
        assert exists(dir_path)
    return train_scene_dirs, test_scene_dirs
