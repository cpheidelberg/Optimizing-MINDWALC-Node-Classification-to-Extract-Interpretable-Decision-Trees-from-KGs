import os, shutil

def make_folder(dir):
    '''
    generates passes direction if not exist.
    '''
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)

    return dir

def create_new_result_folder_in(dir, overwrite=False, foldername='results'):
    '''
    generates passes direction if not exist.
    '''
    if overwrite:
        new_path = f'{dir}/{foldername}'
    else:
        try:
            folders = [name for name in os.listdir(dir)]
        except FileNotFoundError:
            make_folder(dir)
            folders = [name for name in os.listdir(dir)]
        new_path = f'{dir}/{foldername}0'
        if folders:
            nums = []
            for f in folders:
                try:
                    nums.append(int(f.split(foldername)[1]))
                except:
                    pass
            if nums:
                new_num = max(nums) + 1
                new_path = dir + f'/{foldername}{new_num}'

    shutil.rmtree(new_path, ignore_errors=True)
    try:
        os.makedirs(new_path)
    except FileExistsError as error:
        if overwrite:
            raise FileExistsError(f'Failed to delete folder {new_path}. '
                                  f'This can happen if a file is open in the folder.')
        else:
            raise error
    return new_path