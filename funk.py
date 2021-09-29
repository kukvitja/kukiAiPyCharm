from Include.work_json import remember_data, get_remember_data


# pach_file_train_dataset = "data/traindataset.json"

pach_file_remember = "data/memory.json"

def remember_data_write(pach_file_remember = pach_file_remember, **kwargs):
    del kwargs['arr_text_input'][0]
    str = kwargs['arr_text_input'][0] + ' ' + kwargs['arr_text_input'][1] + ' ' + kwargs['arr_text_input'][2]
    remember_data(str, kwargs['arr_text_input'], pach_file_remember)


def get_remember(pach_file_remember = pach_file_remember, **kwargs):
    del kwargs['arr_text_input'][0]
    task = ' '.join(kwargs['arr_text_input'])
    return get_remember_data(task, pach_file_remember)