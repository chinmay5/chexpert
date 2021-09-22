from dataset.chexpert.ChexpertDataloader import get_chexpert_data_loader_dict
from dataset.mimic.MimicCXRDataloader import get_mimic_dataloader_dict
from environment_setup import read_config


def get_dataloader(args, visualization=False):
    configs = read_config()
    if visualization:
        dataset_name = configs['visualization'].get('active_dataset')
    else:
        dataset_name = configs['data'].get('active_dataset')
    uncertainty_labels = configs['data'].get('uncertainty_labels')
    if dataset_name == 'chexpert':
        print("Loading CheXpert dataset")
        return get_chexpert_data_loader_dict(uncertainty_labels=uncertainty_labels, batch_size=args.batch_size, num_workers=args.num_workers, build_grph=args.build_graph)
    elif dataset_name == 'mimic':
        print("Loading MIMIC dataset")
        return get_mimic_dataloader_dict(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise AttributeError("Invalid selection made for dataset")
