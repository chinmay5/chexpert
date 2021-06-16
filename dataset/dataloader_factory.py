from dataset.chexpert.ChexpertDataloader import get_chexpert_data_loader_dict
from dataset.mimic.MimicCXRDataloader import get_mimic_dataloader_dict
from environment_setup import read_config


def get_dataloader(args):
    configs = read_config()
    dataset_name = configs['data'].get('active_dataset')
    if dataset_name == 'chexpert':
        return get_chexpert_data_loader_dict(uncertainty_labels=args.uncertainty_labels, batch_size=args.batch_size, num_workers=args.num_workers, build_grph=args.build_graph)
    elif dataset_name == 'mimic':
        return get_mimic_dataloader_dict(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise AttributeError("Invalid selection made for dataset")
