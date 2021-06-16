from environment_setup import read_config
from models.cnn_base.densenet import DenseNet161, DenseNet121MultiView
from models.graph_base.base_model import BaseGNNModel
from models.graph_extra_labels.graph_model import GNNNetwork


def create_model(model_type):
    configs = read_config()
    if model_type == 'dense':
        num_classes = configs[model_type].getint('num_classes')
        in_channels = configs[model_type].getint('in_channels')
        dropout_p = configs[model_type].getfloat('dropout_p')
        return DenseNet161(in_channels=in_channels, out_size=num_classes, dropout_p=dropout_p)
    elif model_type == 'graph':
        num_classes = configs[model_type].getint('num_classes')
        in_channels = configs[model_type].getint('in_channels')
        img_feat_dim = configs[model_type].getint('img_feat_dim')
        text_feat_dim = configs[model_type].getint('text_feat_dim')
        hidden_dim = configs[model_type].getint('hidden_dim')
        dropout_p = configs[model_type].getfloat('dropout_p')
        return GNNNetwork(img_feat_dim=img_feat_dim, text_feat_dim=text_feat_dim, hidden_dim=hidden_dim,
                          num_classes=num_classes, in_channels=in_channels, dropout_p=dropout_p)
    elif model_type == 'base':
        in_channels = configs[model_type].getint('in_channels')
        text_feat_dim = configs[model_type].getint('text_feat_dim')
        hidden_dim = configs[model_type].getint('hidden_dim')
        out_channels = configs[model_type].getint('out_channels')
        dropout_p = configs[model_type].getfloat('dropout_p')
        load_model_num = configs[model_type].getint('load_model_num', fallback=None)
        return BaseGNNModel(in_channels=in_channels, text_feat_dim=text_feat_dim, hidden_dim=hidden_dim,
                            out_channels=out_channels, dropout_p=dropout_p, load_model_num=load_model_num)
    elif model_type == 'dense_multi':
        num_classes = configs[model_type].getint('num_classes')
        in_channels = configs[model_type].getint('in_channels')
        dropout_p = configs[model_type].getfloat('dropout_p')
        return DenseNet121MultiView(in_channels=in_channels, out_size=num_classes, dropout_p=dropout_p)
    else:
        raise AttributeError("Invalid Model type selected")
