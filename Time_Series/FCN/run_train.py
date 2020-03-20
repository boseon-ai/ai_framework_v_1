import sys
import configparser
sys.path.append('/home/jovyan/framework_v_1/ai_framework_v_1/common/')

from data_manager import *
from log_manager import *
from FCN import *
from utils import *

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    data_dir = config['data']['data_dir']
    log_dir = config['log']['log_dir']
    name = config['model']['name']
    batch_size = int(config['model']['batch_size'])
    epoch = int(config['model']['epoch'])
    lr = float(config['model']['lr'])
    width = int(config['model']['width'])
    pred_itv = int(config['model']['pred_itv'])

    lm = Log_Manager(log_dir, name)
    dm = Data_Manager(data_dir, lm, batch_size, reshape_type='bhwc', frames=width, pred_itv=pred_itv)
    din, dout = dm.get_dims()
    model = FCN(log_dir, name, lr, dm, lm, din, dout)
    t_loss, v_loss = model.train(epoch)
    model.terminate()
    save_path = plot_loss_graph(t_loss, v_loss, name)
    lm.add_line('Loss graph location: {}'.format(save_path))

    model = FCN(log_dir, name, lr, dm, lm, din, dout)
    model.load()
    dict_rst, mse = model.accuracy()
    save_rst_scalable(dict_rst, rst_dir='./scalable_plots/')
    lm.add_line('Scalable plot location: ./scalable_plots/')
    lm.add_line('FINAL MSE (w/o overfitting): {}'.format(mse))
    lm.write_logs()
    
