# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import datetime
import os
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from data.MultiEpochsDataLoader import MultiEpochsDataLoader
from config.config_loader import Config
from utils.metric import compress_wiki, compress, calculate_top_map
from data import datasets
from models.models import ImgNet, TxtNet
import os.path as osp
from tqdm import tqdm
from utils.meters import AverageMeter
from utils.plotter import get_plotter
import visdom
from config.logger import logger


class HNH:
    def __init__(self, config: Config):
        self.load_config(config)

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(self.device)

        self.train_dataset, self.test_dataset, self.database_dataset = datasets.get_dataset(self.dataset_name,
                                                                                            self.data_path)

        self.loss_store = ["common space loss", 'intra loss', 'inter loss', 'loss']
        self.plotter = get_plotter(self.name) if visdom else None
        self.loss_store = self._loss_store_init(self.loss_store)

        self.CodeNet_I = ImgNet(code_len=self.bit)
        self.FeatNet_I = ImgNet(code_len=self.bit)
        self.CodeNet_T = TxtNet(code_len=self.bit, txt_feat_len=datasets.DataSetBase.txt_feat_len)

        self.set_train_loader()
        self.set_optimizer()

    def set_train_loader(self):
        # Data Loader (Input Pipeline)
        self.train_loader = MultiEpochsDataLoader(dataset=self.train_dataset,
                                                  batch_size=self.BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=self.num_workers,
                                                  drop_last=True,
                                                  pin_memory=True)

        self.test_loader = MultiEpochsDataLoader(dataset=self.test_dataset,
                                                 batch_size=self.BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)

        self.database_loader = MultiEpochsDataLoader(dataset=self.database_dataset,
                                                     batch_size=self.BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=self.num_workers)

    def set_optimizer(self):
        if self.dataset_name == "wiki":
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.fc_encode.parameters(), 'lr': self.lr_img},
                                          {'params': self.CodeNet_I.alexnet.classifier.parameters(),
                                           'lr': self.lr_img}],
                                         momentum=self.momentum, weight_decay=self.weight_decay)

        if self.dataset_name == "mirFlickr25k" or self.dataset_name == "nusWide":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=self.lr_img, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=self.lr_txt, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

    def load_config(self, config: Config):
        self.logger = logger
        self.name = 'HNH-O'
        self.method = config.training['method']
        self.dataset_name = config.training['dataName']
        self.model_dir = config.training['modelDir']
        self.bit = int(config.training['bit'])
        self.BATCH_SIZE = int(config.training['batchSize'])
        self.device = config.training['device']
        self.max_epoch = config.training['numEpoch']
        self.num_workers = config.training['numWorkers']
        self.hnh2 = config.training['hnh2']
        self.dataset_config = config.dataset_config
        self.data_path = config.dataset_config['dataPath']
        self.lr_img = self.dataset_config['lrImg']
        self.lr_txt = self.dataset_config['lrTxt']
        self.weight_decay = self.dataset_config['weightDecay']
        self.momentum = self.dataset_config['momentum']
        self.eval_interval = self.dataset_config['evalInterval']
        self.eval = self.dataset_config['eval']
        self.gamma = self.dataset_config['gamma']
        self.lambda_ = self.dataset_config['lambda']
        self.beta = self.dataset_config['beta']
        self.alpha = self.dataset_config['alpha']
        self.k_x = self.dataset_config['kX']
        self.k_y = self.dataset_config['kY']

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        cuda = bool(config.training['cuda'])
        self.img_training_transform = config.img_training_transform
        self.img_valid_transform = config.img_valid_transform
        self.txt_training_transform = config.txt_training_transform
        self.txt_valid_transform = config.txt_valid_transform
        t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # sys.stdout = Logger(os.path.join('..', 'logs', self.name, self.dataset_name, t + '.txt'))
        if cuda:
            print("using gpu device: %s" % str(self.device))
        else:
            print("using cpu")
        print("training transform:")
        print("img:", config.img_training_transform)
        print("txt:", config.txt_training_transform)
        print("valid transform")
        print("img:", config.img_valid_transform)
        print("txt:", config.txt_valid_transform)

    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        for idx, (img, F_T, labels, _) in enumerate(tqdm(self.train_loader)):
            img = Variable(img.cuda())
            # LDA topic vectors or the tag occurrence features ？
            F_T = torch.FloatTensor(F_T.numpy()).cuda()  # batch_size * 1386
            labels = Variable(labels.cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            # 从AlexNet网络提取FI
            F_I, _, _ = self.FeatNet_I(img)  # batch_size * 4096
            _, hid_I, code_I = self.CodeNet_I(img)
            _, hid_T, code_T = self.CodeNet_T(F_T)

            # ==========计算S_tilde=========
            F_I = F.normalize(F_I)
            A_x = torch.matmul(F_I, F_I.t())
            # t_text = torch.squeeze(text, dim=1).squeeze(dim=2)
            F_T = F.normalize(F_T)
            A_y = torch.matmul(F_T, F_T.t())
            A_tilde_x = self.k_x * (A_x * (A_x.t().mm(A_x))) - 1
            A_tilde_y = self.k_y * (A_y * (A_y.t().mm(A_y))) - 1
            S_tilde = self.gamma * A_tilde_x + (1 - self.gamma) * A_tilde_y

            # train

            B_x = F.tanh(hid_I).t()
            B_y = F.tanh(hid_T).t()
            B_x = F.normalize(B_x)
            B_y = F.normalize(B_y)

            if (self.hnh2):
                J3 = self.lambda_ * F.mse_loss(S_tilde, B_x.t() @ B_y)
                # HNH-2
                J1 = self.alpha * F.mse_loss(S_tilde, B_x.t() @ B_x)
                J2 = self.beta * F.mse_loss(S_tilde, B_y.t() @ B_y)
            else:
                # # 计算U
                Ic = torch.eye(32).cuda()
                Ic_1 = torch.eye(32).cuda()
                # 计算 U = (2 * Ic + (beta / alpha) * Bx * Bx^T + (beta / alpha) * By * By^T)^(-1)
                b_d_a = (self.beta / self.alpha)
                U = torch.inverse(2 * Ic + b_d_a * B_x @ B_x.t() + b_d_a * B_y @ B_y.t())
                # 计算临时矩阵 temp = (Bx + By) * (Ic + (beta / alpha) * S_tilde)
                temp = (B_x + B_y) @ (Ic_1 + b_d_a * S_tilde)

                # 计算最终结果 U * temp
                U = U @ temp

                # 计算损失
                J1 = self.alpha * (F.mse_loss(U, B_x) + F.mse_loss(U, B_y))
                J2 = self.beta * (F.mse_loss(S_tilde, U.t() @ B_x) + F.mse_loss(S_tilde, U.t() @ B_y))
                J3 = self.lambda_ * F.mse_loss(S_tilde, B_x.t() @ B_y)

            loss = J1 + J2 + J3

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.loss_store['common space loss'].update(J1.item())
            self.loss_store['intra loss'].update(J2.item())
            self.loss_store['inter loss'].update(J3.item())
            self.loss_store['loss'].update(loss.item())
            self.remark_loss(J1, J2, J3,loss)
        # eval the Model
        if (epoch + 1) % self.eval_interval == 0:
            self.evaluate()
        self.print_loss(epoch)
        self.plot_loss("loss")
        self.reset_loss()
        # self.lr_schedule()
        self.plotter.next_epoch()
        # save the model
        if epoch + 1 == self.max_epoch:
            self.save_checkpoints(step=epoch + 1)

    def evaluate(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if self.dataset_name == "wiki":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T,
                                                                   self.database_dataset, self.test_dataset)

        if self.dataset_name == "mirFlickr25k" or self.dataset_name == "nusWide":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        if self.plotter:
            self.plotter.plot("mAP", 'i->t', MAP_I2T.item())
            self.plotter.plot("mAP", "t->i", MAP_T2I.item())
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(self.model_dir, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.model_dir, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

    @staticmethod
    def _loss_store_init(loss_store):
        """
        initialize loss store, transform list to dict by (loss name -> loss register)
        :param loss_store: the list with name of loss
        :return: the dict of loss store
        """
        dict_store = {}
        for loss_name in loss_store:
            dict_store[loss_name] = AverageMeter()
        loss_store = dict_store
        return loss_store

    def plot_loss(self, title, loss_store=None):
        """
        plot loss in loss_store at a figure
        :param title: the title of figure name
        :param loss_store: the loss store to plot, if none, the default loss store will plot
        """
        if loss_store is None:
            loss_store = self.loss_store
        if self.plotter:
            for name, loss in loss_store.items():
                self.plotter.plot(title, name, loss.avg)

    def print_loss(self, epoch, loss_store=None):
        loss_str = "epoch: [%3d/%3d], " % (epoch + 1, self.max_epoch)
        if loss_store is None:
            loss_store = self.loss_store
        for name, value in loss_store.items():
            loss_str += name + " {:4.3f}".format(value.avg) + "\t"
        print(loss_str)
        sys.stdout.flush()

    def reset_loss(self, loss_store=None):
        if loss_store is None:
            loss_store = self.loss_store
        for store in loss_store.values():
            store.reset()

    def remark_loss(self, *args, n=1):
        """
        store loss into loss store by order
        :param args: loss to store
        :return:
        """
        for i, loss_name in enumerate(self.loss_store.keys()):
            if isinstance(args[i], torch.Tensor):
                self.loss_store[loss_name].update(args[i].item(), n)
            else:
                self.loss_store[loss_name].update(args[i], n)


def run(config_path='default_config.yml', **kwargs):
    config = Config(config_path)
    hnh = HNH(config)
    if hnh.eval:
        hnh.load_checkpoints()
        hnh.eval()

    else:
        for epoch in range(hnh.max_epoch):
            # train the Model
            hnh.train(epoch)
