import h5py
import math
import os
import matplotlib as mpl
import numpy as np
import sklearn.metrics as skm
import torch
import torch.optim as optim
import torch.nn as nn

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import rcParams

from torch.nn import ModuleList, ReLU
from torchlight.torchlight import IO


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()

        #self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[1][1][0].gcn.conv
        first_layer.register_backward_hook(hook_function)

    # def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ModuleList):
                for each_module in module:
                    each_module.relu.register_backward_hook(relu_backward_hook_function)
                    each_module.relu.register_forward_hook(relu_forward_hook_function)

    #
    # def update_relus(self):
    #     """
    #         Updates relu activation functions so that
    #             1- stores output in forward pass
    #             2- imputes zero for gradient values that are less than zero
    #     """
    #     def relu_backward_hook_function(module, grad_in, grad_out):
    #         """
    #         If there is a negative gradient, change it to zero
    #         """
    #         # Get last forward output
    #         corresponding_forward_output = self.forward_relu_outputs[-1]
    #         corresponding_forward_output[corresponding_forward_output > 0] = 1
    #         modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
    #         del self.forward_relu_outputs[-1]  # Remove last forward output
    #         return (modified_grad_out,)
    #
    #     def relu_forward_hook_function(module, ten_in, ten_out):
    #         """
    #         Store results of forward pass
    #         """
    #         self.forward_relu_outputs.append(ten_out)
    #
    #     # Loop through layers, hook up ReLUs
    #     for pos, module in self.model.features._modules.items():
    #         if isinstance(module, ReLU):
    #             module.register_backward_hook(relu_backward_hook_function)
    #             module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):

        # Forward pass
        output, _ = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(output.size()).zero_()
        for idx in range(output.shape[0]):
            one_hot_output[idx, target_class[idx]] = 1
        # Backward pass
        output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_map(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    while '_' not in all_models[-1]:
        all_models = all_models[:-1]
    best_model = all_models[-1]
    all_us = list(find_all_substr(best_model, '_'))
    return int(best_model[5:all_us[0]]), float(best_model[all_us[0]+4:all_us[1]])


def plot_confusion_matrix(confusion_matrix, title='CM', fontsize=50):
    mpl.style.use('seaborn')
    rcParams['text.usetex'] = True
    rcParams['axes.titlepad'] = 20

    columns = ('Angry', 'Neutral', 'Happy', 'Sad')
    rows = columns
    fig, ax = plt.subplots()

    # Set colors
    colors = np.empty((4, 4))
    colors[0] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['goldenrod'], 1.0))
    colors[1] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['bisque'], 1.0))
    colors[2] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['paleturquoise'], 1.0))
    colors[3] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['limegreen'], 1.0))
    # colors[4] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightpink'], 1.0))
    # colors[5] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['hotpink'], 1.0))
    # colors[6] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['mistyrose'], 1.0))
    # colors[7] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightsalmon'], 1.0))
    # colors[8] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lavender'], 1.0))
    # colors[9] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['cornflowerblue'], 1.0))

    n_rows = len(confusion_matrix)
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        # plt.bar(index, confusion_matrix[row], bar_width, bottom=y_offset,
        #                                                 color=colors[row])
        y_offset = y_offset + confusion_matrix[row]
        cell_text.append(['%d' % (x) for x in confusion_matrix[row]])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')
    the_table.set_fontsize(fontsize)
    the_table.scale(1, fontsize/7)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2,
                        bottom=0.1,
                        top=0.99)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    plt.ylabel("\# predictions of each class", fontsize=fontsize)
    plt.xticks([])
    fig.savefig('figures/'+title+'.png', bbox_inches='tight')


def to_multi_hot(labels, threshold=0.25):
    labels_out = np.zeros_like(labels)
    hot_idx = np.argwhere(labels > threshold)
    labels_out[hot_idx[:, 0], hot_idx[:, 1]] = 1
    return labels_out


def calculate_metrics(y_true, y_pred, thres=0.25, eval_time=False):
    y_true_multi_hot = to_multi_hot(y_true)
    aps = skm.average_precision_score(y_true_multi_hot, y_pred, average=None)
    nans = np.sum(np.isnan(aps))
    mean_ap = np.sum(aps) / (len(aps) - nans)
    y_pred_multi_hot = to_multi_hot(y_pred)
    # f1_score = skm.f1_score(y_true_multi_hot, y_pred_multi_hot, average='micro')
    f1_score = 0
    return aps, mean_ap, f1_score

def load_pretrain_weights(model, weights_dir):
    # 第一步:读取当前模型参数
    model_dict = model.state_dict()
    # 第二步:读取预训练模型
    pretrained_dict = torch.load(weights_dir)
    temp = {}
    for k,v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v) and 'fcn' not in k:
                temp[k] = v
        except:
            pass
    #第三步:使用顶训练的模型更新当前模型参数
    model_dict.update(temp)
    # 第四步:加载模型参数
    model.load_state_dict(model_dict)

def load_Bk_layer10_pretrain_weights(model, weights_dir):
    # 第一步:读取当前模型参数
    model_dict = model.state_dict()
    # 第二步:读取预训练模型
    pretrained_dict = torch.load(weights_dir)
    temp = {}
    for k,v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v) and 'fc' not in k:
                temp[k] = v
        except:
            pass

    #第三步:使用顶训练的模型更新当前模型参数
    model_dict.update(temp)

    # 第四步:加载模型参数
    model.load_state_dict(model_dict)

    print(model_dict.items())

class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, num_classes, device='cuda:0', verbose=True):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.verbose = verbose
        self.io = IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)

        # change model
        Model = import_class(self.args.model)
        self.model = Model(C, num_classes)
        print(self.model)


        self.model.cuda('cuda:0')
        self.model.apply(weights_init)

        # load pre-trained weights
        load_Bk_layer10_pretrain_weights(self.model, weights_dir=self.args.weight_dir)
        self.io.print_log('Load weights from {}.'.format(self.args.weight_dir))

        self.loss = nn.BCEWithLogitsLoss()
        self.best_loss = math.inf
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args.topk)))
        self.best_mean_map = 0
        self.map_updated = False

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and\
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            if self.verbose:
                self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            if self.verbose:
                self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)
            if self.verbose:
                self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.map_updated = True
        else:
            self.map_updated = False
        if self.verbose:
            print_epoch = self.best_epoch if self.best_epoch is not None else 0
            self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}% (epoch: {:d}).'.
                              format(k, accuracy, self.best_accuracy[0, k-1], print_epoch))

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        ap_values = []
        mean_ap_value = []

        for data, labels in loader:
            # get data
            data = data.float().to(self.device)
            labels = labels.float().to(self.device)

            # forward
            output, _ = self.model(data.unsqueeze(-1))
            loss = self.loss(output, labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['aps'], self.iter_info['mean_ap'], self.iter_info['f1'] =\
                calculate_metrics(labels.detach().cpu().numpy(),
                                  output.detach().cpu().numpy())
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            ap_values.append(self.iter_info['aps'])
            mean_ap_value.append(self.iter_info['mean_ap'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_ap'] = np.mean(ap_values, axis=0)
        self.epoch_info['mean_map'] = np.mean(mean_ap_value)
        self.show_epoch_info()
        if self.verbose:
            self.io.print_timer()
        # for k in self.args.topk:
        #     self.calculate_topk(k, show=False)
        # if self.accuracy_updated:
        #     self.model.extract_feature()

    def per_test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        ap_values = []
        mean_ap_value = []
        result_frag = []
        label_frag = []
        count = 0

        for data, labels in loader:

            # get data
            data = data.float().to(self.device)
            labels = labels.float().to(self.device)

            # inference
            with torch.no_grad():
                output, _ = self.model(data.unsqueeze(-1))
            # result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, labels)
                loss_value.append(loss.item())
                aps, mean_ap, f1 = \
                    calculate_metrics(labels.detach().cpu().numpy(),
                                      output.detach().cpu().numpy())
                ap_values.append(aps)
                mean_ap_value.append(mean_ap)
                label_frag.append(labels.data.cpu().numpy())

        # self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['mean_ap'] = np.mean(ap_values, axis=0)
            self.epoch_info['mean_map'] = np.mean(mean_ap_value)
            if self.epoch_info['mean_map'] > self.best_mean_map:
                self.best_mean_map = self.epoch_info['mean_map']
                self.map_updated = True
            else:
                self.map_updated = False
            self.show_epoch_info()

            # show top-k accuracy
            # for k in self.args.topk:
            #     self.show_topk(k)

    def train(self):

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            if self.verbose:
                self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            if self.verbose:
                self.io.print_log('Done.')

            # evaluation
            if epoch > 200:
                if (epoch % self.args.eval_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    if self.verbose:
                        self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.per_test()
                    if self.verbose:
                        self.io.print_log('Done.')

                # save model and weights
                if self.map_updated:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.args.work_dir,
                                            'epoch{}_map{:.2f}_model.pth.tar'.format(epoch, self.best_mean_map)))
                    if self.epoch_info['mean_loss'] < self.best_loss:
                        self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        if self.verbose:
            self.io.print_log('Model:   {}.'.format(self.args.model))
            self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        if self.verbose:
            self.io.print_log('Evaluation Start:')
        self.per_test()
        if self.verbose:
            self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def smap(self):
        # self.model.eval()
        loader = self.data_loader['test']

        for data, label in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            GBP = GuidedBackprop(self.model)
            guided_grads = GBP.generate_gradients(data, label)

    def load_best_model(self):
        filename = os.path.join(self.args.work_dir, 'epoch363_map0.90_model.pth.tar')
        self.model.load_state_dict(torch.load(filename))

    def generate_predictions(self, data, num_classes, joints, coords):
        # fin = h5py.File('../data/features'+ftype+'.h5', 'r')
        # fkeys = fin.keys()
        self.load_best_model()
        num_samples = data.shape[0]
        with torch.no_grad():
            preds = self.model(torch.from_numpy(data[:num_samples]).float().to(self.device).unsqueeze(-1))[0].cpu().numpy()
        # labels_pred = np.zeros(data.shape[0])
        # output = np.zeros((data.shape[0], num_classes))
        # for i, each_data in enumerate(zip(data)):
        #     # get data
        #     each_data = each_data[0]
        #     each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
        #     each_data = torch.from_numpy(each_data).float().to(self.device)
        #     # get label
        #     with torch.no_grad():
        #         output[i], _ = self.model(each_data)
        #         labels_pred[i] = np.argmax(output[i])
        return np.abs(preds) / np.linalg.norm(preds, ord=1, axis=-1)[:, None]

    def generate_confusion_matrix(self, ftype, data, labels, num_classes, joints, coords):
        self.load_best_model()
        labels_pred = self.generate_predictions(data, num_classes, joints, coords)

        hit = np.nonzero(labels_pred == labels)
        miss = np.nonzero(labels_pred != labels)
        confusion_matrix = np.zeros((num_classes, num_classes))
        for hidx in np.arange(len(hit[0])):
            confusion_matrix[np.int(labels[hit[0][hidx]]), np.int(labels_pred[hit[0][hidx]])] += 1
        for midx in np.arange(len(miss[0])):
            confusion_matrix[np.int(labels[miss[0][midx]]), np.int(labels_pred[miss[0][midx]])] += 1
        confusion_matrix = confusion_matrix.transpose()
        plot_confusion_matrix(confusion_matrix)

    def save_best_feature(self, ftype, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_map(self.args.work_dir)
        else:
            best_accuracy = self.best_accuracy.item()
        filename = os.path.join(self.args.work_dir,
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
        self.model.load_state_dict(torch.load(filename))
        features = np.empty((0, 64))
        fCombined = h5py.File('../data/features'+ftype+'.h5', 'r')
        fkeys = fCombined.keys()
        dfCombined = h5py.File('../data/deepFeatures'+ftype+'.h5', 'w')
        for i, (each_data, each_key) in enumerate(zip(data, fkeys)):

            # get data
            each_data = np.reshape(each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]
                dfCombined.create_dataset(fname, data=feature)
                features = np.append(features, np.array(feature).reshape((1, feature.shape[0])), axis=0)
        dfCombined.close()
        return features

