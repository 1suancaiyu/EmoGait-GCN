import sys
from ptflops import get_model_complexity_info
from thop import profile
import torch

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

if __name__ == "__main__":
    Emo_LiceGCN = import_class('net.sota.Classifier')
    STEP = import_class('net.classifier.Classifier')
    model_chose = "STEP"

    N = 128
    C = 3
    T = 48
    V = 21
    M = 1

    graph_dict = {'strategy': 'spatial'}

    # model = Emo_LiceGCN(C,V)

    model = STEP(
        C,
        V,
        graph_args = graph_dict
    )

    macs, params = get_model_complexity_info(model, (T, V, C, M), as_strings=True, print_per_layer_stat=True, verbose=True)
    print("Model: ", model_chose)
    print("Input tensor shape:", C, T, V, M)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    inputs = torch.randn(1, T, V, C, M)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops)
    print('params: ', params)