# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc                                 # number of classes
        numPoints = 3 if len(anchors[0])==9 else 2   # 2d or 3d
        self.no = nc + 5 if numPoints==2 else nc +7  # number of outputs per anchor
        self.nl = len(anchors)                        # number of detection layers
        self.na = len(anchors[0]) //numPoints        # number of anchors
        self.grid = [torch.zeros(1)] * self.nl       # init grid
        print("ch    : ", ch) # 2d: [128, 256, 512]  3d:[128, 256, 512]
        print("anchor: ", anchors)
        print("number of classes           : ", nc)        # 80
        print("number of coordinates       : ", numPoints) # 2
        print("number of outputs per anchor: ", self.no)   # 85
        print("number of detection layers  : ", self.nl)   # 3
        print("number of anchors          : ", self.na)    # 3
        print("grid: ", len(self.grid),len(self.grid[0]),len(self.grid[1]) ) # 3 1 1
        a = torch.tensor(anchors).float().view(self.nl, -1, numPoints)
        print("a: ",  a.shape)  # 2d: 3,3,2  or 3d: 3,3,3
        #print(ok)
        self.register_buffer('anchors', a)  # shape(nl,na,numPoints)
        self.m = []
        if numPoints==2:
            viewPars = [self.nl, 1, -1, 1, 1, numPoints]
            regBuff =  a.clone().view(viewPars) # shape(nl,1,na,1,1,2)
            print("viewPars : " , viewPars)           # [3, 1, -1, 1, 1, 2]
            print("regBuff.shape : ", regBuff.shape)  # [3, 1,  3, 1, 1, 2]
            self.register_buffer('anchor_grid', regBuff) # regBuff.shape :  torch.Size([3, 1, 3, 1, 1, 2])
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        else:
            viewPars = [self.nl, 1, -1,1, 1, 1, numPoints]
            regBuff =  a.clone().view(viewPars) # shape(nl,1,na,1,1,2)
            print("viewPars : " , viewPars)                 # [3, 1, -1, 1, 1, 1, 3]
            print("regBuff.shape : " , regBuff.shape)       # [3, 1, 3,  1, 1, 1, 3]
            self.register_buffer('anchor_grid', regBuff)
            print("ch : ",ch)
            self.m = nn.ModuleList(nn.Conv3d(x, self.no * self.na, 1) for x in ch)  # output conv
        #print(ok)
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        # print("x.shape : ", len(x) )      # 2d: 3
        # print("x.shape : ", (x[0].shape)) # 2d: torch.Size([1, 128, 32, 32])
        # print("x.shape : ", (x[1].shape)) # 2d: torch.Size([1, 256, 16, 16])
        # print("x.shape : ", (x[2].shape)) # 2d: torch.Size([1, 512, 8, 8])
        # print(ok)
        is2D = 1 if len(list(x[0].shape)) == 4 else 0
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # print("len(list(x[i].shape)) ................", len(list(x[i].shape)))
            # print("(x[i].shape) ................", x[i].shape)
            # print(ok)
            if is2D:
               bs, _, ny, nx = x[i].shape       # x(bs,255,20,20) to x(bs,3,20,20,85)
               x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            else:
                bs, _, ny, nx, nz = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx,nz).permute(0, 1, 3, 4,5, 2).contiguous()


            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    if is2D:
                        self.grid[i] = self._make_grid2D(nx, ny).to(x[i].device)
                    else:
                        self.grid[i] = self._make_grid3D(nx, ny, nz).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]              # wh
                z.append(y.view(bs, -1, self.no))
        out = x if self.training else (torch.cat(z, 1), x)
        # print("forwardout.shape : ",len(out) ) # 3 for 3d
        return out

    @staticmethod
    #this is generated during the training
    def _make_grid2D(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        out = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        #print("_make_grid2D out.shape : ",(out.shape) ) # 2d: torch.Size([1, 1, 56, 84, 2])
        return out

    @staticmethod
    #this is generated during the training
    def _make_grid3D(nx=20, ny=20, nz=20):
        yv, xv,zv = torch.meshgrid([torch.arange(ny), torch.arange(nx),torch.arange(nz)])
        out = torch.stack((xv, yv, zv), 3).view((1, 1, ny, nx,nz, 3)).float()
        print("_make_grid3D out.shape : ",(out.shape) ) # 2d: torch.Size([1, 1, 56, 84, 2])
        return out

class Model(nn.Module):
    def __init__(self, cfg='yolov5s3D.yaml', ch=3, nc=None, anchors=None,numPoints=2):  # model, input channels, number of classes
        super(Model, self).__init__()
        #cfg = '/home/ibr/Downloads/YOLOv5_vertebrae/yolov5/models/yolov5s3D.yaml'
        print("model config path: ", cfg)

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value


        modelOut = parse_model(deepcopy(self.yaml), ch=[ch], numPoints=2)  # model, savelist
        print(" modelOut ............................................")
        print(modelOut)
        print(" ........ ............................................")

        self.model, self.save = modelOut

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            if numPoints == 2:  # for 2D
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            else:  # for 3D
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s, s))])  # forward

            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        # print("model forward   x ", len(x))      #2d :1
        # print("model forward   x ", x[0].shape)  #2d : torch.Size([3, 256, 256])

        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            out =  torch.cat(y, 1), None  # augmented inference, train
        else:
            out = self.forward_once(x, profile)  # single-scale inference, train

        # print("model forward out",  len(out))       #2d :3
        # print("model forward  out ", out[0].shape)  #2d : torch.Size([1, 3, 32, 32, 85]
        # print("model forward  out ", out[1].shape)  #2d : torch.Size([1, 3, 16, 16, 85])
        # print("model forward  out ", out[2].shape)  #2d : torch.Size([1, 3,  8,  8, 85])
        # print(ok)
        return out

    def forward_once(self, x, profile=False):
        # print("model forward_once   x ", len(x))      #2d :1
        # print("model forward_once   x ", x[0].shape)  #2d : torch.Size([3, 256, 256])
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        # print("model forward_once   x ", len(x))      #2d :3
        # print("model forward_once   x ", x[0].shape)  #2d : torch.Size([1, 3, 32, 32, 85])
        # print("model forward_once   x ", x[1].shape)  #2d : torch.Size([1, 3, 16, 16, 85])
        # print("model forward_once   x ", x[2].shape)  #2d : torch.Size([1, 3,  8,  8, 85])

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv3d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv3d() + BatchNorm3d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch,numPoints=2):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']

    na = (len(anchors[0]) // numPoints) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5) if numPoints == 2 else na * (nc + 7)  # number of outputs = anchors * (classes + 5 or 7)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print("-----------------")
        # print("i :",i)
        # print("f :",f)
        # print("n :",n)
        # print("m :",m)
        # print("-----------------")
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                # print("   j a :",j,a)
            except:
                pass
        # print("=====================")
        # gd: depth_multiple
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        yolo2Dclasses = [Conv2D, GhostConv2D, GhostBottleneck2D, Bottleneck2D, SPP2D, DWConv2D, MixConv2D,Focus2D, CrossConv, BottleneckCSP2D, C32D, C3TR2D]
        yolo3Dclasses = [Conv3D, GhostConv3D, GhostBottleneck3D, Bottleneck3D, SPP3D, DWConv3D, MixConv3D,Focus3D, CrossConv, BottleneckCSP3D, C33D, C3TR3D]
        yolo2d3dClasses = list(set(yolo2Dclasses) | set(yolo3Dclasses))
        if m in yolo2d3dClasses:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP2D, C32D, C3TR2D, BottleneckCSP3D, C33D, C3TR3D]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in [nn.BatchNorm2d,nn.BatchNorm3d]:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m  in [Contract2D,Contract3D]:
            c2 = ch[f] * args[0] ** 2
        elif m in [Expand2D,Expand3D]:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        if i == 0:
            ch = []
        ch.append(c2)
        outModel = nn.Sequential(*layers), sorted(save)
    return outModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s3D.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file

    print("yolo opt: .......................")
    print(opt)
    print(".................................")

    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
