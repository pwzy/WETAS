import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()

        self.dilation = dilation

        self.diconv = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                dilation=dilation)

        self.conv1by1_skip = nn.Conv1d(in_channels=output_size,
                                       out_channels=output_size,
                                       kernel_size=1,
                                       dilation=1)

        self.conv1by1_out = nn.Conv1d(in_channels=output_size,
                                      out_channels=output_size,
                                      kernel_size=1,
                                      dilation=1)

    def forward(self, x):
        x = f.pad(x, (self.dilation, 0), "constant", 0)
        z = self.diconv(x)
        z = torch.tanh(z) * torch.sigmoid(z)
        s = self.conv1by1_skip(z)
        z = self.conv1by1_out(z) + x[:,:,-z.shape[2]:]
        return z, s

class DilatedCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers, pooling_type='max', granularity=1, \
      local_threshold=0.5, global_threshold=0.5, beta=10, split_size=500, dtw=None):
        super(DilatedCNN, self).__init__()

    # model = DilatedCNN(input_size=8,
                # hidden_size=128,
                # output_size=128,
                # kernel_size=2,
                # n_layers=7,
                # pooling_type='avg',
                # local_threshold=0.3,
                # granularity=4,
                # beta=0.1,
                # split_size=500,
                # dtw=dtw)

        self.input_size = input_size   # 8
        self.hidden_size = hidden_size # 128
        self.output_size = output_size # 128
        self.kernel_size = kernel_size # 2
        self.n_layers = n_layers       # 7
        
        self.rf_size = self.kernel_size ** self.n_layers # the size of receptive field  # 2^7=128
        self.pooling_type = pooling_type                 # 'avg'

        self.local_threshold = local_threshold   # 0.3
        self.global_threshold = global_threshold # 0.5
        self.granularity = granularity           # 4
        self.beta = beta                         # 0.1
        
        self.split_size = split_size # 500
        self.dtw = dtw               # soft dtw
        
        self.build_model(input_size, hidden_size, output_size, kernel_size, n_layers)

    def build_model(self, input_size, hidden_size, output_size, kernel_size, n_layers):
        # causal conv. layer
        self.causal_conv = nn.Conv1d(in_channels=input_size,
                                     out_channels=hidden_size,
                                     kernel_size=kernel_size,
                                     stride=1, dilation=1)

        # dilated conv. layer
        self.diconv_layers = nn.ModuleList()
        for i in range(n_layers):
            diconv = ResidualBlock(input_size=hidden_size,
                                   output_size=hidden_size,
                                   kernel_size=kernel_size,
                                   dilation=kernel_size**i)
            self.diconv_layers.append(diconv)

        # 1x1 conv. layer (for skip-connection)
        self.conv1by1_skip1 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=1, dilation=1)

        self.conv1by1_skip2 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=output_size,
                                        kernel_size=1, dilation=1)

        self.fc = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, x):
        
        x = x.transpose(1, 2)                    # x.shape: torch.Size([32, 8, 500])

        padding_size = self.rf_size - x.shape[2] # 128-500 = -372
        if padding_size > 0:
            x = f.pad(x, (padding_size, 0), "constant", 0)
        
        x = f.pad(x, (1, 0), "constant", 0)      # x.shape: torch.Size([32, 8, 501])
        z = self.causal_conv(x)                  # z.shape: torch.Size([32, 128, 500])

        out = torch.zeros(z.shape).cuda()        # out.shape: torch.Size([32, 128, 500])
        for diconv in self.diconv_layers:
            z, s = diconv(z)
            out += s

        out = f.relu(out)                              # torch.Size([32, 128, 500])
        out = self.conv1by1_skip1(out)                 # torch.Size([32, 128, 500])
        out = f.relu(out)                              # torch.Size([32, 128, 500])
        out = self.conv1by1_skip2(out).transpose(1, 2) # torch.Size([32, 500, 128])
        return out                                     # torch.Size([32, 500, 128])

    def get_scores(self, x):
        ret = {}
        out = self.forward(x)
        ret['output'] = out                            # torch.Size([32, 500, 128])

        # Compute weak scores
        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1)   # torch.Size([32, 128])
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]
        
        ret['wscore'] = torch.sigmoid(self.fc(_out).squeeze(dim=1))
        ret['wpred'] = (ret['wscore'] >= self.global_threshold).type(torch.cuda.FloatTensor)

        # Compute dense scores
        h = self.fc(out).squeeze(dim=2)
        ret['dscore'] = torch.sigmoid(h)
        ret['dpred'] = (ret['dscore'] >= self.local_threshold).type(torch.cuda.FloatTensor)
        return ret

    def get_seqlabel(self, actmap, wlabel):
        actmap   *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1])                                           # torch.Size([32, 500])
        seqlabel  = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor)                                # torch.Size([32, 500])
        seqlabel  = f.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0) # torch.Size([32, 504])
        seqlabel  = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity)))  # torch.Size([32, 4, 126])
        seqlabel  = torch.max(seqlabel, dim=2)[0]                                                                # torch.Size([32, 4])

        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1) #torch.Size([32, 6])

        return seqlabel

    def dtw_loss(self, out, wlabel):
        # dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)

        h = self.fc(out).squeeze(dim=2)                     # torch.Size([32, 500])
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2)) # torch.Size([32, 500])

        with torch.no_grad():
            # Activation map
            actmap = h                                         # torch.Size([32, 500])
            actmin = torch.min(actmap, dim=1)[0]               # torch.Size([32])
            actmap = actmap - actmin.unsqueeze(dim=1)          # torch.Size([32, 500])
            actmax = torch.max(actmap, dim=1)[0]               # torch.Size([32])
            actmap = actmap / actmax.unsqueeze(dim=1)          # torch.Size([32, 500])
            # Sequential labels
            pos_seqlabel = self.get_seqlabel(actmap, wlabel)   # torch.Size([32, 6])
            neg_seqlabel = self.get_seqlabel(actmap, 1-wlabel) # torch.Size([32, 6])

        pos_dist = self.dtw(pos_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size # torch.Size([32])
        neg_dist = self.dtw(neg_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size # torch.Size([32])
        loss = f.relu(self.beta + pos_dist - neg_dist)
        return loss

    def get_alignment(self, label, score):
        # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1)) # torch.Size([32, 6, 500])
        indices = torch.max(A, dim=1)[1]                                   # torch.Size([32, 500])
        return torch.gather(label, 1, indices)                             # torch.Size([32, 500])

    def get_dpred(self, out, wlabel):
        h = self.fc(out).squeeze(dim=2)                     # torch.Size([32, 500])
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2)) # torch.Size([32, 500])

        with torch.no_grad():
            # Activation map
            actmap = h                                # torch.Size([32, 500])
            actmin = torch.min(actmap, dim=1)[0]      # torch.Size([32])
            actmap = actmap - actmin.unsqueeze(dim=1) # torch.Size([32, 500])
            actmax = torch.max(actmap, dim=1)[0]      # torch.Size([32])
            actmap = actmap / actmax.unsqueeze(dim=1) # torch.Size([32, 500])
            # Sequential labels
            seqlabel = self.get_seqlabel(actmap, wlabel) # torch.Size([32, 6])  

        return self.get_alignment(seqlabel, dscore)


if __name__ == "__main__":
    # model = DilatedCNN(input_size=train_dataset.input_size,
                # hidden_size=args.hidden_size,
                # output_size=args.output_size,
                # kernel_size=args.kernel_size,
                # n_layers=args.n_layers,
                # pooling_type=args.pooling_type,
                # local_threshold=args.local_threshold,
                # granularity=args.granularity,
                # beta=args.beta,
                # split_size=args.split_size,
                # dtw=dtw)
    # /hidden_size=128, output_size=128, kernel_size=2, n_layers=7, dropout=0.5, pooling_type='avg', local_threshold=0.3, granularity=4, beta=0.1, gamma=0.1, batch_size=32, n_epochs=200, learning_rate=0.0001, gpuidx=0, patience=50, stopping='f1', seed=0, dataset='EMG', split_size=500, data_dir='./data/EMG'
    import ipdb 

    from softdtw_cuda import SoftDTW
    dtw = SoftDTW(use_cuda=True, gamma=0.1, normalize=False)
    model = DilatedCNN(input_size=8,
                hidden_size=128,
                output_size=128,
                kernel_size=2,
                n_layers=7,
                pooling_type='avg',
                local_threshold=0.3,
                granularity=4,
                beta=0.1,
                split_size=500,
                dtw=dtw)
    model = model.cuda()
    data = torch.randn(32, 500, 8).cuda()    

    out = model.get_scores(data)
    wlabel = torch.tensor([1], dtype=torch.float32, device=torch.device('cuda:0'))
    dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)


    dpred = model.get_dpred(out['output'], out['wpred'])
    print(dpred)

    # ipdb.set_trace()

    # print(out)
    # print(model)


    # for epoch in range(args.n_epochs):
        # model.train()
        # total_step = len(train_loader)
        # total, total_loss, total_bce_loss, total_dtw_loss = 0, 0, 0, 0
        # for itr, batch in enumerate(train_loader):
            # data = batch['data'].cuda()
            # wlabel = batch['wlabel'].cuda()

            # ipdb.set_trace()

            # out = model.get_scores(data)
            # bce_loss = bce(out['wscore'], wlabel)
            # dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)
            # loss = bce_loss + dtw_loss

            # with torch.no_grad():
                # total += data.size(0)
                # total_bce_loss += bce_loss.item() * data.size(0)
                # total_dtw_loss += dtw_loss.item() * data.size(0)
                # total_loss += loss.item() * data.size(0)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
