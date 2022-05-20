import torch.nn as nn
import torch
import yaml


train_file = open('./param/train_param.yaml','r',encoding='utf-8')

train_param = yaml.load(train_file, Loader=yaml.FullLoader)



lr_decay = train_param['lr_decay']
epoch_decay = train_param['epoch_decay']
lr = train_param['lr']
device = train_param['device']




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sz=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(sz, sz), padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(sz, sz), padding=1)

    def forward(self, x):
        output = self.leaky_relu(self.conv1(x))
        output = self.leaky_relu(self.conv2(output))
        return output


class MaxPoolBlock(nn.Module):
    def __init__(self, channels):
        super(MaxPoolBlock, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout = nn.Dropout2d()
        self.bn = nn.BatchNorm2d(channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.max_pool(x)
        output = self.dropout(output)
        output = self.bn(output)
        output = self.leaky_relu(output)
        return output


class FeatureNet(nn.Module):
    def __init__(self,in_channels=4, sz=2):
        super(FeatureNet, self).__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=32, sz=sz)
        self.maxPool = MaxPoolBlock(32)

    def forward(self, x):
        output = self.conv(x)
        output = self.maxPool(output)
        return output


class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=16),
            MaxPoolBlock(16)
        )
        self.fc = nn.Linear(1920, 26)

    def forward(self, x):
        output = self.conv(x)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output


class DomainNet(nn.Module):
    def __init__(self):
        super(DomainNet, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=32),
            MaxPoolBlock(32),
            ConvBlock(in_channels=32, out_channels=64),
            MaxPoolBlock(64)
        )
        self.fc = nn.Linear(3968, 5)

    def forward(self, x):
        output = self.conv(x)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output


def train1(fea_net, pred_net, train_loader, test_loader, epochs, criterion):
    print("=================step1: fix domain_net and train others====================")
    optimizer1 = torch.optim.Adam(fea_net.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(pred_net.parameters(), lr=lr)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,step_size=epoch_decay,gamma=lr_decay)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=epoch_decay,gamma=lr_decay)
    for epoch in range(epochs):
        all_loss = 0.0
        val_loss = 0.0
        total_train = len(train_loader)
        total_test = len(test_loader)
        for i, (data, label_x, label_y, _) in enumerate(train_loader):
            print("\r {} / {}".format(i, total_train), flush=True, end='')
            data = data.to(device)
            label_x = label_x.long().to(device)
            label_y = label_y.long().to(device)
            fea = fea_net(data)
            pred = pred_net(fea)
            loss = criterion(pred[:, :18], label_x) + criterion(pred[:, 18:], label_y)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            all_loss += loss.item()
        scheduler1.step()
        scheduler2.step()
        for data, label, _ in test_loader:
            data = data.to(device)
            label = label.long().to(device)
            fea = fea_net(data)
            pred = pred_net(fea)
            loss = criterion(pred, label)
            val_loss += loss.item()

        print('Epoch [{}/{}],Loss: {:.4f}, val_Loss: {:.4f}'.format(epoch + 1, epochs, all_loss / total_train,
                                                                    val_loss / total_test))


def train2(fea_net, domain_net, train_loader, test_loader, epochs, criterion):
    print("==================step2: train domain_net and fix others=================")
    optimizer = torch.optim.Adam(domain_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=epoch_decay,gamma=lr_decay)
    for epoch in range(epochs):
        all_loss = 0.0
        total_train = len(train_loader)
        for i, (data, _, _, label) in enumerate(train_loader):
            print("\r {} / {}".format(i, total_train), flush=True, end='')
            data = data.to(device)
            label = label.long().to(device)
            fea = fea_net(data)
            out = domain_net(fea)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        scheduler.step()
        print('Epoch [{}/{}],Loss: {:.4f}'.format(epoch + 1, epochs, all_loss / total_train))


def train3(fea_net, domain_net, pred_net, train_loader, test_loader, epochs, criterion):
    print("==================step3: train domain_net and others===================")
    optimizer1 = torch.optim.Adam(fea_net.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(pred_net.parameters(), lr=lr)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,step_size=epoch_decay,gamma=lr_decay)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=epoch_decay,gamma=lr_decay)
    all_loss = 0.0
    val_loss = 0.0
    total_train = len(train_loader)
    total_test = len(test_loader)
    for epoch in range(epochs):
        for i, (data, pos_label_x, pos_label_y, domain_label) in enumerate(train_loader):
            print("\r {} / {}".format(i, total_train), flush=True, end='')
            data = data.to(device)
            pos_label_x = pos_label_x.long().to(device)
            pos_label_y = pos_label_y.long().to(device)
            domain_label = domain_label.long().to(device)
            fea = fea_net(data)
            domain_pred = domain_net(fea)
            pos_pred = pred_net(fea)

            pred_loss = criterion(pos_pred[:, :18], pos_label_x) + criterion(pos_pred[:, 18:], pos_label_y)
            domain_loss = criterion(domain_pred, domain_label)
            loss = pred_loss - 0.0001 * domain_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            all_loss += pred_loss.item()
        scheduler1.step()
        scheduler2.step()
        for data, label_x, label_y, _ in test_loader:
            data = data.to(device)
            label_x = label_x.long().to(device)
            label_y = label_y.long().to(device)
            fea = fea_net(data)
            pred = pred_net(fea)
            loss = criterion(pred[:, 0], label_x) + criterion(pred[:, 1], label_y)

            val_loss += loss.item()

        print('Epoch [{}/{}],Loss: {:.4f}, val_Loss: {:.4f}'.format(epoch + 1, epochs, all_loss / total_train,
                                                                    val_loss / total_test))
