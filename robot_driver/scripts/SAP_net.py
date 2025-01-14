import torch
from torch import nn

class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y

class Recurrent(nn.Module):
    def __init__(self, position_num, in_info_size, out_info_size, hidden_size) -> None:  #position_size = 2d_position_num * 2
        super().__init__()
        self.in_fc1_position = nn.Sequential(nn.Linear(position_num*2, 64),
                                    nn.ReLU() )
        
        self.in_fc2_info = nn.Sequential( nn.Linear(in_info_size, 32),
                                     nn.ReLU()  )


        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, num_layers=1, batch_first=True)  # input_size=8, so sequence length = 8 since the feature size from upstream is 64 
        self.info_lstm = nn.LSTM(input_size=3, hidden_size=8, num_layers=1, batch_first=True)
                
        self.attention_points_out_layer = nn.Sequential(
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, position_num*2),
                                             nn.Tanh()
                                             )

        self.predict_info_out_layer = nn.Sequential(
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, out_info_size),
                                            nn.Sigmoid() )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, position_2d, input_info):
        n = position_2d.shape[0]
        position_2d = position_2d.reshape(n, -1)  # (B, position_size)
        position_2d = self.in_fc1_position(position_2d).reshape(n, -1, 8)  #-> (B, 64) ->(B, 8, 8) namely (B, sequence_l, ele_feature_size)
        
        if input_info.shape[1] == 0:
            x = position_2d
        else:
            # input_info = self.in_fc2_info(input_info).reshape(n, -1, 8) #-> (B, 64) ->(B, 4, 8)
            # x, _ =  self.sigmoid(torch.concat((position_2d, input_info), dim=1)) # along sequence_length
            input_info, _ = self.info_lstm(input_info.reshape(n, -1, 3))
            x =  torch.concat((position_2d, input_info[:, -1:, :]), dim=1) # along sequence_length
        
        x, _ = self.lstm(x) #(B, sequence_length, hidden_size) with default h0, c0 (zeros)
        
        # x[:, -1, :] # (B, hidden_size), only focus on the last element hn
        new_att_points = self.attention_points_out_layer(x[:, -1, :]).reshape(n, -1, 2)   #->  (B, position_num, 2)
        
        new_info = self.predict_info_out_layer(x[:, -1, :]) #->  (B, info_size)

        return (new_att_points, new_info)
        
class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)  ## (256, 8, 3, 3, 1)
        image_coordinates = image_coordinates.unsqueeze(0)  ## (1, 3, 3, 2), (3,3)对应的是x，和y的坐标，乘以对应的softmax
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])  ## 这里是元素相乘，目标是每个元素的softmax值分别乘以其x、y坐标 延展： (256, 8, 3, 3, 1) => (256, 8, 3, 3, 2), (1, 3, 3, 2) => (1, 1, 3, 3, 2)，在2,3维度求和
        # (N, C, 2)
        return out


class Heatmap(nn.Module):
    def __init__(self, img_height, img_width, sigma = 5) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_width = img_width
        self.img_height = img_height
        self.sigma = sigma
        
        X1 = torch.linspace(1, self.img_width, self.img_width)
        Y1 = torch.linspace(1, self.img_height, self.img_height)
        [self.X, self.Y] = torch.meshgrid(X1, Y1, indexing='xy')
        self.X = self.X.to(self.device)
        self.Y = self.Y.to(self.device)  
        
    def forward(self, x):
        # x: B, n_p, 2
        # out: B, n_p, h, w
        # out = torch.zeros(x.shape[0], x.shape[1], self.img_height, self.img_width).to(self.device)
        out_batch = []
        for batch_idx in range(x.shape[0]):
            out_ = []
            for point_idx in range(x.shape[1]):
                p_x = x[batch_idx, point_idx, 0]
                p_y = x[batch_idx, point_idx, 1] 
                attend_x_pix = (p_x + 1) * (self.img_height - 1) / 2
                attend_y_pix = (p_y + 1) * (self.img_width - 1) / 2
                X_ = self.X - attend_x_pix
                Y_ = self.Y - attend_y_pix
                D2 = X_ * X_ + Y_ * Y_
                E2 = 2.0 * self.sigma * self.sigma 
                Exponent = D2 / E2
                out_.append(torch.exp(-Exponent))
                
                # out[batch_idx, point_idx, :, :] = torch.exp(-Exponent)
            out_batch.append(torch.stack(out_))
        return torch.stack(out_batch)

class SAP(nn.Module):
    
    '''
        :param out_info_size: 6 for wrench only, 3 for position only, 9 for both 
        :param in_info_size:  9 for wrench only, 6 for position only, 12 for both (since actions take 3) 
    '''
    def __init__(self, in_channels, encoder_out_channels, decoder_input_size, in_info_size, out_info_size, temperature=None):
        super().__init__()
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels[0], kernel_size=5, stride=2),
            nn.BatchNorm2d(encoder_out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[0], out_channels=encoder_out_channels[1], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[1], out_channels=encoder_out_channels[2], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[2]),
            nn.ReLU()
        ) ## -> (B, encoder_out_channels[2], out_H, out_W), (out_H, out_W)应与decoder_input_size相等
        
        self.feature_area_extractor_argmax = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encoder_out_channels[0], kernel_size=5, stride=2),
            nn.BatchNorm2d(encoder_out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[0], out_channels=encoder_out_channels[1], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_out_channels[1], out_channels=encoder_out_channels[2], kernel_size=5),
            nn.BatchNorm2d(encoder_out_channels[2]),
            nn.ReLU(),
            SpatialSoftArgmax(temperature=temperature, normalise=True)
        ) ## -> (B, encoder_out_channels[2], 2)
        
        self.recurrent = Recurrent(position_num=encoder_out_channels[2], in_info_size=in_info_size, out_info_size=out_info_size, hidden_size=256) ## -> (B, encoder_out_channels[2], 2) , (B, out_info_size)
        
        self.heatmap = Heatmap(decoder_input_size[0], decoder_input_size[1], sigma=20) ## -> (B, encoder_out_channels[2], out_H, out_W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_channels[2], 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=6, stride=2),
            nn.Tanh()
        )

    def forward(self, i, a): # (256, 1, 28, 28), (256, 6)
        att_points_pre = self.feature_area_extractor_argmax(i)
        att_points, a = self.recurrent(att_points_pre, a)   # (256, 1, 28, 28), (256, 6) -> (256, 8, 2), (256, 6)
        x = self.image_feature_extractor(i) * self.heatmap(att_points)# (256, 1, 28, 28) -> (256, 8, 16, 16)
        return (att_points, self.decoder(x), a)  # ((256, 8, 16, 16) -> (256, 1, 28, 28)  ,  (256, 6))


class SAP_loss(object):

    def __init__(self, alpha = 1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.alpha = alpha
        
    def __call__(self, reconstructed, target, a_hat, a, att_points_plus1, att_points):
        """
        Performs the loss computation, and returns both loss components.
        :param reconstructed: Reconstructed, grayscale image
        :param target: Target, grayscale image
        :param a_hat: Predicted info
        :param a: target into
        :param ft: Features produced by the encoder for the target image
        :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
        :return: Loss
        """ 
        gi = self.mse_loss(reconstructed, target)/len(reconstructed)
        
        # ga = 3000 / gi.detach() * self.mse_loss(a_hat, a)/len(reconstructed)
        ga = 100 * self.mse_loss(a_hat, a)/len(reconstructed)
        gf = self.mse_loss(att_points_plus1, att_points)/len(reconstructed)
        return gi + ga + self.alpha*gf, gi, ga, gf
        

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sap = SAP(in_channels=1, encoder_out_channels=(4, 8, 8), decoder_input_size=(90,90), in_info_size=8, out_info_size=6).to(device)
    images = torch.rand((256,1,200,200)).to(device)
    input_info = torch.rand((256,8)).to(device)
    out_key_points, out_images, out_info = sap(images, input_info)
    print(out_key_points.shape)
    print(out_images.shape)
    print(out_info.shape)
 
