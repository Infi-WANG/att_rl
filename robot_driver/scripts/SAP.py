import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from SAP_dataloader import SAP_DataSet
from data_recorder import DataRecorded
import sys, os
sys.path.append(os.path.abspath(os.curdir))  ## only for vscode debug
from SAP_net import SAP, SAP_loss
from PIL import Image, ImageDraw

TRAIN_MODE = True
VAL_FILE_IDX = 80

def draw_spatial_features(numpy_image, features, image_size=(28, 28)):
    image_size_x, image_size_y = image_size
    img = Image.fromarray((numpy_image*255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for sp in features:
        x, y = sp
        attend_x_pix = int((x + 1) * (image_size_x - 1) / 2)
        attend_y_pix = int((y + 1) * (image_size_y - 1) / 2)
               
        attend_y_pix = max(0, attend_y_pix)
        attend_x_pix = max(0, attend_x_pix)
        attend_y_pix = min(attend_y_pix, image_size_y-1)
        attend_x_pix = min(attend_x_pix, image_size_x-1)
        
        # numpy_image[attend_y_pix, attend_x_pix] = np.array([0.0, 0.0, 1.0])
        draw.ellipse((attend_x_pix-1.5, attend_y_pix-1.5, attend_x_pix+1.5, attend_y_pix+1.5), fill=(255,255,0))
    return (np.array(img)/255).astype(np.float32)


def draw_figure(filename, num_images_to_draw, spatial_features_to_draw, images_to_draw, reconstructed_images_to_draw):
    f, axarr = plt.subplots(num_images_to_draw, 2, figsize=(10, 15), dpi=100)
    plt.tight_layout()
    for idx, im in enumerate(reconstructed_images_to_draw[:num_images_to_draw]):
        # original image
        og_image = (images_to_draw[:num_images_to_draw][idx] + 1) / 2
        og_im_res = np.repeat(og_image.numpy().reshape(200, 200, 1), 3, axis=2)
        img =  draw_spatial_features(og_im_res, spatial_features_to_draw[idx], image_size=(200, 200))
        # axarr[idx, 0].imshow(og_im_res)
        axarr[idx, 0].imshow(img)
        # reconstructed image
        scaled_image = (im + 1) / 2
        axarr[idx, 1].imshow(scaled_image.numpy().reshape(200, 200), cmap="gray")

    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--file_name", type=str, default=os.path.join(os.path.abspath("robot_driver/results"), "out_covT_heatmap.png"))
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--weight_path", type=str, default="/home/infi/att_ws/src/attention2/robot_driver/weight")
    args = parser.parse_args()

    # parameters and miscellaneous
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    # Adam learning rate
    lr = args.learning_rate
    out_file_name = args.file_name
    mode = args.mode
    weight_path = args.weight_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_dataset = SAP_DataSet("data_20240729_0928_L1097.pkl", i_use_action=True, i_use_wrench=False, i_use_position=False, o_use_wrench=False, o_use_position=True)
    test_dataset = SAP_DataSet("data_20240729_0929_L256.pkl", i_use_action=True, i_use_wrench=False, i_use_position=False, o_use_wrench=False, o_use_position=True)
    
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=2, shuffle=True)

    sap_model = SAP(in_channels=1, encoder_out_channels=(4, 8, 8), decoder_input_size=(90,90), in_info_size=15, out_info_size=3).to(device)

    optimiser = torch.optim.Adam(sap_model.parameters(), lr=lr)

    sap_loss = SAP_loss(alpha = 0.01)
    
    if TRAIN_MODE:
        for epoch in range(num_epochs):
            sap_model.train()
            for batch_idx, (image, info, image_plus1, info_plus1) in enumerate(train_loader):
                image = image.to(device)
                info = info.to(device)
                image_plus1 = image_plus1.to(device)
                info_plus1 = info_plus1.to(device)
                
                optimiser.zero_grad()
                key_points, image_predict, info_predict = sap_model(image, info)
                key_points_predict = sap_model.recurrent(sap_model.feature_area_extractor_argmax(image_plus1), info)[0].detach()
                
                loss, gi, info_loss, gf = sap_loss(image_predict, image, info_predict, info_plus1, key_points_predict, key_points)
                loss.backward()

                optimiser.step()
                if batch_idx % 5 == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\gi: {:.6f}, info loss: {:.6f}, gf: {:.6f}'.format(
                            epoch, batch_idx * len(image), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), gi.item(), info_loss.item(), gf.item()
                        )
                    )
            if epoch % 10 == 0 and epoch != 0:
                torch.save(sap_model.state_dict(),weight_path + "/weight_" + str(epoch) + ".plk")
                print('保存成功')
    else:

        sap_model.eval()
        sap_model.load_state_dict(torch.load(weight_path + "/weight_" + str(VAL_FILE_IDX) + ".plk"))
        with torch.no_grad():
            image, info, image_plus1, info_plus1 = next(iter(test_loader))
            image = image.to(device)
            info = info.to(device)
            image_plus1 = image_plus1.to(device)
            info_plus1 = info_plus1.to(device)
            
            key_points, image_predict, info_predict = sap_model(image, info)
            key_points_predict = sap_model.recurrent(sap_model.feature_area_extractor_argmax(image_plus1), info)[0].detach()
            
            loss, gi, ga, gf = sap_loss(image_predict, image, info_predict, info_plus1, key_points_predict, key_points)
            print(f"loss: {loss}, gi: {gi}, ga: {ga}, gf: {gf}")
            
            loss2 = torch.nn.MSELoss(reduction="sum")
            # loss = loss2(info[:, 3:6], info_plus1)/info.shape[0]
            # print(f"MSE from info to next info: {loss}")
            loss = loss2(info_predict, info_plus1)/info.shape[0]
            print(f"MSE from predicted info to next info: {loss}")
            
            num_images = 10
            draw_figure(out_file_name, num_images, key_points.to("cpu"), image.to("cpu"), image_predict.to("cpu"))
