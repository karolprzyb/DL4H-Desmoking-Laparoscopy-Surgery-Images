import torch

# DEFINE THE PAPER DISCRIMINATOR
class Discriminator(torch.nn.Module):
    def __init__(self, input_channels:int = 3):
        super().__init__()

        sequence = [torch.nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), 
                    torch.nn.LeakyReLU(0.2, True)]
        
        sequence += [torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
                     torch.nn.BatchNorm2d(128),
                     torch.nn.LeakyReLU(0.2, True)]
        
        sequence += [torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
                     torch.nn.BatchNorm2d(256),
                     torch.nn.LeakyReLU(0.2, True)]
        
        sequence += [torch.nn.ZeroPad2d(2)]
        
        sequence += [torch.nn.Conv2d(256, 512, kernel_size=4, stride=1), 
                     torch.nn.BatchNorm2d(512),
                     torch.nn.LeakyReLU(0.2, True)]
        
        sequence += [torch.nn.ZeroPad2d(2)]
        
        sequence += [torch.nn.Conv2d(512, 1, kernel_size=4, stride=1)]
       
       # sequence += [torch.nn.Sigmoid()] #I think this was not used in the paper since L1 loss was used. Experiments with this and BCE included in final paper
        
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)





# DEFINE THE PAPER UNET
class UNET(torch.nn.Module):
    def __init__(self, input_channels:int = 3):
        super().__init__()

        self.drp = torch.nn.Dropout(0.5) # We will apply this in forward. Technically this could be in the list below instead too.

        # Note that all convolutions with a BatchNorm after them have bias false as BatchNorm contains a bias term itself.
        # It would therefore be redundant and add nothing.

        sequence = [torch.nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),                # 0         -> 42
                    torch.nn.LeakyReLU(0.2, True)]                                                          # 1
        
        sequence += [torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),              # 2
                     torch.nn.BatchNorm2d(128),                                                             # 3         -> 39
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 4
        sequence += [torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),             # 5
                     torch.nn.BatchNorm2d(256),                                                             # 6         -> 36
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 7
        sequence += [torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 8
                     torch.nn.BatchNorm2d(512),                                                             # 9         -> 33
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 10
        
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 11
                     torch.nn.BatchNorm2d(512),                                                             # 12        -> 30
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 13
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 14
                     torch.nn.BatchNorm2d(512),                                                             # 15        -> 27
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 16
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 17
                     torch.nn.BatchNorm2d(512),                                                             # 18        -> 24
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 19
        
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),                         # 20
                     torch.nn.ReLU(True)]                                                                   # 21
        sequence += [torch.nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),    # 22
                     torch.nn.BatchNorm2d(512),                                                             # 23
                     torch.nn.ReLU(True)]                                                                   # 24        <- 18
        
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 25
                     torch.nn.BatchNorm2d(512),                                                             # 26
                     torch.nn.ReLU(True)]                                                                   # 27        <- 15
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 28
                     torch.nn.BatchNorm2d(512),                                                             # 29
                     torch.nn.ReLU(True)]                                                                   # 30        <- 12
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 31
                     torch.nn.BatchNorm2d(512),                                                             # 32
                     torch.nn.ReLU(True)]                                                                   # 33        <- 9
        
        sequence += [torch.nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),   # 34
                     torch.nn.BatchNorm2d(256),                                                             # 35
                     torch.nn.ReLU(True)]                                                                   # 36        <- 6
        sequence += [torch.nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),    # 37
                     torch.nn.BatchNorm2d(128),                                                             # 38
                     torch.nn.ReLU(True)]                                                                   # 39        <- 3
        sequence += [torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),    # 40
                     torch.nn.BatchNorm2d(64),                                                              # 41
                     torch.nn.ReLU(True)]                                                                   # 42        <- 0
        
        sequence += [torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),                  # 43
                     torch.nn.Tanh()]                                                                       # 44
        
        for x in range(0, 20):
            if type(sequence[x]) == torch.nn.modules.conv.Conv2d:
                torch.nn.init.kaiming_uniform_(sequence[x].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        
        for x in range(20, len(sequence)):
            if type(sequence[x]) == torch.nn.modules.conv.Conv2d or type(sequence[x]) == torch.nn.modules.conv.ConvTranspose2d:
                torch.nn.init.kaiming_uniform_(sequence[x].weight, mode='fan_in', nonlinearity='relu')
        
        for x in range(0, len(sequence)):
            if type(sequence[x]) == torch.nn.modules.batchnorm.BatchNorm2d:
                torch.nn.init.constant_(sequence[x].bias.data, 0.0)
                torch.nn.init.normal_(sequence[x].weight.data, 1.0, 0.02)

        self.sequence = torch.nn.ParameterList(sequence)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x0 = x = self.sequence[0](x)
        self.sequence[1](x)
        x = self.sequence[2](x)
        x3 = x = self.sequence[3](x)
        self.sequence[4](x)
        x = self.sequence[5](x)
        x6 = x = self.sequence[6](x)
        self.sequence[7](x)
        x = self.sequence[8](x)
        x9 = x = self.sequence[9](x)
        self.sequence[10](x)
        x = self.sequence[11](x)
        x12 = x = self.sequence[12](x)
        self.sequence[13](x)
        x = self.sequence[14](x)
        x15 = x = self.sequence[15](x)
        self.sequence[16](x)
        x = self.sequence[17](x)
        x18 = x = self.sequence[18](x)
        self.sequence[19](x)
        x = self.sequence[20](x)
        self.sequence[21](x)
        x = self.sequence[22](self.drp(x))
        x = self.sequence[23](x)
        x = torch.cat((x, x18), 1)
        self.sequence[24](x)
        x = self.sequence[25](x)
        x = self.sequence[26](x)
        x = torch.cat((x, x15), 1)
        self.sequence[27](x)
        x = self.sequence[28](x)
        x = self.sequence[29](x)
        x = torch.cat((x, x12), 1)
        self.sequence[30](x)
        x = self.sequence[31](x)
        x = self.sequence[32](x)
        x = torch.cat((x, x9), 1)
        self.sequence[33](x)
        x = self.sequence[34](x)
        x = self.sequence[35](x)
        x = torch.cat((x, x6), 1)
        self.sequence[36](x)
        x = self.sequence[37](x)
        x = self.sequence[38](x)
        x = torch.cat((x, x3), 1)
        self.sequence[39](x)
        x = self.sequence[40](x)
        x = self.sequence[41](x)
        x = torch.cat((x, x0), 1)
        self.sequence[42](x)
        x = self.sequence[43](x)
        return self.sequence[44](x)





# DEFINE THE ABLATED UNET
class UNETsmolr(torch.nn.Module):
    def __init__(self, input_channels:int = 3):
        super().__init__()

        self.drp = torch.nn.Dropout(0.5) # We will apply this in forward. Technically this could be in the list below instead too.

        # Note that instead of changing the number of entries, I substituted the parts of the net I planned to ablate with None
        # and removed them from forward. This means I don't have to bother as much with changing indices.

        # Note that all convolutions with a BatchNorm after them have bias false as BatchNorm contains a bias term itself.
        # It would therefore be redundant and add nothing.

        sequence = [torch.nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),                # 0         -> 42
                    torch.nn.LeakyReLU(0.2, True)]                                                          # 1
        
        sequence += [torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),              # 2
                     torch.nn.BatchNorm2d(128),                                                             # 3         -> 39
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 4
        sequence += [torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),             # 5
                     torch.nn.BatchNorm2d(256),                                                             # 6         -> 36
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 7
        sequence += [torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 8
                     torch.nn.BatchNorm2d(512),                                                             # 9         -> 33
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 10
        
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 11
                     torch.nn.BatchNorm2d(512),                                                             # 12        -> 30
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 13
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 14
                     torch.nn.BatchNorm2d(512),                                                             # 15        -> 27
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 16
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),             # 17
                     torch.nn.BatchNorm2d(512),                                                             # 18        -> 24
                     torch.nn.LeakyReLU(0.2, True)]                                                         # 19
        
        sequence += [torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),                         # 20
                     torch.nn.ReLU(True)]                                                                   # 21
        sequence += [torch.nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),    # 22
                     torch.nn.BatchNorm2d(512),                                                             # 23
                     torch.nn.ReLU(True)]                                                                   # 24        <- 18
        
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 25
                     torch.nn.BatchNorm2d(512),                                                             # 26
                     torch.nn.ReLU(True)]                                                                   # 27        <- 15
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 28
                     torch.nn.BatchNorm2d(512),                                                             # 29
                     torch.nn.ReLU(True)]                                                                   # 30        <- 12
        sequence += [torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),   # 31
                     torch.nn.BatchNorm2d(512),                                                             # 32
                     torch.nn.ReLU(True)]                                                                   # 33        <- 9
        
        sequence += [torch.nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),   # 34
                     torch.nn.BatchNorm2d(256),                                                             # 35
                     torch.nn.ReLU(True)]                                                                   # 36        <- 6
        sequence += [torch.nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),    # 37
                     torch.nn.BatchNorm2d(128),                                                             # 38
                     torch.nn.ReLU(True)]                                                                   # 39        <- 3
        sequence += [torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),    # 40
                     torch.nn.BatchNorm2d(64),                                                              # 41
                     torch.nn.ReLU(True)]                                                                   # 42        <- 0
        
        sequence += [torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),                  # 43
                     torch.nn.Tanh()]                                                                       # 44
        
        # Initilizing here. Initializing the generator should have impact, so making sure to do that.

        for x in range(0, 20):
            if type(sequence[x]) == torch.nn.modules.conv.Conv2d:
                torch.nn.init.kaiming_uniform_(sequence[x].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        
        for x in range(20, len(sequence)):
            if type(sequence[x]) == torch.nn.modules.conv.Conv2d or type(sequence[x]) == torch.nn.modules.conv.ConvTranspose2d:
                torch.nn.init.kaiming_uniform_(sequence[x].weight, mode='fan_in', nonlinearity='relu')
        
        for x in range(0, len(sequence)):
            if type(sequence[x]) == torch.nn.modules.batchnorm.BatchNorm2d:
                torch.nn.init.constant_(sequence[x].bias.data, 0.0)
                torch.nn.init.normal_(sequence[x].weight.data, 1.0, 0.02)

        self.sequence = torch.nn.ParameterList(sequence)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x0 = x = self.sequence[0](x)
        self.sequence[1](x)
        x = self.sequence[2](x)
        x3 = x = self.sequence[3](x)
        self.sequence[4](x)
        x = self.sequence[5](x)
        x6 = x = self.sequence[6](x)
        self.sequence[7](x)
        x = self.sequence[8](x)
        x9 = x = self.sequence[9](x)
        self.sequence[10](x)
        x = self.sequence[11](x)
        x12 = x = self.sequence[12](x)
        self.sequence[13](x)
        # x = self.sequence[14](x)
        # x15 = x = self.sequence[15](x)
        # self.sequence[16](x)
        # x = self.sequence[17](x)
        # x18 = x = self.sequence[18](x)
        # self.sequence[19](x)
        x = self.sequence[20](x)
        self.sequence[21](x)
        x = self.sequence[22](self.drp(x))
        x = self.sequence[23](x)
        x = torch.cat((x, x12), 1)              # MODIFIED TO USE x12 instead of the no longer existing x18
        self.sequence[24](x)
        # x = self.sequence[25](x)
        # x = self.sequence[26](x)
        # x = torch.cat((x, x15), 1)
        # x = self.sequence[27](x)
        # x = self.sequence[28](self.drop(x))
        # x = self.sequence[29](x)
        # x = torch.cat((x, x12), 1)
        # self.sequence[30](x)
        x = self.sequence[31](x)
        x = self.sequence[32](x)
        x = torch.cat((x, x9), 1)
        self.sequence[33](x)
        x = self.sequence[34](x)
        x = self.sequence[35](x)
        x = torch.cat((x, x6), 1)
        self.sequence[36](x)
        x = self.sequence[37](x)
        x = self.sequence[38](x)
        x = torch.cat((x, x3), 1)
        self.sequence[39](x)
        x = self.sequence[40](x)
        x = self.sequence[41](x)
        x = torch.cat((x, x0), 1)
        self.sequence[42](x)
        x = self.sequence[43](x)
        return self.sequence[44](x)