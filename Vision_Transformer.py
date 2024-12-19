import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Implementation of Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,in_chans,embed_dim,drop_prob=0.001):
        super().__init__()
        # Image Size is Square (16 by 16)
        # Patch Size is Square (4 by 4)
        # Number of Patches Inside of Image
        self.num_patches = (img_size//patch_size)*(img_size//patch_size)
        # The Embedding Dimension C X H X W
        self.embed_dim = embed_dim
        # Convolutional Layer which does both the
        # Splitting into Patches and their Embedding
        self.projection = nn.Conv2d(in_channels=in_chans,
                        out_channels=embed_dim,
                        kernel_size=patch_size,
                        stride=patch_size)
        
        # Generate Learnable CLS Token
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim),requires_grad=True)
        # print(self.cls_token.shape)
        self.pos_embed = nn.Parameter(torch.randn((1+self.num_patches),self.embed_dim),requires_grad=True)
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        print('1',x.device)
        # x is the Input Tensor which is a 4D Tensor
        # (batch_size, in_channels, height_of_image, width_of_image)
        (batch_size,C,H,W) = x.shape
        # Output Tensor after Convolution
        # Output Tensor is a 4D Tensor
        # (batch_size, embed_dim, img_size//patch_size, img_size//patch_size)
        x = self.projection(x)
        print('2',x.device)
        #print(x.shape)
        # Converting 4D Tensor to a 3D Tensor
        # Perform Flattening of Last Two Dimension
        # (batch_size, embed_dim, num_patches)
        # num_patches = (img_size//patch_size)*(img_size//patch_size)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        print('3',x.device)
        #print(x.shape)
        # Permute the 3D Tensor in Last Two Dimensions
        # (batch_size, num_patches, embed_dim)
        x = x.permute(0,2,1)
        print('4',x.device)
        # print(x.shape)
        # Repeat CLS Token in the batch dimension
        # It is a  3D Tensor
        # (batch_size, 1, embed_dim)
        cls_token = self.cls_token.repeat(batch_size,1,1)
        print('5',cls_token.device)
        # Concatinate the CLS Token with Patch Embedding
        # After Concatination we have a 3D Tensor
        # (batch_size, 1+num_patches, embed_dim)
        # 1+num_patches = sequence_length
        # (batch_size, sequence_length, embed_dim)
        y = torch.cat((cls_token,x),dim=1)
        print('6',y.device)
        # print(y.shape)
        # Generate Positional Encoding
        z = self.pos_embed
        print('7',z.device)
        # z is a 2D Tensor
        # print(z.shape)
        # print(z)
        # Adding Positional Encoding Information with Patch Embedding
        # y is a 3D Tensor and z is a 2D Tensor
        # Hence, (y + z ) is a Broadcasting Addition
        out = self.dropout(y + z)
        # print(out.shape)
        return out


# Implementation of Multi Head Attention
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_prob=0.001):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        self.q_layer = nn.Linear(embed_dim, embed_dim)
        self.k_layer = nn.Linear(embed_dim, embed_dim)
        self.v_layer = nn.Linear(embed_dim, embed_dim)
        self.linear_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        # x is a 3D Tensor
        # x is the input to the Transformer Encoder section
        batch_size, sequence_length, embed_dim = x.shape
        #print(f'Size of input tensor: {batch_size, sequence_length, embed_dim}')

        # Generation of q Tensor
        q = self.q_layer(x)
        # print(q.shape)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # print(q.shape)
        q = q.permute(0, 2, 1, 3)
        # Now the Shape is (batch_size, num_heads, sequence_length, head_dim)
        # print(q.shape)
        print('8',q.device)

        # Generation of k Tensor
        k = self.k_layer(x)
        # print(k.shape)
        k = k.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # print(k.shape)
        k = k.permute(0, 2, 1, 3)
        # Now the Shape is (batch_size, num_heads, sequence_length, head_dim)
        # print(k.shape)
        print('9',k.device)

        # Generation of v Tensor
        v = self.v_layer(x)
        # print(v.shape)
        v = v.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # print(v.shape)
        v = v.permute(0, 2, 1, 3)
        # Now the Shape is (batch_size, num_heads, sequence_length, head_dim)
        # print(v.shape)
        print('10',v.device)
        
        # Calculation of Scaled Dot Product Attention
        d_k = q.shape[-1]
        scaled_dot_product = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
        print('11',scaled_dot_product.device)
        # Shape of scaled_dot_product is (batch_size, num_heads, sequence_length, sequence_length)
        attention = F.softmax(scaled_dot_product, dim=-1)
        attention = self.dropout(attention)
        attention_head = torch.matmul(attention, v)
        # Shape of attention_head is (batch_size, num_heads, sequence_length, head_dim)
        print('12',attention_head.device)

        # Concatination of Multiple Heads
        attention_head = attention_head.permute(0, 2, 1, 3)
        # Now the Shape of Attention Head is (batch_size, sequence_length, num_heads, head_dim)
        # print(attention_head.shape)
        attention_head = attention_head.reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        # Now the Shape of Attention Head is (batch_size, sequence_length, embed_dim)
        # print(attention_head.shape)

        # Inter Communication between Multiple Heads
        z = self.linear_layer(attention_head)
        # Shape of z tensor will be (batch_size, sequence_length, embed_dim)
        # print(z.shape)
        z = self.dropout(z)
        print('13',z.device)
        return z


# Implementation of MLP
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden, drop_prob=0.001):
        super().__init__()
        self.linear1 = nn.Linear(in_features = embed_dim, out_features = hidden)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.linear2 = nn.Linear(in_features = hidden, out_features = embed_dim)
        self.dropout2 = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


# Implementation of Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 0.00001
        self.gamma = nn.Parameter(torch.ones(embed_dim),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(embed_dim),requires_grad=True)
    
    def forward(self, x):
        mean_values = (x.mean(dim=-1, keepdim=True))
        # print(mean_values.shape)
        variance_values = (x.var(dim=-1, unbiased=False, keepdim=True))
        # print(variance_values.shape)
        out = ((x-mean_values)/torch.sqrt(variance_values + self.eps))*self.gamma + self.beta
        return out


# Implementation of Single Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self,embed_dim, num_heads, hidden, drop_prob=0.001):
        super().__init__()
        self.ln1 = LayerNormalization(embed_dim=embed_dim)
        self.attention = MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ln2 = LayerNormalization(embed_dim=embed_dim)
        self.mlp = MLP(embed_dim=embed_dim,hidden=hidden)
        self.dropout2 = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        residual_x = x.clone()
        x = self.ln1(x)
        x = self.attention(x)
        x = self.dropout1(x + residual_x)
        residual_x = x.clone()
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout2(x + residual_x)
        return x


# Implementation of Stack of Encoder Layers
class SequentialEncoderLayersGeneration(nn.Sequential):
    def forward(self, x):
        for i in self._modules.values():
            x = i(x)
        return x


# Implementation of Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self,img_size,patch_size,in_chans,embed_dim,num_heads,hidden,num_layers,num_classes,drop_prob = 0.001):
        super().__init__()
        self.patchembed = PatchEmbedding(img_size,patch_size,in_chans,embed_dim,drop_prob=drop_prob)
        self.layers = SequentialEncoderLayersGeneration(*[EncoderLayer(embed_dim,num_heads,hidden,drop_prob=drop_prob) 
                                                          for i in range(num_layers)])
        self.ln = LayerNormalization(embed_dim)
        self.mlp_head = nn.Linear(in_features=embed_dim,out_features=num_classes)
        
    def forward(self, x):
        x = self.patchembed(x)
        x = self.layers(x)
        # We are selecting only the “classification token”
        # We have represented the “classification token” as CLS Token
        x = x[:,0,:]
        x = self.ln(x)
        x = self.mlp_head(x)
        return x


