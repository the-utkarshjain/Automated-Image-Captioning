import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions.categorical as categorical

import torchvision.models as models

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    img_embedding_size = config_data['model']['img_embedding_size']
    model_type = config_data['model']['model_type']
    
    if model_type not in ["LSTM", "RNN", "Architecture2"]:
        raise NotImplementedError("Requested model_type not supported")
        
    if model_type == "LSTM":
        return Decoder(hidden_size, embedding_size, img_embedding_size, vocab, "LSTM")
    
    if model_type == "RNN":
        return Decoder(hidden_size, embedding_size, img_embedding_size, vocab, "RNN")
    
    if model_type == "Architecture2":
        return Decoder(hidden_size, embedding_size, img_embedding_size, vocab, "Architecture2")
        


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Downloads the pretrained ResNet50 model
        self.ResNet = models.resnet50(pretrained = True)
        
        # Freezing the model parameters to prevent retraining
        for parameters in self.ResNet.parameters():
            parameters.requires_grad = False
        
        # Number of in_features in the last fc layer
        self.in_features = self.ResNet.fc.in_features
        
        # Creating a sequential model from all the layers of ResNet except the last one
        self.layers = list(self.ResNet.children())[:-1]
        self.frozen_encoder = nn.Sequential(*self.layers)
    
    def forward(self, image):
        self.frozen_encoder.eval()
        
        with torch.no_grad():
            encoded_image = self.frozen_encoder(image)
        
        return encoded_image


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, img_embedding_size, vocab, model_type, num_layers = 2):
        """
        hidden_size: the hidden_size for LSTM
        embedding_size: the embedding_size for mapping the words
        num_layers: number of layers to use for LSTM
        
        """
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.img_embedding_size = img_embedding_size
        self.vocab = vocab
        self.num_layers = num_layers
        self.model_type = model_type
        
        if self.model_type == "LSTM" or self.model_type == "RNN":
            assert self.embedding_size == self.img_embedding_size, "embedding_size should be equal to img_embedding_size for LSTM and RNN experiments"
        
        self.ResnetEncoder = ResNetEncoder()
        self.linear1 = nn.Linear(self.ResnetEncoder.in_features, self.img_embedding_size)
        
        self.embedding = nn.Embedding(len(self.vocab), self.embedding_size)
        
        if self.model_type == "LSTM":
            self.RecurrentUnit = nn.LSTM(self.embedding_size, self.hidden_size, batch_first = True, num_layers = self.num_layers)
            
        if self.model_type == "RNN":
            self.RecurrentUnit = nn.RNN(self.embedding_size, self.hidden_size, batch_first = True, num_layers = self.num_layers, nonlinearity = 'relu')
            
        if  self.model_type == "Architecture2":
            self.RecurrentUnit = nn.LSTM(self.embedding_size + self.img_embedding_size, self.hidden_size, batch_first = True, num_layers = self.num_layers)
            
        self.linear2 = nn.Linear(self.hidden_size, len(self.vocab))
        
    def forward(self, image, token_ids, device, debug = False):
        """
        image : [N, C, H, W]
        token_ids: [N, seq_len]
        debug: if True, prints the shape of the input as it passes through the network
        """
        if self.model_type == "Architecture2":
            padding_tokens = [self.vocab("<pad>")]*token_ids.shape[0]
            padding_tokens = torch.tensor(padding_tokens).to(device, dtype = torch.long)
            padding_tokens = padding_tokens.unsqueeze(1)
            token_ids = torch.cat([padding_tokens, token_ids], dim =-1)
        
        if debug: print(f"Input image shape: {image.shape}, Tokens ids shape: {token_ids.shape}")
                        
        encoded_image = self.ResnetEncoder(image).squeeze(-1).squeeze(-1) # [N, self.ResnetEncoder.in_features]
        if debug: print(f"Original image: {image.shape}\nEncoded image: {encoded_image.shape}")
            
        encoded_image = self.linear1(encoded_image).unsqueeze(1) # [N, 1, img_embedding_size]
        if debug: print("Encoded image after first linear layer: ", encoded_image.shape)
        
        embeddings = self.embedding(token_ids) # [N, seq_len, embedding_size] or [N, 1 + seq_len, embedding_size]
        if debug: print(f"\nOriginal tokens: {token_ids.shape} \nAfter embedding layer: {embeddings.shape}")
        
        if self.model_type == "LSTM" or self.model_type == "RNN":
            embeddings = torch.cat([encoded_image, embeddings], dim = 1) # [N, 1 + seq_len, embedding_size]
            if debug: print("Concatenation of encoded image and word embeddings: ", embeddings.shape)
        
        if self.model_type == "Architecture2":
            encoded_image = encoded_image.expand(-1, token_ids.shape[1], -1)
            embeddings = torch.cat([encoded_image, embeddings], dim = -1) # [N, 1 + seq_len, embedding_size + img_embedding_size]
            if debug: print("Concatenation of encoded image and word embeddings: ", embeddings.shape)
        
        if self.model_type == "LSTM" or self.model_type == "Architecture2":
            output, (_, _) = self.RecurrentUnit(embeddings) #[ N, 1 + seq_len, hidden_size]
            if debug: print("\nOutput from LSTM: ", output.shape)
        
        if self.model_type == "RNN":
            output, _ = self.RecurrentUnit(embeddings) #[ N, 1 + seq_len, hidden_size]
            if debug: print("\nOutput from RNN: ", output.shape)

        output = self.linear2(output) # [N, 1 + seq_len, vocab_size]
        if debug: print("\nOutput after final linear layer: ", output.shape)
            
        output = output[:,:-1,:] # [N, seq_len, vocab_size]
        output = output.permute(0,2,1) # [N, vocab_size, seq_len]

        return output
    
    def generate_tokens(self, image, generation, device):
        """
        Function to generate tokens for the images
        
        image: [N, C, H, W]
        generation: dictionary containing information about generation strategy
        device: {'cuda', 'cpu'}
        """
        
        max_len = generation["max_length"]
        deterministic = generation["deterministic"]
        temperature = generation["temperature"]
        
        encoded_image = self.ResnetEncoder(image).squeeze(-1).squeeze(-1) # [N, self.ResnetEncoder.in_features]
        encoded_image = self.linear1(encoded_image).unsqueeze(1) # [N, 1, img_embedding_size]
        
        generated_word_tokens = torch.ones(encoded_image.shape[0], max_len, dtype = torch.long, device = device) * -1
        
        if self.model_type == "LSTM" or self.model_type == "RNN":
            input_word = encoded_image
        
        if self.model_type == "Architecture2":
            padding_tokens = [self.vocab("<pad>")]*encoded_image.shape[0]
            padding_tokens = torch.tensor(padding_tokens).to(device, dtype = torch.long)
            padding_tokens = padding_tokens.unsqueeze(1)
            input_word = self.embedding(padding_tokens)
        
        # initializing the hidden and cell state for LSTM
        hidden = torch.zeros(self.num_layers, encoded_image.shape[0], self.hidden_size).to(device, dtype = torch.float)
        cell = torch.zeros(self.num_layers, encoded_image.shape[0], self.hidden_size).to(device, dtype = torch.float)
        
        for i in range(max_len):
            if self.model_type == "LSTM":
                output, (hidden, cell) = self.RecurrentUnit(input_word, (hidden, cell))
            if self.model_type == "RNN":
                output, hidden = self.RecurrentUnit(input_word, hidden)
            if self.model_type == "Architecture2":
                input_word = torch.cat([encoded_image, input_word], dim = -1)
                output, (hidden, cell) = self.RecurrentUnit(input_word, (hidden, cell))
                
            output = self.linear2(output.squeeze(1))
            
            if deterministic:
                predictions = F.softmax(output, dim = -1)
                predictions = torch.argmax(predictions, dim = -1)
                generated_word_tokens[:, i] = predictions.to(dtype = torch.long)
            else:
                predictions = F.softmax(output/temperature, dim = -1)
                predictions = categorical.Categorical(predictions).sample()
                generated_word_tokens[:, i] = predictions.to(dtype = torch.long)
             
            input_word = self.embedding(generated_word_tokens[:, i].unsqueeze(1))
        
        return generated_word_tokens
    
    def generate_captions(self, image, generation, device):
        """
        Function to generate clean captions (without the <start>, <end>, <pad> tokens) for the images
        
        image: [N, C, H, W]
        generation: dictionary containing information about generation strategy
        device: {'cuda', 'cpu'}
        """
        
        generated_word_tokens = self.generate_tokens(image, generation, device).tolist()   
        generated_captions = []
        
        for tokens in generated_word_tokens:
            temporary = []
            
            for token in tokens:
                if token == self.vocab("<end>"):
                    generated_captions.append(temporary)
                    break;
                
                if token in [self.vocab("<start>"), self.vocab("<pad>")]:
                    continue
                
                temporary.append(self.vocab.idx2word[token])
        
        return generated_captions