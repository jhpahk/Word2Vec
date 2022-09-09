import torch
import torch.nn as nn

EMBED_DIM = 300
EMBED_MAX_NORM = 1


class Word2Vec(nn.Module):
    def __init__(self, vsize, type="SkipGram") -> None:
        super().__init__()
        
        self.type = type
        
        self.embedding = nn.Embedding(
            num_embeddings=vsize,
            embedding_dim=EMBED_DIM,
            max_norm=EMBED_MAX_NORM
        )
        
        self.linear = nn.Linear(in_features=EMBED_DIM, out_features=vsize)
    
    def forward_CBOW(self, words):
        embeds = self.embedding(words)
        mean_embeds = embeds.mean(axis=1)
        out = self.linear(mean_embeds)
        
        return out
        
    def forward_SkipGram(self, word):
        embed = self.embedding(word)
        out = self.linear(embed)
        
        return out
    
    def forward(self, words):
        if self.type == "SkipGram":
            return self.forward_SkipGram(words)
        else:
            return self.forward_CBOW(words)
        

