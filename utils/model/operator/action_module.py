import torch.nn as nn

class ActionEncoder(nn.Module):

    def __init__(
        self,
        categorization_num=12,
        latent_dim=768,
    ) -> None:

        super().__init__()

        self.categorization_num = categorization_num
        self.latent_dim = latent_dim

        self.embeddings = nn.Embedding(categorization_num, latent_dim)

    def forward(self, action_id):
        return self.embeddings(action_id)