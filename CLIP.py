import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel

# ---- 图像编码器 ----
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, images):
        # images: [B, 3, H, W]
        x = self.cnn(images)  # [B, embed_dim]
        x = F.normalize(x, dim=-1)  # L2 normalize
        return x

# ---- 文本编码器 ----
class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        tokens = {k:v.to(next(self.parameters()).device) for k,v in tokens.items()}
        x = self.bert(**tokens).pooler_output  # [B, hidden_dim]
        x = self.proj(x)  # [B, embed_dim]
        x = F.normalize(x, dim=-1)
        return x

# ---- CLIP 模型 ----
class CLIP(nn.Module):
    def __init__(self, embed_dim=512, temperature=0.07):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim)
        self.txt_encoder = TextEncoder(embed_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, images, texts):
        img_emb = self.img_encoder(images)
        txt_emb = self.txt_encoder(texts)

        # 余弦相似度
        logits = img_emb @ txt_emb.t() / self.temperature
        labels = torch.arange(len(images), device=logits.device)
        # 双向对比损失
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.t(), labels)
        loss = (loss_img + loss_txt) / 2
        return loss, logits

# ---- 测试 ----
if __name__ == "__main__":
    model = CLIP()
    images = torch.randn(4,3,224,224)
    texts = ["a cat", "a dog", "a car", "a tree"]
    loss, logits = model(images, texts)
    print("Loss:", loss.item())
    print("Logits shape:", logits.shape)  # [B, B]
