from module import * 

if __name__ == '__main__':
    model = ViViT()

    x = torch.randn(4, 16, 3, 224, 224)
    y = torch.randn(4, 64)

    out = model(x, y)

    print(out.shape)

