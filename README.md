# AM simulation tool based on transformers (ViT & ViViT)

## Notes: 
* The sliced geometry is encoded by spatil- and temporal- transformer 
* A MLP is used to predict the temperature at (x, y, z, t) inspired by NeRF 

## Usage:
```python
model = ViViT()
x = torch.randn(4, 16, 3, 224, 224)     # input geometry 
y = torch.randn(4, 64)
out = model(x, y)
```