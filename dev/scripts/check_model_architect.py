from ultralytics import YOLO

model = YOLO("yolo26s.pt")

print("=== ALL NAMED LAYERS ===")
for name, param in model.model.named_parameters():
    print(f"{name:60s} | shape={str(list(param.shape)):20s} | numel={param.numel()}")

print("\n=== UNIQUE TOP-LEVEL INDICES ===")
indices = set()
for name, _ in model.model.named_parameters():
    parts = name.split('.')
    if len(parts) > 1 and parts[1].isdigit():
        indices.add(int(parts[1]))
print(sorted(indices))

for i, module in enumerate(model.model.model):
    name = module.__class__.__name__
    has_params = sum(p.numel() for p in module.parameters())
    print(f"Layer {i:2d}: {name:30s} params={has_params:>10,}")