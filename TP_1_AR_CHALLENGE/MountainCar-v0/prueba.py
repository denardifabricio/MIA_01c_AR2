import os

base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")

model_path = os.path.join(base_path,"q_network_mountaincar.pth")

print(f"Ruta del modelo: {model_path}")