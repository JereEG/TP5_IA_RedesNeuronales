import os
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn

# Usaremos solo algunos campos numéricos para el modelo actual
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

# Otros campos de entrada (no usados por ahora en el modelo, pero pedidos en la interfaz)
OTROS_CAMPOS = ["Name", "Ticket", "Cabin", "Embarked", "Boat", "Body"]


class MLP(nn.Module):
    def __init__(self, n_inputs, capas, activacion):
        super().__init__()
        capas_completas = []
        for c in capas:
            capas_completas.append(nn.Linear(n_inputs, c))
            capas_completas.append(activacion())
            n_inputs = c
        capas_completas.append(nn.Linear(n_inputs, 1))
        capas_completas.append(nn.Sigmoid())
        self.net = nn.Sequential(*capas_completas)

    def forward(self, x):
        return self.net(x)


def predecir_sobrevive(entrada_dict, modelo_path):
    datos = []
    for f in FEATURES:
        val = entrada_dict[f]
        if val is None or val == "":
            raise ValueError(f"El campo '{f}' no puede estar vacío.")
        if f == "Sex":
            val = 1.0 if val.lower() == "male" else 0.0
        else:
            val = float(val)
        datos.append(val)

    modelo = MLP(len(FEATURES), [128], nn.ReLU())
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu"))
    modelo.eval()

    with torch.no_grad():
        entrada = torch.tensor([datos], dtype=torch.float32)
        salida = modelo(entrada).item()
        return (salida >= 0.5), salida


class Interfaz:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Predicción de supervivencia Titanic")
        self.frame = ttk.Frame(self.root, padding=20)
        self.frame.pack()

        self.vars = {}
        campos = FEATURES + OTROS_CAMPOS
        for f in campos:
            ttk.Label(self.frame, text=f).pack()
            if f == "Sex":
                self.vars[f] = tk.StringVar(value="male")
                entrada = ttk.OptionMenu(
                    self.frame, self.vars[f], "male", "male", "female")
            elif f == "Embarked":
                self.vars[f] = tk.StringVar(value="S")
                entrada = ttk.OptionMenu(
                    self.frame, self.vars[f], "S", "S", "C", "Q")
            else:
                self.vars[f] = tk.StringVar()
                entrada = ttk.Entry(self.frame, textvariable=self.vars[f])
            entrada.pack()

        # Modelo
        ttk.Label(self.frame, text="Modelo").pack()
        self.modelo_var = tk.StringVar()
        self.modelo = ttk.Combobox(
            self.frame, textvariable=self.modelo_var, state="readonly")
        self.modelo.pack()

        # Botones
        ttk.Button(self.frame, text="Predecir",
                   command=self.predecir).pack(pady=5)
        ttk.Button(self.frame, text="Limpiar campos",
                   command=self.limpiar_campos).pack(pady=5)

        # Resultado
        self.resultado = ttk.Label(self.frame, text="", font=("Arial", 14))
        self.resultado.pack(pady=10)

        self.barra = ttk.Progressbar(
            self.frame, orient='horizontal', length=200, mode='determinate')
        self.barra.pack()

        self.root.after(100, self.cargar_modelos)
        self.root.mainloop()

    def cargar_modelos(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        archivos = os.listdir("models")
        modelos = [f for f in archivos if f.endswith(".pt")]
        self.modelo["values"] = modelos
        if modelos:
            self.modelo.set(modelos[0])

    def predecir(self):
        try:
            entrada_dict = {f: self.vars[f].get()
                            for f in FEATURES + OTROS_CAMPOS}
            modelo_path = os.path.join("models", self.modelo.get())
            resultado, prob = predecir_sobrevive(entrada_dict, modelo_path)

            if resultado:
                self.resultado["text"] = f"¡Sobrevive! ({prob:.2%})"
                self.resultado["foreground"] = "green"
            else:
                self.resultado["text"] = f"No sobrevive ({prob:.2%})"
                self.resultado["foreground"] = "red"

            self.barra["value"] = prob * 100

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def limpiar_campos(self):
        for f in FEATURES + OTROS_CAMPOS:
            if f == "Sex":
                self.vars[f].set("male")
            elif f == "Embarked":
                self.vars[f].set("S")
            else:
                self.vars[f].set("")
        self.resultado["text"] = ""
        self.barra["value"] = 0


if __name__ == "__main__":
    Interfaz()
