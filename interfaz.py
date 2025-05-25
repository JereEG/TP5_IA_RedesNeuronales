import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn

# Campos que utiliza el modelo
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden_layers, activation_fn):
        super().__init__()
        layers = []
        last_size = n_inputs
        for h in hidden_layers:
            layers.append(nn.Linear(last_size, h))
            layers.append(activation_fn())
            last_size = h
        layers.append(nn.Linear(last_size, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def obtener_hidden_layers_desde_nombre(nombre_modelo: str):
    """
    Extrae cuántas capas ocultas usa el modelo a partir de su nombre.
    Busca un patrón como '5capa' o '8capa' en el nombre.
    Si no encuentra, asume [128] (una capa).
    """
    m = re.search(r'(\d+)capa', nombre_modelo)
    if m:
        num_capas = int(m.group(1))
        return [128] * num_capas
    else:
        return [128]


def predecir_sobrevive(entrada_dict, modelo_path):
    """
    Toma un diccionario con las claves de FEATURES (valores como strings o numéricos),
    carga el modelo guardado en modelo_path (ajustando la arquitectura según nombre),
    y devuelve (bool_pred, probabilidad_float).
    """
    datos = []
    for f in FEATURES:
        val = entrada_dict[f]
        if val is None or val == "":
            raise ValueError(f"El campo '{f}' no puede estar vacío.")
        if f == "Sex":
            texto = val.lower()
            if texto not in ("male", "female"):
                raise ValueError("El campo 'Sex' debe ser 'male' o 'female'.")
            val_num = 1.0 if texto == "male" else 0.0
        else:
            try:
                val_num = float(val)
            except ValueError:
                raise ValueError(f"El campo '{f}' debe ser numérico.")
        datos.append(val_num)

    # Determinar hidden_layers según el nombre de archivo del modelo
    nombre_sin_ext = os.path.splitext(os.path.basename(modelo_path))[0]
    hidden_layers = obtener_hidden_layers_desde_nombre(nombre_sin_ext)

    # Instanciar MLP con esa arquitectura
    modelo = MLP(len(FEATURES), hidden_layers, nn.ReLU)
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu"))
    modelo.eval()

    with torch.no_grad():
        entrada_tensor = torch.tensor([datos], dtype=torch.float32)
        salida = modelo(entrada_tensor).item()
        return (salida >= 0.5), salida


class Interfaz:
    def __init__(self):
        # Ventana principal
        self.root = tk.Tk()
        self.root.title("Predicción Titanic")
        self.root.geometry("650x600")
        self.frame = ttk.Frame(self.root, padding=20)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.vars = {}

        # ─── 1) Entrada para Pclass ─────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Clase del billete (estatus socioeconómico)",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["Pclass"] = tk.StringVar(value="1")
        opcion_pclass = ttk.OptionMenu(
            self.frame,
            self.vars["Pclass"],
            "1",
            "1",  # 1 = 1ª
            "2",  # 2 = 2ª
            "3"   # 3 = 3ª
        )
        opcion_pclass.pack(fill=tk.X, pady=(0, 10))

        # ─── 2) Entrada para Sex ─────────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Sexo (male / female)",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["Sex"] = tk.StringVar(value="male")
        opcion_sex = ttk.OptionMenu(
            self.frame,
            self.vars["Sex"],
            "male",
            "male",
            "female"
        )
        opcion_sex.pack(fill=tk.X, pady=(0, 10))

        # ─── 3) Entrada para Age ─────────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Edad (años)", font=(
            "Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["Age"] = tk.StringVar()
        entrada_age = ttk.Entry(self.frame, textvariable=self.vars["Age"])
        entrada_age.pack(fill=tk.X, pady=(0, 10))

        # ─── 4) Entrada para SibSp ───────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Hermano/a o cónyuge a bordo (SibSp)",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["SibSp"] = tk.StringVar()
        entrada_sibsp = ttk.Entry(self.frame, textvariable=self.vars["SibSp"])
        entrada_sibsp.pack(fill=tk.X, pady=(0, 10))

        # ─── 5) Entrada para Parch ────────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Padre/madre o hijo/a a bordo (Parch)",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["Parch"] = tk.StringVar()
        entrada_parch = ttk.Entry(self.frame, textvariable=self.vars["Parch"])
        entrada_parch.pack(fill=tk.X, pady=(0, 10))

        # ─── 6) Entrada para Fare ─────────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Tarifa (Fare)", font=(
            "Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.vars["Fare"] = tk.StringVar()
        entrada_fare = ttk.Entry(self.frame, textvariable=self.vars["Fare"])
        entrada_fare.pack(fill=tk.X, pady=(0, 10))

        # ─── 7) Selector de modelo ──────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Seleccionar modelo entrenado:",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.modelo_var = tk.StringVar()
        self.modelo_combo = ttk.Combobox(
            self.frame, textvariable=self.modelo_var, state="readonly")
        self.modelo_combo.pack(fill=tk.X, pady=(0, 15))

        # ─── 8) Botones de acción ───────────────────────────────────────────────────────────────────────
        botones_frame = ttk.Frame(self.frame)
        botones_frame.pack(fill=tk.X, pady=(0, 10))

        btn_predict = ttk.Button(
            botones_frame, text="Predecir supervivencia", command=self.predecir)
        btn_predict.pack(side=tk.LEFT, expand=True, padx=(0, 5))

        btn_clear = ttk.Button(
            botones_frame, text="Limpiar campos", command=self.limpiar_campos)
        btn_clear.pack(side=tk.LEFT, expand=True, padx=(5, 5))

        btn_test = ttk.Button(
            botones_frame, text="Hacer test con ejemplos", command=self.hacer_test_ejemplos)
        btn_test.pack(side=tk.LEFT, expand=True, padx=(5, 0))

        # ─── 9) Área de resultados ────────────────────────────────────────────────────────────────────────
        ttk.Label(self.frame, text="Resultados:", font=(
            "Arial", 10, "bold")).pack(anchor="w")
        self.text_area = tk.Text(self.frame, height=12, wrap="word")
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # ─── 10) Barra de probabilidad ────────────────────────────────────────────────────────────────────
        self.barra = ttk.Progressbar(
            self.frame, orient='horizontal', length=400, mode='determinate')
        self.barra.pack(pady=(10, 0))

        # Cargar modelos en el Combobox (después de inicializar la ventana)
        self.root.after(100, self.cargar_modelos)
        self.root.mainloop()

    def cargar_modelos(self):
        """Verifica la carpeta 'models/' y llena el Combobox con los archivos .pt."""
        if not os.path.exists("models"):
            os.makedirs("models")
        archivos = os.listdir("models")
        modelos = [f for f in archivos if f.endswith(".pt")]
        self.modelo_combo["values"] = modelos
        if modelos:
            self.modelo_var.set(modelos[0])

    def predecir(self):
        """Toma los valores ingresados, ejecuta la predicción y muestra el resultado."""
        try:
            entrada_dict = {f: self.vars[f].get() for f in FEATURES}

            modelo_seleccionado = self.modelo_var.get()
            if modelo_seleccionado == "":
                raise ValueError("Debe seleccionar un modelo para predecir.")

            modelo_path = os.path.join("models", modelo_seleccionado)
            pred_bool, prob = predecir_sobrevive(entrada_dict, modelo_path)

            # Mostrar resultado individual
            self.text_area.delete("1.0", tk.END)
            if pred_bool:
                texto = f"Predicción: Sobrevive ✔  (Probabilidad: {prob:.2%})\n"
                self.text_area.insert(tk.END, texto)
                self.text_area.tag_configure("sobrevive", foreground="green")
                self.text_area.tag_add("sobrevive", "1.0", "1.end")
            else:
                texto = f"Predicción: No sobrevive ✘  (Probabilidad: {prob:.2%})\n"
                self.text_area.insert(tk.END, texto)
                self.text_area.tag_configure("no_sobrevive", foreground="red")
                self.text_area.tag_add("no_sobrevive", "1.0", "1.end")

            # Actualizar barra
            self.barra["value"] = prob * 100

        except Exception as e:
            messagebox.showerror("Error de predicción", str(e))

    def limpiar_campos(self):
        """Limpia todos los campos de entrada, el área de texto y la barra."""
        for f in FEATURES:
            if f == "Pclass":
                self.vars[f].set("1")
            elif f == "Sex":
                self.vars[f].set("male")
            else:
                self.vars[f].set("")
        self.text_area.delete("1.0", tk.END)
        self.barra["value"] = 0

    def hacer_test_ejemplos(self):
        """
        Realiza la predicción de cuatro ejemplos hardcodeados y 
        muestra en el área de texto: Nombre, Predicción, Probabilidad, 
        y Valor real de 'survived'.
        """
        ejemplos = [
            {
                "Name": "Allen, Miss. Elisabeth Walton",
                "Pclass": "1", "Sex": "female", "Age": "29", "SibSp": "0", "Parch": "0", "Fare": "211.3375",
                "Survived": 1
            },
            {
                "Name": "Allison, Master. Hudson Trevor",
                "Pclass": "1", "Sex": "male", "Age": "0.9167", "SibSp": "1", "Parch": "2", "Fare": "151.55",
                "Survived": 1
            },
            {
                "Name": "Allison, Miss. Helen Loraine",
                "Pclass": "1", "Sex": "female", "Age": "2", "SibSp": "1", "Parch": "2", "Fare": "151.55",
                "Survived": 0
            },
            {
                "Name": "Allison, Mr. Hudson Joshua Creighton",
                "Pclass": "1", "Sex": "male", "Age": "30", "SibSp": "1", "Parch": "2", "Fare": "151.55",
                "Survived": 0
            }
        ]

        modelo_seleccionado = self.modelo_var.get()
        if modelo_seleccionado == "":
            messagebox.showerror(
                "Error", "Debe seleccionar un modelo para realizar el test.")
            return

        modelo_path = os.path.join("models", modelo_seleccionado)
        if not os.path.exists(modelo_path):
            messagebox.showerror(
                "Error", f"No se encontró el modelo: {modelo_seleccionado}")
            return

        # Limpiamos área de texto y barra antes de mostrar los resultados
        self.text_area.delete("1.0", tk.END)
        self.barra["value"] = 0

        # Iteramos sobre cada ejemplo
        for idx, ej in enumerate(ejemplos, start=1):
            name = ej["Name"]
            survived_real = bool(ej["Survived"])
            try:
                pred_bool, prob = predecir_sobrevive(ej, modelo_path)
            except Exception as e:
                texto_error = f"Ejemplo {idx}: Error al predecir '{name}': {e}\n\n"
                self.text_area.insert(tk.END, texto_error)
                continue

            linea = (
                f"Ejemplo {idx}: {name}\n"
                f"  • Predicción: {'Sobrevive ✔' if pred_bool else 'No sobrevive ✘'} "
                f"(Prob: {prob:.2%})\n"
                f"  • Valor real: {'Sobrevive' if survived_real else 'No sobrevive'}\n\n"
            )
            self.text_area.insert(tk.END, linea)

        # En el test múltiple, dejamos la barra en 0 para evitar confusión
        self.barra["value"] = 0


if __name__ == "__main__":
    Interfaz()
