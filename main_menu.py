import subprocess
import tkinter as tk
from ttkbootstrap import Style
from ttkbootstrap.widgets import Button

def run_game(script_name):
    subprocess.Popen(["python", script_name])

# Janela Principal
app = tk.Tk()
app.title("Menu de Jogos")
app.geometry("400x400")
style = Style("darkly")
style.master = app

#Titulo
title_label = tk.Label(app, text="Selecione um Jogo", font=("Helvetica", 18), bg=style.colors.bg, fg="white")
title_label.pack(pady=20)

# Lista dos Jogos
games = {
    "Memory Game (Jogo da Memória)": "memoryGame.py",
    "POC (Identificador de Expressões)": "poc.py",
    "POCMemory (Jogo da Memória + Expressões)": "pocmemory.py",
    "TouchGame (Acerte o Alvo)": "touchGame.py",
    "TouchGame2 (Acerte o Alvo 2)": "touchGameatt.py"
}

# Botões
for game_name, script_file in games.items():
    btn = Button(app, text=game_name, bootstyle="info", command=lambda s=script_file: run_game(s))
    btn.pack(pady=10, fill="x", padx=40)


app.mainloop()