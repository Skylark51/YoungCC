"""Tkinter front end for the shared single-point engine."""

import json
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import filedialog, messagebox, ttk

from ..inputs.molecule import parse_input
from ..reporting.output import calculate

DEFAULT_INPUT = {
    "method": "LDA", "basis": "STO-3G", "units": "angstrom",
    "charge": 0, "multiplicity": 1,
    "atoms": [["H", 0, 0, -0.37], ["H", 0, 0, 0.37]],
}


def main():
    root = tk.Tk()
    root.title("YoungCC for Windows")
    root.geometry("820x650")
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Calculation input (JSON)").pack(anchor="w")
    editor = tk.Text(frame, height=17, wrap="none")
    editor.pack(fill="x")
    editor.insert("1.0", json.dumps(DEFAULT_INPUT, indent=2))
    bar = ttk.Frame(frame, padding=(0, 8))
    bar.pack(fill="x")
    status = tk.StringVar(value="Ready")
    output = tk.Text(frame, height=12, state="disabled", wrap="none")
    output.pack(fill="both", expand=True)
    ttk.Label(frame, textvariable=status).pack(anchor="w")
    events = Queue()
    input_path = None

    def load():
        nonlocal input_path
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                source = Path(filename)
                contents = source.read_text(encoding="utf-8-sig")
            except (OSError, ValueError) as error:
                messagebox.showerror("Input error", str(error))
                return
            editor.delete("1.0", "end")
            editor.insert("1.0", contents)
            input_path = source.resolve()

    def append(line):
        output.configure(state="normal")
        output.insert("end", line + "\n")
        output.see("end")
        output.configure(state="disabled")

    def poll():
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "line":
                    append(value)
                else:
                    status.set(value)
                    run_button.configure(state="normal")
                    return
        except Empty:
            root.after(60, poll)

    def run():
        try:
            molecule = parse_input(json.loads(editor.get("1.0", "end")))
        except ValueError as error:
            messagebox.showerror("Input error", str(error))
            return
        filename = filedialog.asksaveasfilename(
            initialfile="result.json", defaultextension=".json",
            filetypes=[("JSON result", "*.json")],
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".json":
            messagebox.showerror("Output error", "Choose a filename ending in .json.")
            return
        if path.resolve() == input_path:
            messagebox.showerror("Output error", "Choose a different file from the input.")
            return
        prefix = path.with_suffix("")
        run_button.configure(state="disabled")
        status.set("Running SCF...")
        append(f"Output prefix: {prefix}")

        def worker():
            try:
                result = calculate(molecule, prefix, emit=lambda line: events.put(("line", line)))
                message = "Converged" if result.converged else "SCF did not converge"
                events.put(("done", f"{message}. Results saved to {prefix}.json and .log"))
            except Exception as error:
                events.put(("done", f"Calculation failed: {error}"))

        Thread(target=worker, daemon=True).start()
        root.after(60, poll)

    ttk.Button(bar, text="Open input", command=load).pack(side="left")
    run_button = ttk.Button(bar, text="Run calculation", command=run)
    run_button.pack(side="left", padx=8)

    def close():
        if run_button.instate(["disabled"]):
            messagebox.showinfo("Running SCF", "Wait for the calculation to finish before closing.")
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


if __name__ == "__main__":
    main()
