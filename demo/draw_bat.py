import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json

class PolygonDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bat Icon Tracer")
        
        # --- UI Frame ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame, text="1. Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="2. Save JSON", command=self.save_json).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear Points", command=self.clear_points).pack(side=tk.RIGHT, padx=5)
        
        # --- Canvas ---
        self.canvas = tk.Canvas(root, bg="gray", width=600, height=600, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.add_point)
        
        # --- State ---
        self.image = None
        self.image_tk = None
        self.points = []
        self.lines = []
        self.ovals = []
        
    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if not filepath:
            return
            
        self.image = Image.open(filepath)
        
        # Resize image to fit inside 800x800 bounding box for easy tracing
        self.image.thumbnail((800, 800))
        self.image_tk = ImageTk.PhotoImage(self.image)
        
        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.clear_points()
        
    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        
        # Draw a small dot
        r = 3
        oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="white")
        self.ovals.append(oval)
        
        # Draw a line connecting to the previous point
        if len(self.points) > 1:
            x_prev, y_prev = self.points[-2]
            line = self.canvas.create_line(x_prev, y_prev, x, y, fill="red", width=2)
            self.lines.append(line)
            
    def clear_points(self):
        self.points.clear()
        for item in self.lines + self.ovals:
            self.canvas.delete(item)
        self.lines.clear()
        self.ovals.clear()
        
    def save_json(self):
        if len(self.points) < 3:
            messagebox.showwarning("Warning", "Please click at least 3 points to make a polygon.")
            return
            
        # 1. Calculate the center of the drawn polygon
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # 2. Find the max distance from center to scale everything between -1.0 and 1.0
        max_dist = max(max(xs)-min(xs), max(ys)-min(ys)) / 2.0
        if max_dist == 0: max_dist = 1
        
        normalized_points = []
        for x, y in self.points:
            nx = (x - cx) / max_dist
            # 3. INVERT the Y axis! Tkinter Y goes down, Matplotlib Y goes up.
            ny = -(y - cy) / max_dist 
            # Round to 3 decimal places for a clean JSON file
            normalized_points.append((round(nx, 3), round(ny, 3)))
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="custom_bat.json",
            filetypes=[("JSON Files", "*.json")]
        )
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(normalized_points, f, indent=4)
            messagebox.showinfo("Success", f"Saved {len(normalized_points)} points to JSON!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonDrawer(root)
    root.mainloop()