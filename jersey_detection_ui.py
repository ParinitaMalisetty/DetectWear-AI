import os
import cv2
import time
import threading
import datetime
import pandas as pd
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import torch


# =========================
# YOUR PATHS
# =========================

VIDEO_PATH = r"uploads/testvidslo.mp4"

HUMAN_MODEL_PATH = r"best1/best.pt"
JERSEY_MODEL_PATH = r"bestf1/best.pt"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = "outputs"
ALERT_DIR = "alerts"
OUTPUT_PATH = rf"outputs/output_{timestamp}.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)


# =========================
# THEME
# =========================

YELLOW = "#FFC300"
LIGHT_YELLOW = "#FFD95A"
BG = "#050505"
CARD = "#101010"
INNER = "#020202"
ALERT = "#FF3B30"
WHITE = "#FFFFFF"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class JerseyDetectionUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI-Based Human-Centric Wearable Number Detection in CCTV Streams")
        self.geometry("1400x820")
        self.minsize(1200, 720)
        self.configure(fg_color=BG)

        self.video_source = VIDEO_PATH
        self.cap = None
        self.running = False
        self.thread = None

        self.human_model = None
        self.jersey_model = None

        self.writer = None
        self.output_video_path = OUTPUT_PATH

        self.frame_count = 0
        self.start_time = None
        self.alert_count = 0
        self.logs = []
        self.alert_images = []

        # =========================
        # ALERT FIX
        # =========================

        self.last_alert_time = 0
        self.alert_cooldown = 8.0

        self.no_jersey_streak = 0
        self.no_jersey_required_frames = 8

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.save_var = ctk.BooleanVar(value=True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.build_ui()
        self.load_models()

        self.status.configure(text=f"Status: Default video loaded - {VIDEO_PATH}")

    def build_ui(self):
        header = ctk.CTkFrame(self, fg_color=BG)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="AI-BASED HUMAN-CENTRIC WEARABLE NUMBER DETECTION IN CCTV STREAMS",
            font=("Segoe UI", 34, "bold"),
            text_color=LIGHT_YELLOW
        ).grid(row=0, column=0)

        control = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=18,
            border_width=2,
            border_color=YELLOW
        )
        control.grid(row=1, column=0, sticky="ew", padx=20, pady=12)
        control.grid_columnconfigure(3, weight=1)

        ctk.CTkButton(
            control,
            text="Upload Video",
            height=45,
            fg_color=YELLOW,
            hover_color=LIGHT_YELLOW,
            text_color="black",
            font=("Segoe UI", 15, "bold"),
            command=self.upload_video
        ).grid(row=0, column=0, padx=14, pady=14, sticky="ew")

        ctk.CTkButton(
            control,
            text="▶  Start Detection",
            height=45,
            fg_color=YELLOW,
            hover_color=LIGHT_YELLOW,
            text_color="black",
            font=("Segoe UI", 15, "bold"),
            command=self.start_detection
        ).grid(row=0, column=1, padx=14, pady=14, sticky="ew")

        ctk.CTkButton(
            control,
            text="■  Stop",
            height=45,
            fg_color=ALERT,
            hover_color="#FF6666",
            text_color="white",
            font=("Segoe UI", 15, "bold"),
            command=self.stop_detection
        ).grid(row=0, column=2, padx=14, pady=14, sticky="ew")

        self.rtsp_entry = ctk.CTkEntry(
            control,
            placeholder_text="Enter CCTV / RTSP Link",
            height=45,
            fg_color=INNER,
            border_color=YELLOW,
            text_color=WHITE,
            font=("Segoe UI", 14)
        )
        self.rtsp_entry.grid(row=0, column=3, padx=14, pady=14, sticky="ew")

        ctk.CTkButton(
            control,
            text="Connect CCTV",
            height=45,
            fg_color=YELLOW,
            hover_color=LIGHT_YELLOW,
            text_color="black",
            font=("Segoe UI", 15, "bold"),
            command=self.connect_cctv
        ).grid(row=0, column=4, padx=14, pady=14, sticky="ew")

        ctk.CTkCheckBox(
            control,
            text="Save Output MP4",
            variable=self.save_var,
            fg_color=YELLOW,
            hover_color=LIGHT_YELLOW,
            border_color=YELLOW,
            checkmark_color="black",
            text_color=WHITE,
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=5, padx=14, pady=14, sticky="ew")

        main = ctk.CTkFrame(self, fg_color=BG)
        main.grid(row=3, column=0, sticky="nsew", padx=20, pady=8)

        main.grid_columnconfigure(0, weight=4)
        main.grid_columnconfigure(1, weight=4)
        main.grid_columnconfigure(2, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.original_label = self.make_video_panel(
            main, 0, "ORIGINAL FEED", "🎥", "Original Video Window"
        )

        self.processed_label = self.make_video_panel(
            main, 1, "AI PROCESSED FEED", "⚙", "Detection Output Window"
        )

        alert_panel = ctk.CTkFrame(
            main,
            fg_color=CARD,
            corner_radius=18,
            border_width=2,
            border_color=ALERT
        )
        alert_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=5)
        alert_panel.grid_rowconfigure(1, weight=1)
        alert_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            alert_panel,
            text="⚠  NO-JERSEY ALERTS",
            font=("Segoe UI", 18, "bold"),
            text_color=ALERT
        ).grid(row=0, column=0, pady=(16, 8))

        self.alert_scroll = ctk.CTkScrollableFrame(
            alert_panel,
            fg_color=INNER,
            corner_radius=14
        )
        self.alert_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(5, 12))
        self.alert_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            alert_panel,
            text="Export Alert Log CSV",
            height=36,
            fg_color=ALERT,
            hover_color="#FF6666",
            font=("Segoe UI", 14, "bold"),
            command=self.export_logs
        ).grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        bottom = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=18,
            border_width=2,
            border_color=YELLOW
        )
        bottom.grid(row=4, column=0, sticky="ew", padx=20, pady=(8, 18))

        for i in range(6):
            bottom.grid_columnconfigure(i, weight=1)

        self.fps_value = self.bottom_card(bottom, 0, "◉", "FPS", "0.0")
        self.human_value = self.bottom_card(bottom, 1, "♙", "HUMANS", "0")
        self.jersey_value = self.bottom_card(bottom, 2, "▣", "JERSEYS", "0")
        self.alert_value = self.bottom_card(bottom, 3, "⚠", "ALERTS", "0")
        self.time_value = self.bottom_card(bottom, 4, "◷", "TIME", "00:00:00")
        self.output_value = self.bottom_card(bottom, 5, "▱", "OUTPUT", "Not Saved")

        self.status = ctk.CTkLabel(
            self,
            text="Status: Starting...",
            font=("Segoe UI", 13, "bold"),
            text_color=LIGHT_YELLOW
        )
        self.status.grid(row=5, column=0, pady=(0, 8))

    def make_video_panel(self, parent, col, title_text, icon, center_text):
        panel = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=18,
            border_width=2,
            border_color=YELLOW
        )
        panel.grid(row=0, column=col, sticky="nsew", padx=8, pady=5)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text=f"{icon}  {title_text}",
            font=("Segoe UI", 21, "bold"),
            text_color=LIGHT_YELLOW
        ).grid(row=0, column=0, pady=(16, 8))

        display = ctk.CTkFrame(panel, fg_color=INNER, corner_radius=14)
        display.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))
        display.grid_rowconfigure(0, weight=1)
        display.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(
            display,
            text=center_text,
            font=("Segoe UI", 16),
            text_color=LIGHT_YELLOW
        )
        label.grid(row=0, column=0)

        return label

    def bottom_card(self, parent, col, icon, label, value):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=col, padx=10, pady=14, sticky="ew")

        ctk.CTkLabel(
            box,
            text=icon,
            font=("Segoe UI", 25),
            text_color=LIGHT_YELLOW
        ).pack(side="left", padx=(5, 10))

        text_box = ctk.CTkFrame(box, fg_color="transparent")
        text_box.pack(side="left")

        ctk.CTkLabel(
            text_box,
            text=label,
            font=("Segoe UI", 12, "bold"),
            text_color=LIGHT_YELLOW
        ).pack(anchor="w")

        value_label = ctk.CTkLabel(
            text_box,
            text=value,
            font=("Segoe UI", 18, "bold"),
            text_color=WHITE
        )
        value_label.pack(anchor="w")

        return value_label

    def load_models(self):
        try:
            self.status.configure(text="Status: Loading YOLO models...")

            self.human_model = YOLO(HUMAN_MODEL_PATH)
            self.jersey_model = YOLO(JERSEY_MODEL_PATH)

            self.status.configure(
                text=f"Status: Models Loaded | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
            )

        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            self.status.configure(text="Status: Model loading failed")

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )

        if path:
            self.video_source = path
            self.status.configure(text=f"Status: Video selected - {os.path.basename(path)}")

    def connect_cctv(self):
        source = self.rtsp_entry.get().strip()

        if source == "":
            source = "0"

        self.video_source = int(source) if source == "0" else source
        self.status.configure(text="Status: CCTV/Webcam source selected")

    def start_detection(self):
        if self.video_source is None:
            messagebox.showerror("Missing Source", "Upload video or connect CCTV first.")
            return

        if self.human_model is None or self.jersey_model is None:
            messagebox.showerror("Missing Model", "Models are not loaded.")
            return

        if self.running:
            return

        self.running = True
        self.frame_count = 0
        self.alert_count = 0
        self.logs = []
        self.alert_images = []

        self.no_jersey_streak = 0
        self.last_alert_time = 0

        self.alert_value.configure(text="0")
        self.output_value.configure(text="Starting")

        self.start_time = time.time()

        self.thread = threading.Thread(target=self.process_stream, daemon=True)
        self.thread.start()

    def stop_detection(self):
        self.running = False
        self.status.configure(text="Status: Stopping... Finalizing output video")

    def process_stream(self):
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            self.status.configure(text="Status: Could not open video source")
            self.running = False
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = self.cap.get(cv2.CAP_PROP_FPS)

        if fps_in <= 0:
            fps_in = 25

        self.writer = None

        if self.save_var.get():
            self.output_video_path = rf"outputs/output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                fps_in,
                (width, height)
            )

            if not self.writer.isOpened():
                self.writer = None
                self.output_value.configure(text="Save Failed")
                self.status.configure(text="Status: Output video writer failed")
            else:
                self.output_value.configure(text="Saving")
                self.status.configure(text="Status: Detection Running | Saving Output")
        else:
            self.output_video_path = None
            self.output_value.configure(text="Not Saved")
            self.status.configure(text="Status: Detection Running")

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                break

            self.frame_count += 1

            original = frame.copy()

            processed, human_count, jersey_count = self.detect_frame(frame)

            if self.writer is not None:
                self.writer.write(processed)

            self.show_frame(self.original_label, original)
            self.show_frame(self.processed_label, processed)

            elapsed = time.time() - self.start_time
            fps_now = self.frame_count / elapsed if elapsed > 0 else 0

            self.fps_value.configure(text=f"{fps_now:.1f}")
            self.human_value.configure(text=str(human_count))
            self.jersey_value.configure(text=str(jersey_count))
            self.alert_value.configure(text=str(self.alert_count))
            self.time_value.configure(
                text=time.strftime("%H:%M:%S", time.gmtime(elapsed))
            )

        if self.cap:
            self.cap.release()

        if self.writer:
            self.writer.release()
            self.output_value.configure(text="Saved")
            self.status.configure(text=f"Status: Output saved - {self.output_video_path}")
        else:
            if self.save_var.get():
                self.output_value.configure(text="Save Failed")
            else:
                self.output_value.configure(text="Not Saved")
                self.status.configure(text="Status: Completed")

        self.running = False

    def detect_frame(self, frame):
        human_count = 0
        jersey_count = 0

        human_results = self.human_model.predict(
            frame,
            conf=0.45,
            imgsz=640,
            device=self.device,
            verbose=False
        )

        for result in human_results:
            for box in result.boxes:

                cls_id = int(box.cls[0])

                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                human_count += 1

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                jersey_detected = False

                jersey_results = self.jersey_model.predict(
                    crop,
                    conf=0.35,
                    imgsz=640,
                    device=self.device,
                    verbose=False
                )

                for jr in jersey_results:
                    for jbox in jr.boxes:

                        j_cls = int(jbox.cls[0])
                        j_conf = float(jbox.conf[0])

                        if j_conf < 0.45:
                            continue

                        label = self.jersey_model.names[j_cls]

                        jx1, jy1, jx2, jy2 = map(int, jbox.xyxy[0])

                        fx1 = x1 + jx1
                        fy1 = y1 + jy1
                        fx2 = x1 + jx2
                        fy2 = y1 + jy2

                        jersey_detected = True
                        jersey_count += 1

                        cv2.rectangle(
                            frame,
                            (fx1, fy1),
                            (fx2, fy2),
                            (0, 255, 255),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Jersey {label}",
                            (fx1, max(30, fy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                if jersey_detected:

                    self.no_jersey_streak = 0

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "Human",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                else:

                    self.no_jersey_streak += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "NO JERSEY NUMBER",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

                    current_time = time.time()

                    if (
                        self.no_jersey_streak >= self.no_jersey_required_frames
                        and current_time - self.last_alert_time >= self.alert_cooldown
                    ):

                        self.last_alert_time = current_time
                        self.no_jersey_streak = 0

                        self.create_alert(crop)

        return frame, human_count, jersey_count

    def create_alert(self, crop):

        self.alert_count += 1

        timestamp = time.strftime("%H:%M:%S")

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(crop_rgb)

        alert_path = os.path.join(
            ALERT_DIR,
            f"alert_{self.alert_count}.jpg"
        )

        pil_img.save(alert_path)

        self.logs.append({
            "alert_id": self.alert_count,
            "time": timestamp,
            "frame": self.frame_count,
            "reason": "Person detected without jersey number",
            "image": alert_path
        })

        self.after(
            0,
            lambda:
            self.add_alert_card(
                pil_img.copy(),
                timestamp
            )
        )

    def add_alert_card(self, pil_img, timestamp):

        thumb = pil_img.copy()
        thumb.thumbnail((82, 62))

        tk_thumb = ImageTk.PhotoImage(thumb)

        self.alert_images.append(tk_thumb)

        alert_id = self.alert_count

        card = ctk.CTkFrame(
            self.alert_scroll,
            fg_color="#180404",
            corner_radius=14,
            border_width=1,
            border_color=ALERT
        )

        card.grid(
            row=alert_id,
            column=0,
            sticky="ew",
            padx=6,
            pady=7
        )

        card.grid_columnconfigure(1, weight=1)

        img_btn = ctk.CTkButton(
            card,
            image=tk_thumb,
            text="",
            width=88,
            height=68,
            fg_color="#0A0A0A",
            hover_color="#2A0505",
            command=lambda:
            self.open_alert_popup(
                pil_img,
                f"Alert #{alert_id} | Person Without Jersey Number"
            )
        )

        img_btn.grid(row=0, column=0, padx=8, pady=8)

        ctk.CTkLabel(
            card,
            text=f"Alert #{alert_id}\nNo Jersey Detected\nTime: {timestamp}",
            justify="left",
            anchor="w",
            font=("Segoe UI", 11, "bold"),
            text_color="white",
            wraplength=150
        ).grid(row=0, column=1, sticky="w", padx=(6, 8), pady=8)

    def open_alert_popup(self, pil_img, alert_text):

        popup = ctk.CTkToplevel(self)

        popup.title("No-Jersey Alert Preview")
        popup.configure(fg_color=BG)

        popup.geometry("560x450")

        ctk.CTkLabel(
            popup,
            text=alert_text,
            font=("Segoe UI", 21, "bold"),
            text_color=ALERT
        ).pack(pady=(22, 15))

        img = pil_img.copy()
        img.thumbnail((420, 300))

        tk_big = ImageTk.PhotoImage(img)

        img_label = ctk.CTkLabel(
            popup,
            image=tk_big,
            text=""
        )

        img_label.image = tk_big
        img_label.pack(pady=10)

        ctk.CTkButton(
            popup,
            text="Close",
            command=popup.destroy,
            fg_color=ALERT,
            hover_color="#FF6666",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=12)

    def show_frame(self, label, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_w = max(label.winfo_width(), 400)
        label_h = max(label.winfo_height(), 300)

        img = Image.fromarray(frame)

        img.thumbnail((label_w, label_h))

        tk_img = ImageTk.PhotoImage(img)

        label.configure(image=tk_img, text="")
        label.image = tk_img

    def export_logs(self):

        if not self.logs:
            messagebox.showwarning(
                "No Logs",
                "No alert logs available."
            )
            return

        path = os.path.join(
            OUTPUT_DIR,
            "no_jersey_alert_log.csv"
        )

        pd.DataFrame(self.logs).to_csv(
            path,
            index=False
        )

        messagebox.showinfo(
            "Export Complete",
            f"Saved to:\n{path}"
        )


if __name__ == "__main__":

    app = JerseyDetectionUI()
    app.mainloop()