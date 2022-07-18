#!/usr/bin/env python

import cv2
from tkinter import filedialog as fd
import tkinter as tk
from tkinter import ttk

import threading

from amplify_spatial_gdown_temporal_ideal import process_input


class App:

    def __init__(self):
        self.buffer = []
        self.filename = ""

        self.window = tk.Tk()
        self.window.title("Eulerian Colour Magnification Demo")
        self.window.bind("<<Halt_Thread>>", self.amplify_thread_finished)

        self.main_container = tk.Frame(self.window)
        self.main_container.pack(side="top", fill="both", expand=True)

        self.fl_frame = tk.Frame(self.main_container)
        self.fl_frame.pack(side="top", fill="both", expand=True)
        self.fh_frame = tk.Frame(self.main_container)
        self.fh_frame.pack(side="top", fill="both", expand=True)
        self.alpha_frame = tk.Frame(self.main_container)
        self.alpha_frame.pack(side="top", fill="both", expand=True)
        self.level_frame = tk.Frame(self.main_container)
        self.level_frame.pack(side="top", fill="both", expand=True)
        self.sample_frame = tk.Frame(self.main_container)
        self.sample_frame.pack(side="top", fill="both", expand=True)       

        self.progress_frame = tk.Frame(self.main_container)
        self.progress_frame.pack(side="bottom",expand=True, fill="x")

        self.buttons2 = tk.Frame(self.main_container)
        self.buttons2.pack(side="bottom",expand=True, fill="y")

        self.buttons1 = tk.Frame(self.main_container)
        self.buttons1.pack(side="bottom",expand=True, fill="y")
    
        self.fl_label = tk.Label(
            self.fl_frame,
            text="low frequency cutoff Hz",
            anchor="w",
        )
        self.fl_label.pack(side="left",expand=True,fill='x')
        self.fl_entry = tk.Entry(self.fl_frame)
        self.fl_entry.insert(0, str(0.8))
        self.fl_entry.pack(side="right",expand=True,fill='x')

        self.fh_label = tk.Label(
            self.fh_frame,
            text="high frequency cutoff Hz",
            anchor="w",
        )
        self.fh_label.pack(side="left",expand=True,fill='x')
        self.fh_entry = tk.Entry(self.fh_frame)
        self.fh_entry.insert(0, str(1.0))
        self.fh_entry.pack(side="right",expand=True,fill='x')

        self.alpha_label = tk.Label(
            self.alpha_frame,
            text="magnification factor",
            anchor="w",
        )
        self.alpha_label.pack(side="left",expand=True,fill='x')
        self.alpha_entry = tk.Entry(self.alpha_frame)
        self.alpha_entry.insert(0, str(100))
        self.alpha_entry.pack(side="right",expand=True,fill='x')

        self.level_label = tk.Label(
            self.level_frame,
            text="number of levels",
            anchor="w",
        )
        self.level_label.pack(side="left",expand=True,fill='x')
        self.level_entry = tk.Entry(self.level_frame)
        self.level_entry.insert(0, str(4))
        self.level_entry.pack(side="right",expand=True,fill='x')

        self.sample_label = tk.Label(
            self.sample_frame,
            text="sample rate",
            anchor="w",
        )
        self.sample_label.pack(side="left",expand=True,fill='x')
        self.sample_entry = tk.Entry(self.sample_frame)
        self.sample_entry.insert(0, str(30))
        self.sample_entry.pack(side="right",expand=True,fill='x')

        self.file_button = tk.Button(
            self.buttons1,
            text="choose file",
            width=5,
            height=2,
            command=self.choose_file,
        )
        self.file_button.pack(side="left")

        self.play_file_button = tk.Button(
            self.buttons1,
            text="play file",
            width=5,
            height=2,
            command=self.play_file,
        )
        self.play_file_button.pack(side="left")

        self.amplify_file_button = tk.Button(
            self.buttons2,
            text="process file",
            width=5,
            height=2,
            command=self.start_amplify_thread,
        )
        self.amplify_file_button.pack(side="left")

        self.play_buffer_button = tk.Button(
            self.buttons2,
            text="play result",
            width=5,
            height=2,
            command=self.play_buffer,
        )
        self.play_buffer_button.pack(side="left")

        self.clear_buffer_button = tk.Button(
            self.buttons2,
            text="clear cache",
            width=5,
            height=2,
            command=self.clear_buffer,
        )
        self.clear_buffer_button.pack(side="left")

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            length=300,
            orient='horizontal',
            mode='determinate'
        )
        self.progress_bar.pack(side='bottom')


    def encode_file(self):
        vid = cv2.VideoCapture(self.filename)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                success, framejpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
                if success:
                    self.buffer.append(framejpg)
                else:
                    break
            else:
                break
        vid.release()

    def play_file(self):
        vid = cv2.VideoCapture(self.filename)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                cv2.imshow(
                    "Frame",
                    frame
                )
            else:
                break
            cv2.waitKey(33)
        vid.release()
        cv2.destroyAllWindows()

    def play_buffer(self):
        for frame in self.buffer:
            cv2.imshow(
                "Buffer",
                cv2.imdecode(frame, cv2.IMREAD_COLOR)
            )
            k = cv2.waitKey(33)
            if k == 27:
                break
        cv2.destroyAllWindows()

    def clear_buffer(self):
        self.buffer = []
        print("Emptying playback buffer")

    def progress_callback(self, increment):
        self.progress_bar["value"] += increment

    def amplify_file(self):
        self.progress_bar["value"] = 0
        for frame in process_input(
            self.filename,
            int(self.alpha_entry.get()),
            int(self.level_entry.get()),
            float(self.fl_entry.get()),
            float(self.fh_entry.get()),
            int(self.sample_entry.get()),
            self.progress_callback,
        ):
            success, framejpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            )
            if success:
                self.buffer.append(framejpg)
            else:
                break
        self.window.event_generate("<<Halt_Thread>>")

    def amplify_thread_finished(self, event=None):
        print("Initiating Playback")
        self.play_buffer()
        self.amplify_thread = None

    def choose_file(self):
        self.filename = fd.askopenfilename()

    def start_amplify_thread(self):
        self.amplify_thread = threading.Thread(target=self.amplify_file)
        self.amplify_thread.start()

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
