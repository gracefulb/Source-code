import re
from ui.parameter_frame import ParameterFrame
from ui.parameterdisplay import ParameterDisplay
from ui.operation_frame import OperationFrame

import sys
from PyQt5.QtWidgets import QApplication,  QVBoxLayout, QWidget,  QLabel
from detect import run, parse_opt
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QTextEdit, QScrollArea, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
# from utils.datasets import LoadImages, LoadStreams
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
# from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, \
#     colorstr, \
#     increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, scale_boxes, \
#     strip_optimizer, xyxy2xywh
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors
from utils.torch_utils import smart_inference_mode, select_device, time_sync
# Call display_results to update image
from ui.result_frame import ResultFrame
import csv

from PyQt5.QtWidgets import QFrame, QLabel, QTextEdit
from PyQt5.QtWidgets import QPushButton, QApplication
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt  # Import QSize


class BackupFrame(QFrame):
    def __init__(self, parent=None, run_detection=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: white; border: 2px solid white;")
        self.setFixedSize(440, 140)
        self.setGeometry(20, 440 + 20 + 320, 440, 140)
        # You can add other functional code for this frame here

        self.set_title()  # Add title
        self.run_detection = run_detection  # Save the passed detection function

        self.is_detecting = True

        # Create buttons
        self.button1 = QPushButton("Run", self)  # Run code
        self.button2 = QPushButton("Exit", self)  # Exit system
        self.button3 = QPushButton("Stop", self)  # Add stop detection button

        # Set button position and size
        self.button1.setGeometry(15, 80, 125, 50)
        self.button2.setGeometry(300, 80, 125, 50)
        self.button3.setGeometry(17+140, 80, 125, 50)  # Stop button position

        # Insert image in top-left corner
        #self.image_label = QLabel(self)
        # pixmap = QPixmap("ui/image/logo.jpg")  # Replace with your image path
        #pixmap = pixmap.scaled(30, 30, Qt.KeepAspectRatio)  # Target size 150x150, keep aspect ratio
        # self.image_label.setPixmap(pixmap)
        # self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Set image in top-left corner

        # Adjust button text size
        font = QFont()
        font.setPointSize(12)  # Set text size to 12
        self.button1.setFont(font)
        self.button2.setFont(font)
        self.button3.setFont(font)

        # Set button style (including raised and pressed effects)
        self.set_button_style(self.button1,  background_color="#90EE90")
        self.set_button_style(self.button2,  background_color="#FFB6C1")
        self.set_button_style(self.button3,  background_color="#FFFFE0")

        # Connect button click events
        self.button1.clicked.connect(self.run_code)
        self.button2.clicked.connect(self.exit_system)  # 修复方法名
        self.button3.clicked.connect(self.stop_detection)  # Stop button click event

    def set_button_icon(self, button, image_path):
        # Set button icon
        icon = QIcon(image_path)
        button.setIcon(icon)
        button.setIconSize(QSize(40, 40))  # Set icon size

    def set_button_style(self, button, background_color="#f0f0f0", border_color="#ccc", pressed_color="#ddd"):
        # Set button style
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {background_color};
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 20px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {border_color};
                padding: 10px 18px;  /* Button slightly smaller when clicked */
            }}
        """)

    def set_title(self):
        title = QLabel("System Operation & Exit", self)  # Set title text
        title.setFont(QFont("SimSun", 16))  # Set font to SimSun, size 16
        title.setAlignment(Qt.AlignCenter)  # Center text
        title.setGeometry(0, 15, self.width(), 50)  # Set position, 0 for X-axis, 10 for top margin, width to window width, height 30
        title.setStyleSheet("color: black;  font-weight: bold;")  # Set text color to black

    def stop_detection(self):
        print("Stopping detection...")
        if self.is_detecting:
            self.is_detecting = False  # Set to False to stop detection

    def run_code(self):
        print("Running code")
        self.is_detecting = True  # 重置检测标志
        # Trigger YOLO detection
        if self.run_detection:
            self.run_detection()  # Call run_detection method in main window

    # 修复方法名 - 添加下划线
    def exit_system(self):
        # Exit system
        print("Exiting system...")
        QApplication.quit()  # Exit application


class DetectionThread(QThread):
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    result_ready = pyqtSignal(list)

    def __init__(self, run_function, opt):
        super().__init__()
        self.run_function = run_function
        self.opt = opt
        self.new_results = []

    def run(self):
        try:
            self.run_function(**vars(self.opt))
            self.result_ready.emit(self.new_results)
        except Exception as e:
            import traceback
            error_msg = f"Error during detection: {str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            self.finished.emit()


class MainWindow(QWidget):
    # update_image_signal = pyqtSignal(np.ndarray)  # Define signal to pass image data

    def __init__(self):
        super().__init__()

        # Set window title
        self.setWindowTitle("Bibo (EFDS)")

        "Set interface color"
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, 
                                            stop: 0 #E58391, stop: 0.85 #E58391, 
                                            stop: 0.85 #808CD6, stop: 1 #808CD6);
            }
        """)

        # Set window size
        screen = QApplication.primaryScreen()  # Get primary screen object
        screen_rect = screen.availableGeometry() # Return available area rectangle coordinates including width and height
        width, height = 1500, 950  # Window width 1000, height 600
        x = (screen_rect.width() - width) // 2  # Calculate horizontal position to center
        y = (screen_rect.height() - height) // 2 # Calculate vertical position to center
        self.setGeometry(x, y, width, height)  # Set window size to 1000x600 and position to x, y
        self.setFixedSize(width, height)  # Fix window size to 1000x600 to prevent resizing

        # Create vertical layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)  # Set spacing between controls to 15 pixels

        # Create QLabel to display text
        self.title_label = QLabel("YOLOV5s-based Elderly Fall Detection System (EFDS)", self)
        self.title_label.setAlignment(Qt.AlignCenter)  # Center text
        self.layout.addWidget(self.title_label)  # Add text label to layout
        self.layout.addStretch(1)  # Add stretch space at bottom to keep text label at top
        # Set text size and font
        font = QFont()  # Create font object
        font.setFamily("SimSun")  # Set font to Arial (can be changed as needed)
        font.setPointSize(25)  # Set font size to 25
        font.setBold(True)  # Set bold
        self.title_label.setFont(font)  # Apply font to QLabel
        self.layout.addWidget(self.title_label)  # Add text label to layout

        # Set layout as window layout
        self.setLayout(self.layout)

        # Create and add frames
        self.result_frame = ResultFrame(self)
        self.operation_frame = OperationFrame(self)
        self.parameter_frame = ParameterFrame(self)
        self.parameter_display = ParameterDisplay(self)
        
        # 只创建一个 BackupFrame 实例
        self.backup_frame = BackupFrame(self, self.run_detection)  # Pass run_detection to BackupFrame

        # Connect signal to ResultFrame's display_image_path_results method
        self.operation_frame.display_image.connect(self.result_frame.display_image_path_results)
        
        # 初始化检测线程
        self.detection_thread = None

    def run_detection(self):
        # 清理之前的线程
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait()
        
        # 定义内部run函数
        def run_detection_internal(weights=ROOT / 'E:/program/yolov5_ui/weight/fall.pt',  # model.pt path(s)
                source=ROOT / 'IMG_3095.MP4',  # file/dir/URL/glob, 0 for webcam
                data=ROOT / "data/customer.yaml",#coco128.yaml",  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
                save_csv=False,  # save results in CSV format
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=False,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project=ROOT / "runs/detect",  # save results to project/name
                name="exp",  # save results to project/name
                exist_ok=False,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride
                ):
                
            new_results = []
            source = str(source)
            save_img = not nosave and not source.endswith(".txt")  # save inference images
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
            webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
            screenshot = source.lower().startswith("screen")
            if is_url and is_file:
                source = check_file(source)  # download

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            # Dataloader
            bs = 1  # batch_size
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
            for path, im, im0s, vid_cap, s in dataset:
                # 检查是否应该停止检测
                if not self.backup_frame.is_detecting:
                    break

                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    if model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                        pred = [pred, None]
                    else:
                        pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # Define the path for the CSV file
                csv_path = save_dir / "predictions.csv"

                # Create or append to the CSV file
                def write_to_csv(image_name, prediction, confidence):
                    """Writes prediction data for an image to a CSV file, appending if the file exists."""
                    data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                    file_exists = os.path.isfile(csv_path)
                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(data)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                    s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = names[c] if hide_conf else f"{names[c]}"
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"

                            if save_csv:
                                write_to_csv(p.name, label, confidence_str)

                            if save_txt:  # Write to file
                                if save_format == 0:
                                    coords = (
                                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    )  # normalized xywh
                                else:
                                    coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                                line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # Print time (inference-only)
                    # print(f'{s}Done. ({t3 - t2:.3f}s)')

                    # Stream results
                    im0 = annotator.result()

                    # 在主线程中更新UI
                    self.result_frame.display_results(im0)

                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                            new_results.append({"Path": save_path})
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                new_results.append({"Path": save_path})
                            vid_writer[i].write(im0)

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                print(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
            
            return new_results

        # 创建选项对象
        opt = parse_opt()

        def contains_chinese(path):
            # Check if path contains Chinese characters
            return bool(re.search(r'[\u4e00-\u9fff]', path))

        # Get model input parameters
        if self.parameter_frame.path_label4.text() == "Open":
            # If camera is enabled
            opt.source = "0"  # Assume "0" represents camera
        elif self.parameter_frame.path_label1.text():
            # If image is selected
            opt.source = self.parameter_frame.path_label1.text()
            if contains_chinese(opt.source):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Path contains Chinese characters")
                msg.setText(f"Image path contains Chinese, please reselect path!\nCurrent path: {opt.source}")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
        elif self.parameter_frame.path_label2.text():
            # If folder is selected
            opt.source = self.parameter_frame.path_label2.text()
            if contains_chinese(opt.source):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Path contains Chinese characters")
                msg.setText(f"Folder path contains Chinese, please reselect path!\nCurrent path: {opt.source}")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return

            # Record files containing Chinese characters
            files_with_chinese = []

            # Traverse all files in folder to check if filename contains Chinese
            for filename in os.listdir(opt.source):
                file_path = os.path.join(opt.source, filename)

                # Only check image files (by extension)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    if contains_chinese(filename):
                        # Add files with Chinese to list
                        files_with_chinese.append((filename, file_path))

            # If Chinese files exist, show scrollable message box
            if files_with_chinese:
                # Build message text
                msg_text = "Folder contains files with Chinese names, please rename them!\n\n"
                for filename, file_path in files_with_chinese:
                    msg_text += f"{filename}\n"

                # Custom dialog
                dialog = QDialog()
                dialog.setWindowTitle("Image filename contains Chinese")
                layout = QVBoxLayout()

                # Create QTextEdit to display message, set to read-only
                text_edit = QTextEdit()
                text_edit.setText(msg_text)
                text_edit.setReadOnly(True)

                # Create scroll area, add QTextEdit to it
                scroll_area = QScrollArea()
                scroll_area.setWidget(text_edit)
                scroll_area.setWidgetResizable(True)

                # Create confirmation button
                btn_ok = QPushButton("Confirmation")
                btn_ok.clicked.connect(dialog.accept)  # Close dialog on click

                # Add widgets to layout
                layout.addWidget(scroll_area)
                layout.addWidget(btn_ok)

                dialog.setLayout(layout)

                # Set dialog size
                dialog.resize(600, 400)  # Adjust dialog size (width 600, height 400)
                dialog.setFixedSize(dialog.size())  # Set fixed size (prevent resizing)

                # Show dialog
                dialog.exec_()
                return

        elif self.parameter_frame.path_label3.text():
            # If video is selected
            opt.source = self.parameter_frame.path_label3.text()
            if contains_chinese(opt.source):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Path contains Chinese characters")
                msg.setText(f"Video path contains Chinese, please reselect path!\nCurrent path: {opt.source}")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
        else:
            # If no input is selected, show warning
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Input Data Notice")
            msg.setText("Please input model data!\nPath should not contain Chinese characters")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()  # Show popup
            return

        # Get weights path
        new_weights = self.parameter_display.path_label1.text()
        if self.parameter_display.path_label1.text() == "":
            # If no input is selected, show warning
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Model Weights")
            msg.setText("Please input model weights!\nPath should not contain Chinese characters")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()  # Show popup
            return

        if contains_chinese(new_weights):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Path contains Chinese characters")
            msg.setText(f"Weights path contains Chinese, please reselect path!\nCurrent path: {new_weights}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        new_weights = Path(new_weights)
        opt.weights = new_weights

        # Get confidence threshold
        new_confidence = self.parameter_display.path_label2.text()
        opt.conf_thres = float(new_confidence)

        # Get IOU threshold
        new_iou_thres = self.parameter_display.path_label3.text()
        opt.iou_thres = float(new_iou_thres)

        # Get image size parameter
        new_imgsz = self.parameter_display.path_label4.text()
        opt.imgsz = [int(new_imgsz), int(new_imgsz)]

        # Whether to save txt results
        if self.parameter_display.path_label5.text() == "Yes":
            # Save txt results
            opt.save_txt = True
        else:
            opt.save_txt = False

        # Whether to save conf results
        if self.parameter_display.path_label6.text() == "Yes":
            # Save conf results
            opt.save_conf = True
        else:
            opt.save_conf = False

        # Whether to save crop results
        if self.parameter_display.path_label7.text() == "Yes":
            # Save crop results
            opt.save_crop = True
        else:
            opt.save_crop = False

        # 创建并启动检测线程
        self.detection_thread = DetectionThread(run_detection_internal, opt)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error_occurred.connect(self.show_error)
        self.detection_thread.result_ready.connect(self.on_detection_result)
        self.detection_thread.start()
        
    def on_detection_finished(self):
        print("Detection finished")
        
    def on_detection_result(self, new_results):
        self.operation_frame.new_detection_result(new_results)
        
    def show_error(self, error_msg):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Detection Error")
        msg.setText("An error occurred during detection:")
        msg.setDetailedText(error_msg)
        msg.exec_()
        
    def closeEvent(self, event):
        # 确保在窗口关闭时停止所有线程
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())